from typing import Iterable, Optional, Tuple, Union, List 

import os
import math
import json
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from PIL import Image
from einops import rearrange
from functools import partial
from timm.layers import DropPath
from dataclasses import dataclass
from torchvision import transforms
from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from flash_attn.bert_padding import pad_input
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func

logger = logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward_(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
    
    def forward(self, grid_shapes, spatial_merge_size=2):
        pos_ids = []
        s = spatial_merge_size
        for t, h, w in grid_shapes:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(h // s, s, w // s, s)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(h // s, s, w // s, s)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = torch.tensor(grid_shapes).max()
        rotary_pos_emb_full = self.forward_(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb


class FlashAttention(nn.Module):
    # https://github.com/Dao-AILab/flash-attention/blob/v0.2.8/flash_attn/flash_attention.py
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self._deterministic = True

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                          device=qkv.device)
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                qkv = qkv.squeeze()  # [1, n, h, d] -> [n, h, d]
                seqlens_in_batch = key_padding_mask.sum(dim=-1, dtype=torch.int32)
                max_seqlen_in_batch = seqlens_in_batch.max().item()
                cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_seqlen_in_batch, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, deterministic=self._deterministic
                )
                output = output.unsqueeze(0)
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm
    RMSNorm = FusedRMSNorm  # noqa
    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of RMSNorm')
except ImportError:  # using the normal RMSNorm
    pass
except Exception:
    logger.warning('discovered apex but it failed to load, falling back to RMSNorm')
    pass


@dataclass
class BaseModelOutputWithKwargs(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    kwargs: Optional[dict] = None


class UniViTARVisionConfig(PretrainedConfig):
    def __init__(
            self,
            resolution_mode="native",
            init_method="xavier",
            num_channels=3,
            patch_size=14,
            temporal_patch_size=2,
            image_size=1792,
            patch_dropout=0.0,
            attention_dropout=0.0,
            dropout=0.0,
            drop_path_rate=0.0,
            initializer_range=1e-10,
            num_hidden_layers=24,
            num_attention_heads=16,
            hidden_size=1024,
            intermediate_size=4224,
            patch_embedding_bias=True,
            qk_normalization=True,
            qkv_bias=False,
            initializer_factor=0.1,
            use_pre_norm=False,
            pe_type="rope2d",
            rope_theta=10000,
            spatial_merge_size=1,
            norm_type="RMSNorm",
            hidden_act='SwiGLU',
            use_flash_attn=True,
            layer_norm_eps=1e-6,
            min_tokens=576,
            max_tokens=16384,
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            relarge_ratio=1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.resolution_mode = resolution_mode
        self.init_method = init_method
        self.pe_type = pe_type
        self.rope_theta = rope_theta
        self.temporal_patch_size = temporal_patch_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.patch_dropout = patch_dropout
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads 
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.patch_embedding_bias = patch_embedding_bias
        self.qk_normalization = qk_normalization
        self.qkv_bias = qkv_bias
        self.initializer_factor = initializer_factor
        self.use_pre_norm = use_pre_norm
        self.norm_type = norm_type
        self.hidden_act = hidden_act
        self.use_flash_attn = use_flash_attn
        self.layer_norm_eps = layer_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.image_mean = image_mean
        self.image_std = image_std
        self.relarge_ratio = relarge_ratio

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)


class UniViTARImageTransform(object):
    def __init__(self, config):
        self.config = config
        self.resolution_mode = config.resolution_mode
        
        self.image_mean, self.image_std = config.image_mean, config.image_std
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.resize_factor = config.patch_size * config.spatial_merge_size * config.resize_factor
        self.relarge_ratio = config.relarge_ratio

        self.forced_transform = None
        self.min_pixels, self.max_pixels = None, None
        assert self.resolution_mode in ["native", "224", "378", "756"]
        if self.resolution_mode == "native":
            self.min_pixels = config.min_tokens * config.patch_size * config.patch_size
            self.max_pixels = config.max_tokens * config.patch_size * config.patch_size
        else:
            image_size = int(self.resolution_mode)
            self.forced_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                    self.convert_to_rgb,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std)
                ]
            )

    def __call__(self, images):
        
        if not isinstance(images, List):
            images = [images]  # shape of each image is [h, w, c]
        assert len(images) == 1 or len(images) % self.temporal_patch_size == 0

        if self.resolution_mode == "native":
            sample_num = 1 if len(images) == 1 else len(images) // self.temporal_patch_size
            min_pixels, max_pixels = self.min_pixels // sample_num, self.max_pixels // sample_num
            width, height = images[0].size  # (w, h)
            if self.relarge_ratio > 0 and self.relarge_ratio != 1:
                height, width = int(height * self.relarge_ratio), int(width * self.relarge_ratio)
            resized_height, resized_width = self.smart_resize(height, width, self.resize_factor, min_pixels, max_pixels)
            processed_images = []
            for image in images:
                image = self.convert_to_rgb(image)
                image = self.resize(image, size=(resized_height, resized_width), resample=Image.Resampling.BICUBIC)
                image = self.rescale(image, scale=1/255)
                image = self.normalize(image=image, mean=self.image_mean, std=self.image_std)
                processed_images.append(image)
            processed_images = np.array(processed_images)  # (num, h, w, c)
            processed_images = processed_images.transpose(0, 3, 1, 2)  # (num, c, h, w)
        else:
            processed_images = [self.forced_transform(image).numpy() for image in images]
            processed_images = np.array(processed_images)

        if processed_images.shape[0] == 1:
            processed_images = np.tile(processed_images, (self.temporal_patch_size, 1, 1, 1))

        return torch.from_numpy(processed_images)    
    
    @staticmethod
    def convert_to_rgb(image):
        if not isinstance(image, Image.Image):
            return image
        # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
        # for transparent images. The call to `alpha_composite` handles this case
        if image.mode == "RGB":
            return image
        image_rgba = image.convert("RGBA")
        background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, image_rgba)
        alpha_composite = alpha_composite.convert("RGB")
        return alpha_composite
    
    @staticmethod
    def resize(image, size, resample, return_numpy: bool = True) -> np.ndarray:
        """
        Resizes `image` to `(height, width)` specified by `size` using the PIL library.
        """
        if not len(size) == 2:
            raise ValueError("size must have 2 elements")
        assert isinstance(image, Image.Image)
        height, width = size
        resample = resample if resample is not None else Image.Resampling.BILINEAR
        # PIL images are in the format (width, height)
        resized_image = image.resize((width, height), resample=resample, reducing_gap=None)
        if return_numpy:
            resized_image = np.array(resized_image)
            resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
        return resized_image

    @staticmethod
    def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")
        rescaled_image = image * scale
        rescaled_image = rescaled_image.astype(dtype)
        return rescaled_image

    @staticmethod
    def normalize(image, mean, std) -> np.ndarray:
        """
        Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.
        image = (image - mean) / std
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array")
        num_channels = image.shape[-1]
        # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
        # We preserve the original dtype if it is a float type to prevent upcasting float16.
        if not np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32)
        if isinstance(mean, Iterable):
            if len(mean) != num_channels:
                raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
        else:
            mean = [mean] * num_channels
        mean = np.array(mean, dtype=image.dtype)
        if isinstance(std, Iterable):
            if len(std) != num_channels:
                raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
        else:
            std = [std] * num_channels
        std = np.array(std, dtype=image.dtype)
        image = (image - mean) / std
        return image
    
    @staticmethod
    def smart_resize(height, width, factor, min_pixels, max_pixels):
        """ 
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if height < factor or width < factor:
            if height < factor:
                ratio = factor / height
                height, width = factor, int(ratio * width) + 1
            if width < factor:
                ratio = factor / width
                width, height = factor, int(ratio * height) + 1
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar


class SwiGLU(nn.Module):
    def __init__(self, config: UniViTARVisionConfig):
        super().__init__()
        self.config = config
        self.inner_hidden_size = int(config.intermediate_size * 2 / 3)
        self.act = ACT2FN['silu']
        self.fc1 = nn.Linear(config.hidden_size, self.inner_hidden_size)
        self.fc2 = nn.Linear(self.inner_hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, self.inner_hidden_size)
        self.norm = RMSNorm(self.inner_hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(x)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(self.norm(hidden_states * self.fc3(x)))
        return hidden_states


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: UniViTARVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert config.use_flash_attn is True,  "FlashAttention must be used!"
        assert self.head_dim * self.num_heads == self.embed_dim
        
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(config.dropout)
        if self.config.qk_normalization:
            self.q_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps) 

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor: 
        key_padding_mask = kwargs.get("key_padding_mask", None)
        rotary_pos_emb = kwargs["rotary_pos_emb"]

        qkv = self.qkv(hidden_states)
        qkv = rearrange(qkv, '... (three h d) -> ... three h d', three=3, h=self.num_heads)
        bind_dim = qkv.dim() - 3
        target_dtype = qkv.dtype
        q, k, v = qkv.unbind(bind_dim)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        if self.config.qk_normalization:
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
        qkv = torch.stack([q, k, v], dim=bind_dim).to(target_dtype)
        context, _ = self.inner_attn(qkv, key_padding_mask=key_padding_mask, causal=False)
        
        outs = self.proj(rearrange(context, '... h d -> ... (h d)')) # input expected to be: [b s h d] or [s h d]
        outs = self.proj_drop(outs)

        return outs


class UniViTARVisionEmbeddings(nn.Module):
    def __init__(self, config: UniViTARVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.use_bias = config.patch_embedding_bias
        self.patch_embedding = nn.Conv3d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.kernel_size, stride=self.kernel_size, bias=self.use_bias)

    def forward(self, pixel_values: torch.FloatTensor, **kwargs) -> torch.Tensor:  
        pixel_values = pixel_values.view(-1, 3, *self.kernel_size)
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.view(1, -1, self.embed_dim)
        self.num_patches = embeddings.shape[1]
        return embeddings


class UniViTARVisionEncoderLayer(nn.Module):
    def __init__(self, config: UniViTARVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        assert config.hidden_act == "SwiGLU"

        self.attn = Attention(config)
        self.norm1 = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SwiGLU(config)
        
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states), **kwargs) * self.ls1)
        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)
        return hidden_states


class UniViTARVisionEncoder(nn.Module):
    """ Transformer encoder consisting of `config.num_hidden_layers` self attention layers. """
    def __init__(self, config: UniViTARVisionConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = True

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([UniViTARVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        if self.config.pe_type == "rope2d":
            head_dim = config.hidden_size // config.num_attention_heads
            self.rotary_pos_emb = VisionRotaryEmbedding2D(head_dim // 2, theta=self.config.rope_theta)
        else:
            raise NotImplementedError

    def forward(self, inputs_embeds, output_hidden_states = False, **kwargs):
        kwargs["rotary_pos_emb"] = self.rotary_pos_emb(kwargs["grid_shapes"], self.config.spatial_merge_size)
        
        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                encoder_layer_forward = partial(encoder_layer, **kwargs)
                layer_outputs = torch.utils.checkpoint.checkpoint(encoder_layer_forward, hidden_states, use_reentrant=True)
            else:
                layer_outputs = encoder_layer(hidden_states, **kwargs)
            hidden_states = layer_outputs
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutputWithKwargs(last_hidden_state=hidden_states, hidden_states=encoder_states, kwargs=kwargs)


class UniViTARVisionModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = UniViTARVisionConfig
    _no_split_modules = ['UniViTARVisionEncoderLayer']

    def __init__(self, model_config_path, *args, **kwargs):

        model_config_dict = json.load(open(model_config_path, "r", encoding="utf8"))
        config = UniViTARVisionConfig.from_dict(model_config_dict)

        super().__init__(config)
        self.config = config
        self.image_transform = UniViTARImageTransform(config)

        self.embeddings = UniViTARVisionEmbeddings(config)
        self.encoder = UniViTARVisionEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings
        
    def get_padding_mask(self, grid_shapes):
        seq_len = torch.tensor([int((np.prod(thw) - 1) + 1) for thw in grid_shapes])
        max_len = torch.max(seq_len)
        batch_size = len(grid_shapes)
        mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        range_matrix = torch.arange(max_len).expand(batch_size, max_len)
        mask = (range_matrix < seq_len.unsqueeze(1))
        return mask.cuda()

    def forward(self, pixel_values, output_hidden_states = False, **kwargs):
        assert len(pixel_values.shape) == 2, "(batch_num_tokens, hidden_size)"
        assert "grid_shapes" in kwargs, "grid_shapes: [(t, h, w), ..., (t, h, w)]"
        kwargs["key_padding_mask"] = self.get_padding_mask(kwargs["grid_shapes"])
        hidden_states = self.embeddings(pixel_values, **kwargs)
        encoder_outputs = self.encoder(hidden_states, output_hidden_states, **kwargs)
        last_hidden_state = encoder_outputs.last_hidden_state
        return last_hidden_state.squeeze(0)
    
    def data_patchify(self, input_data):
        t, c, h, w = input_data.shape
        grid_t, grid_h, grid_w = t // self.config.temporal_patch_size, h // self.config.patch_size, w // self.config.patch_size
        grid_size = c * self.config.temporal_patch_size * self.config.patch_size * self.config.patch_size
        input_data = input_data.reshape(
            grid_t, self.config.temporal_patch_size, c, 
            grid_h // self.config.spatial_merge_size, self.config.spatial_merge_size, self.config.patch_size, 
            grid_w // self.config.spatial_merge_size, self.config.spatial_merge_size, self.config.patch_size
        )
        input_data = input_data.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        input_data = input_data.reshape(grid_t * grid_h * grid_w, grid_size).contiguous()
        grid_shape = (grid_t, grid_h, grid_w)
        return input_data, grid_shape
