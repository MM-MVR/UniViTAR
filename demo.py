import torch
import argparse
import numpy as np
from PIL import Image
from modeling_univitar import UniViTARVisionModel


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default="UniViTAR-0.3B", help="模型名称")
    args = parser.parse_args()

    # Prepare Model
    model_name = f"weights/{args.model_name}"
    model = UniViTARVisionModel(f"{model_name}/config.json")
    print(f"Model Architecture: {model}")
    model.load_state_dict(torch.load(f"{model_name}/pytorch_model.bin", map_location="cpu"))
    model = model.to(torch.bfloat16).cuda()
    model.eval()

    # Prepare Data: [(3, H1, W1), ..., (3, Hn, Wn)] --> (N1+...+Nn, P)
    images = [Image.open(f"assets/demo1.jpg"), Image.open(f"assets/demo2.jpg")]
    data_inputs, grid_shapes = [], []
    for image in images:
        data_item = model.image_transform(image)
        input_data, grid_shape = model.data_patchify(data_item)
        data_inputs.append(input_data.to(torch.bfloat16).cuda())
        grid_shapes.append(grid_shape)
    data_inputs = torch.concatenate(data_inputs, dim=0)

    # Forward: (N1+...+Nn, P) --> [(N1, D), ..., (Nn, D)]
    with torch.no_grad(), torch.cuda.amp.autocast():
        data_embeds = model(pixel_values=data_inputs, grid_shapes=grid_shapes)
        data_embeds = data_embeds.split([np.prod(grid_shape) for grid_shape in grid_shapes])
        print(data_embeds[0].shape, data_embeds[1].shape)


if __name__ == '__main__':
    main()

# 运行命令: python3 demo.py --model-name UniViTAR-0.3B
