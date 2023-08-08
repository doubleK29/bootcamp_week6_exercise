import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

IMAGE_PATH = "cat100.jpg"
SAVED_IMAGES_DIR = "saved_images"

# Layer types for hooking
LAYER_TYPES = (
    nn.Conv2d,
    nn.ConvTranspose2d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.SiLU,
    nn.BatchNorm2d,
)


def layer_hook(module, input, output, layer_name):
    layer_outputs[layer_name] = output


def hook_layers(model):
    layer_hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, LAYER_TYPES):
            hook = layer.register_forward_hook(
                lambda module, input, output, name=name: layer_hook(
                    module, input, output, name
                )
            )
            layer_hooks.append(hook)
    return layer_hooks


def save_images(layer_outputs):
    for layer_name, output in tqdm(layer_outputs.items(), desc="Saving hooked images"):
        layer_dir = os.path.join(SAVED_IMAGES_DIR, layer_name)
        os.makedirs(layer_dir, exist_ok=True)

        num_channels = min(output.size(1), 20)
        channel_images = output[0, :num_channels].clone().detach().cpu()

        channel_images = (channel_images - channel_images.min()) / (
            channel_images.max() - channel_images.min()
        )

        channel_images = channel_images.numpy()

        for channel_idx in range(num_channels):
            channel_image = channel_images[channel_idx]
            channel_image_path = os.path.join(layer_dir, f"{channel_idx}.jpg")
            plt.imsave(channel_image_path, channel_image, cmap="viridis")
            # print(
            #     f"Saved image for layer '{layer_name}', channel {channel_idx} \
            #         at \'{channel_image_path}'"
            # )


def main():
    # Load YOLOv5s model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.eval()

    global layer_outputs
    layer_outputs = {}

    # Attach hooks to layers
    layer_hooks = hook_layers(model)

    # Inference
    model(IMAGE_PATH)

    # Detach the hooks after inference
    for hook in layer_hooks:
        hook.remove()

    # Directory to save images
    if os.path.exists(SAVED_IMAGES_DIR):
        shutil.rmtree(SAVED_IMAGES_DIR)
    os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)
    save_images(layer_outputs)
    print("Number of hooked layers:", len(layer_outputs))


if __name__ == "__main__":
    main()
