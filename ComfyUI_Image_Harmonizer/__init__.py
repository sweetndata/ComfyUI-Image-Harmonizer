import os
import torch
from PIL import Image
import torchvision.transforms.functional as tf
from .model import Harmonizer  # Harmonizer 모델 import (경로에 따라 수정 필요)
import logging

class HarmonizerNode:
    def __init__(self):
        self.harmonizer = None
        self.cuda = torch.cuda.is_available()
        self.load_model()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composite_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("harmonized_image",)
    FUNCTION = "harmonize_image"
    CATEGORY = "image"

    def load_model(self):
        try:
            self.harmonizer = Harmonizer()
            if self.cuda:
                self.harmonizer = self.harmonizer.cuda()
            model_path = os.path.join(os.path.dirname(__file__), 'harmonizer.pth')
            self.harmonizer.load_state_dict(torch.load(model_path), strict=True)
            self.harmonizer.eval()
        except Exception as e:
            logging.exception("Failed to load Harmonizer model.")
            raise e

    def harmonize_image(self, composite_image, mask_image, *args, **kwargs):
        try:
            # Convert composite_image to (N, C, H, W)
            composite_image = composite_image.permute(0, 3, 1, 2)  # From (N, H, W, C) to (N, C, H, W)
            if composite_image.shape[1] != 3:
                raise ValueError(f"Expected composite_image to have 3 channels (RGB), but got {composite_image.shape[1]} channels.")

            # Convert mask_image to (N, C, H, W) and ensure single channel
            mask_image = mask_image.permute(0, 3, 1, 2)  # From (N, H, W, C) to (N, C, H, W)
            if mask_image.shape[1] != 1:
                mask_image = mask_image[:, :1, :, :]  # Keep only the first channel

            # Move to CUDA if available
            if self.cuda:
                composite_image, mask_image = composite_image.cuda(), mask_image.cuda()

            # Perform harmonization
            with torch.no_grad():
                arguments = self.harmonizer.predict_arguments(composite_image, mask_image)
                harmonized = self.harmonizer.restore_image(composite_image, mask_image, arguments)[-1]

            # Convert harmonized tensor back to PIL Image
            harmonized_image = (harmonized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
            harmonized_pil = Image.fromarray(harmonized_image)

            # Use PIL.Image size for logging or other operations
            image_size = harmonized_pil.size  # (Width, Height)
            harmonized_pil = tf.to_tensor(harmonized_pil)
            harmonized_pil = harmonized_pil.unsqueeze(dim=0)
            harmonized_pil = harmonized_pil.permute(0, 2, 3, 1)  # From (N, H, W, C) to (N, C, H, W)


            # Return the harmonized image as output
            return (harmonized_pil,)
        except Exception as e:
            logging.exception("Error during harmonization.")
            raise e




NODE_CLASS_MAPPINGS = {
    "harmonizer": HarmonizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "harmonizer": "Image Harmonizer",
}
