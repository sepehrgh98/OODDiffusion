import random
import math
import torch

import numpy as np
from PIL import Image

# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# def crop_image(pil_image, x, y, crop_width, crop_height):
#     """
#     Crop a specific part of an image without downsampling.
    
#     Parameters:
#         pil_image (PIL.Image): Input image.
#         x (int): X-coordinate (left) of the crop area.
#         y (int): Y-coordinate (top) of the crop area.
#         crop_width (int): Width of the crop.
#         crop_height (int): Height of the crop.
    
#     Returns:
#         np.ndarray: Cropped image as a NumPy array.
#     """
#     # Ensure the crop is within image bounds
#     img_width, img_height = pil_image.size
#     if x + crop_width > img_width or y + crop_height > img_height:
#         raise ValueError("Crop dimensions exceed image boundaries.")

#     # Perform cropping
#     cropped_image = pil_image.crop((x, y, x + crop_width, y + crop_height))
    
#     return np.array(cropped_image)


import torch
import numpy as np
from PIL import Image

def crop_image(image, center_x, center_y, crop_size):
    """
    Crop a square region from a specified center of the image.
    
    Parameters:
    image (torch.Tensor, PIL.Image, or np.ndarray): Input image.
    center_x (int): X-coordinate of the square's center.
    center_y (int): Y-coordinate of the square's center.
    crop_size (int): Size of the square crop.
    
    Returns:
    torch.Tensor, PIL.Image, or np.ndarray: Cropped image.
    """
    half_crop = crop_size // 2

    if torch.is_tensor(image):  # If image is a PyTorch tensor
        print("#######",image.shape)
        if image.ndim == 3:  # (C, H, W) format
            _, h, w = image.shape
        elif image.ndim == 4:  # (B, C, H, W) format
            _, _, h, w = image.shape
        else:
            raise ValueError("Unsupported tensor shape. Expected (C, H, W) or (B, C, H, W).")

        # Define crop boundaries
        left = max(center_x - half_crop, 0)
        right = min(center_x + half_crop, w)
        top = max(center_y - half_crop, 0)
        bottom = min(center_y + half_crop, h)

        # Crop the tensor
        return image[..., top:bottom, left:right]

    elif isinstance(image, np.ndarray):  # If image is a NumPy array
        h, w = image.shape[:2]
        left = max(center_x - half_crop, 0)
        right = min(center_x + half_crop, w)
        top = max(center_y - half_crop, 0)
        bottom = min(center_y + half_crop, h)
        return image[top:bottom, left:right]

    elif isinstance(image, Image.Image):  # If image is a PIL image
        w, h = image.size
        left = max(center_x - half_crop, 0)
        right = min(center_x + half_crop, w)
        top = max(center_y - half_crop, 0)
        bottom = min(center_y + half_crop, h)
        return image.crop((left, top, right, bottom))

    else:
        raise TypeError("Unsupported image type. Must be torch.Tensor, np.ndarray, or PIL.Image.")





def stretch_image(image):
    # Stretch image values to span the full range [0, 1]
    min_val, max_val = image.min(), image.max()
    if min_val == max_val:
        return image  # Avoid divide by zero in case of a completely uniform image
    return (image - min_val) / (max_val - min_val)
