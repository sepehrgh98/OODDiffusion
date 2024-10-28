import os
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset.HybridDataset import HybridDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def save_preprocessed_image(tensor, save_path):
    """
    Saves a preprocessed tensor image to a specified path.

    Args:
    - tensor (torch.Tensor): The preprocessed image tensor. Shape should be [3, H, W].
    - save_path (str): The file path to save the image to.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    
    # Ensure the tensor has the correct shape
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected tensor shape [3, H, W]")

    # Reverse the normalization to bring the tensor values back to the [0, 1] range
    reverse_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    
    tensor = reverse_normalize(tensor)

    # Clip the values to be in [0, 1] range (after reversing normalization)
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert the tensor to a PIL image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)

    # Save the image to the specified path
    image.save(save_path)

    print(f"Image saved at: {save_path}")

def save_preprocessed_batch(batch_tensor, save_prefix):
    """
    Saves each tensor in a batch as an individual image.

    Args:
    - batch_tensor (torch.Tensor): The batch tensor of shape [B, 3, H, W].
    - save_prefix (str): The prefix for the file paths where images will be saved.
    """
    if batch_tensor.ndim != 4:
        raise ValueError("Expected batch tensor with shape [B, 3, H, W]")

    batch_size = batch_tensor.shape[0]
    for i in range(batch_size):
        image_tensor = batch_tensor[i]
        save_path = f"{save_prefix}_{i}.png"  # Save each image with a unique name
        save_preprocessed_image(image_tensor, save_path)



def test_hybrid_dataset():
    # Test directory and parameters
    data_dir = "./test_dataset"  # Create a test dataset directory with the expected structure
    out_size = 224
    crop_type = "center"
    blur_kernel_size = 41
    kernel_list = ['iso', 'aniso']
    kernel_prob = [0.5, 0.5]
    blur_sigma = [0.8, 3.2]
    downsample_range = [2, 4]
    noise_range = [0, 15]
    jpeg_range = [30, 100]
    valid_extensions = [".png", ".jpg", ".jpeg"]

    # Create a HybridDataset instance
    dataset = HybridDataset(
        data_dir=data_dir,
        out_size=out_size,
        crop_type=crop_type,
        blur_kernel_size=blur_kernel_size,
        kernel_list=kernel_list,
        kernel_prob=kernel_prob,
        blur_sigma=blur_sigma,
        downsample_range=downsample_range,
        noise_range=noise_range,
        jpeg_range=jpeg_range,
        valid_extensions=valid_extensions,
    )

    # Create a DataLoader instance for the dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Iterate over the DataLoader to test the dataset
    for i, (gt, real, synthetic) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"Ground Truth (GT) Image Tensor Shape: {gt.shape}")
        print(f"Real LQ Image Tensor Shape: {real.shape}")
        print(f"Synthetic Image Tensor Shape: {synthetic.shape}")

        print(f"Shape of gt before saving: {gt.shape}")

        save_preprocessed_batch(gt, "gt")
        save_preprocessed_batch(real, "real")
        save_preprocessed_batch(synthetic, "synthetic")



        if i == 0:  # Limit to displaying the first batch for the test
            break

if __name__ == "__main__":
    # Before running this test, make sure to create a directory structure as expected by HybridDataset
    # ./test_dataset/
    #    ├── dataset1/
    #    │     ├── lq/
    #    │     │     ├── image1.jpg
    #    │     │     ├── image2.jpg
    #    │     └── hq/
    #    │           ├── image1.jpg
    #    │           ├── image2.jpg
    test_hybrid_dataset()
