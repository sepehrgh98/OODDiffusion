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

def test_hybrid_dataset():

    data_dir = "./test_dataset"  # Update this with your dataset directory

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

    # Instantiate the HybridDataset
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
        valid_extensions = valid_extensions
    )

    # Create a DataLoader to iterate through the dataset
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Iterate through a few batches of data to test
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")

        # Visualize a few images
        for i in range(images.size(0)):
            img = images[i].permute(1, 2, 0).numpy()  # Convert tensor to numpy format for plotting
            img = (img * 0.229 + 0.485)  # De-normalize (Note: adapt based on your normalization)
            img = img.clip(0, 1)  # Clip values to [0, 1] for visualization

            plt.imshow(img)
            plt.title(f"Label: {labels[i].item()}")
            plt.axis("off")
            plt.show()

        # Exit after visualizing the first batch (optional)
        if batch_idx == 0:
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
