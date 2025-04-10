import os
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset.HybridDataset import HybridDataset, ISP_HybridDataset, TestDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def test_hybrid_dataset():

    data_dir = "./data/ComparitiveDataset/IDDatset/BSD100-x2"  # Update this with your dataset directory

    out_size = 256
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
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(len(dataset))
    
    # real_save_path = "./real.png"
    # syn_save_path = "./syn.png"
    # gt_save_path = "./qt.png"

    save_path_LQ = "./results/CDDataset/BSD100-x2/LQ"
    save_path_GT = "./results/CDDataset/BSD100-x2/GT"

    if not os.path.exists(save_path_LQ):
        os.makedirs(save_path_LQ)  # Create the directory
        print(f"Directory '{save_path_LQ}' created.")
    else:
        print(f"Directory '{save_path_LQ}' already exists.")

    if not os.path.exists(save_path_GT):
        os.makedirs(save_path_GT)  # Create the directory
        print(f"Directory '{save_path_GT}' created.")
    else:
        print(f"Directory '{save_path_GT}' already exists.")


    # Iterate through a few batches of data to test
    for batch_idx, (real, syn, gt, lq_name) in enumerate(dataloader):
  
        # Visualize a few images
        for i in range(syn.size(0)):
            syn_image = syn[i].detach().squeeze() 
            syn_image = syn_image.permute(1, 2, 0) 
            # if real.dtype == torch.float32:
            syn_image = (syn_image * 255).clamp(0, 255).byte()
            syn_image = syn_image.numpy()

            path_lq = os.path.join(save_path_LQ, f"{lq_name[0]}.png")
            Image.fromarray(syn_image).save(path_lq)

            gt_image = gt[i].detach().squeeze() 
            gt_image = gt_image.permute(1, 2, 0) 
            # if real.dtype == torch.float32:
            gt_image = (gt_image * 255).clamp(0, 255).byte()
            gt_image = gt_image.numpy()

            print(gt_image.shape, syn_image.shape)

            path_gt = os.path.join(save_path_GT, f"{lq_name[0]}.png")
            Image.fromarray(gt_image).save(path_gt)
            print(lq_name[0])

        


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
