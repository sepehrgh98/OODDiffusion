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

    data_dir = "./data/image_test"  # Update this with your dataset directory

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
    dataset = TestDataset(
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

    save_path = "./results/dataset_test"

    # Iterate through a few batches of data to test
    for batch_idx, (img, label) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
  
        # Visualize a few images
        for i in range(img.size(0)):
            image = img[i].detach().squeeze() 
            image = image.permute(1, 2, 0) 
            # if real.dtype == torch.float32:
            image = (image * 255).clamp(0, 255).byte()
            image = image.numpy()

            path = os.path.join(save_path,f"{batch_idx}_{label[i]}.png")
            print(path)

            Image.fromarray(image).save(path)

        
    # for batch_idx, (r, s, g) in enumerate(dataloader):
    #     print(f"Batch {batch_idx + 1}")
  
    #     # Visualize a few images
    #     for i in range(r.size(0)):


    #         real = r[i].detach().squeeze() 
    #         syn = s[i].detach().squeeze() 
    #         gt = g[i].detach().squeeze() 


    #         real = real.permute(1, 2, 0) 
    #         syn = syn.permute(1, 2, 0) 
    #         gt = gt.permute(1, 2, 0) 





    #         # if real.dtype == torch.float32:
    #         real = (real * 255).clamp(0, 255).byte()

    #         # if syn.dtype == torch.float32:
    #         syn = (syn * 255).clamp(0, 255).byte()


    #         # if gt.dtype == torch.float32:
    #         gt = (gt * 255).clamp(0, 255).byte()



    #         real = real.numpy()
    #         syn = syn.numpy()
    #         gt = gt.numpy()



    #         Image.fromarray(real).save(real_save_path)
    #         # Image.fromarray(syn).save(syn_save_path)
    #         # Image.fromarray(gt).save(gt_save_path)
    #         break
    #     if batch_idx == 0:
    #         break


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
