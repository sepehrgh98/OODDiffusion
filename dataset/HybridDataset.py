import torch
import torch.utils.data as data
import torchvision.transforms as transforms



from typing import Sequence

import os
import numpy as np
from PIL import Image
import math
import cv2




from dataset.utils import center_crop_arr, random_crop_arr, stretch_image
from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class HybridDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        out_size: int,
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        valid_extensions: Sequence[str] = [".png", ".jpg", ".jpeg"],
    ) -> "HybridDataset":
        super(HybridDataset, self).__init__()

        self.data_dir = data_dir
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]

        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.valid_extensions = valid_extensions
        
        # Define the preprocessing steps
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.Resize((out_size, out_size)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])

        # load images
        self._check_dir()
        self.image_pairs = self._load_image_pairs()


    def _check_dir(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"The directory '{self.data_dir}' does not exist.")
    
        for dataset_name in os.listdir(self.data_dir):
            dataset_path = os.path.join(self.data_dir, dataset_name)
            
            if os.path.isdir(dataset_path):
                lq_path = os.path.join(dataset_path, 'lq')
                hq_path = os.path.join(dataset_path, 'hq')
                
                if not os.path.isdir(lq_path) or not os.path.isdir(hq_path):
                    raise FileNotFoundError(
                        f"The dataset '{dataset_name}' is missing required 'lq' or 'hq' directories.\n"
                        f"Valid structure:\n"
                        f"{self.data_dir}/\n"
                        f"  ├── {dataset_name}/\n"
                        f"  │   ├── lq/\n"
                        f"  │   └── hq/"
                    )
            else:
                raise NotADirectoryError(f"Expected a dataset directory but found '{dataset_name}', which is not a directory.")

        print("Dataset structure verified successfully.")


    def _load_image_pairs(self) -> list:
        image_pairs = []
        
        for dataset_name in os.listdir(self.data_dir):
            dataset_path = os.path.join(self.data_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            
            lq_path = os.path.join(dataset_path, 'lq')
            hq_path = os.path.join(dataset_path, 'hq')

            # Get all files in lq and hq folders
            lq_images = [f for f in os.listdir(lq_path) if os.path.splitext(f)[1].lower() in self.valid_extensions]
            hq_images = [f for f in os.listdir(hq_path) if os.path.splitext(f)[1].lower() in self.valid_extensions]

            # Sort to ensure matching pairs (assuming matching filenames in lq and hq)
            lq_images.sort()
            hq_images.sort()

            # Check if the number of LQ and HQ images match
            if len(lq_images) != len(hq_images):
                raise ValueError(f"Mismatch in the number of images in 'lq' and 'hq' folders for dataset '{dataset_name}'.")

            # Pair the images by their filenames
            for lq_image, hq_image in zip(lq_images, hq_images):
                lq_image_path = os.path.join(lq_path, lq_image)
                hq_image_path = os.path.join(hq_path, hq_image)
                image_pairs.append((lq_image_path, hq_image_path))

        print(f"Loaded {len(image_pairs)} image pairs successfully.")
        return image_pairs


    def __getitem__(self, index):
        # Get LQ and HQ image paths
        lq_image_path, hq_image_path = self.image_pairs[index]

        # Load images using PIL
        try:
            hq_image = Image.open(hq_image_path).convert("RGB")
            lq_image = Image.open(lq_image_path).convert("RGB")
        except (IOError, ValueError) as e:
            print(f"Warning: Skipping corrupted image ({hq_image_path} or {lq_image_path}) due to error: {e}")
            # Instead of returning None, use a neighboring sample
            index = (index + 1) % len(self.image_pairs)
            lq_image_path, hq_image_path = self.image_pairs[index]
            hq_image = Image.open(hq_image_path).convert("RGB")
            lq_image = Image.open(lq_image_path).convert("RGB")

        
        # Crop
        if self.crop_type != "none":
            if hq_image.height == self.out_size and hq_image.width == self.out_size:
                hq_image = np.array(hq_image)
            else:
                if self.crop_type == "center":
                    hq_image = center_crop_arr(hq_image, self.out_size)
                elif self.crop_type == "random":
                    hq_image = random_crop_arr(hq_image, self.out_size, min_crop_frac=0.7)
        else:
            assert hq_image.height == self.out_size and hq_image.width == self.out_size
            hq_image = np.array(hq_image)

        
        
        hq_image = np.array(hq_image)
        lq_image = np.array(lq_image)

        hq_image = (hq_image[..., ::-1] / 255.0).astype(np.float32)
        lq_image = (lq_image[..., ::-1] / 255.0).astype(np.float32)

        # Apply degradations to generate synthetic LQ image (degraded HQ)
        synthetic_image = self._apply_degradations(hq_image)

        # print(f"Before pre process/lq_image - min: {lq_image.min()}, max: {lq_image.max()}")
        # print(f"Before pre process/hq_image - min: {hq_image.min()}, max: {hq_image.max()}")
        # print(f"Before pre process/synthetic_image - min: {synthetic_image.min()}, max: {synthetic_image.max()}")

        
        
        # Preprocess images to match required format
        lq_image = self._preprocess_image(lq_image)
        hq_image = self._preprocess_image(hq_image)
        synthetic_image = self._preprocess_image(synthetic_image)

        # print(f"After pre process/lq_image - min: {lq_image.min()}, max: {lq_image.max()}")
        # print(f"After pre process/hq_image - min: {hq_image.min()}, max: {hq_image.max()}")
        # print(f"After pre process/synthetic_image - min: {synthetic_image.min()}, max: {synthetic_image.max()}")


        return hq_image, lq_image, synthetic_image

    def _apply_degradations(self, img_gt):
        # print("-------------------------------------")
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        # img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        h, w, _ = img_gt.shape
        img_lq = img_gt.copy()


        # Blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_lq, -1, kernel)
        # img_lq = np.clip(img_lq, 0, 1)
        # print(f"After Blur - min: {img_lq.min()}, max: {img_lq.max()}")


        # Downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # img_lq = np.clip(img_lq, 0, 1)
        # print(f"After Downsample - min: {img_lq.min()}, max: {img_lq.max()}")

        # Add noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
            # img_lq = np.clip(img_lq, 0, 1)

        
        # print(f"After Noise - min: {img_lq.min()}, max: {img_lq.max()}")

        
        # JPEG compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
            # img_lq = np.clip(img_lq, 0, 1)

        # print(f"After JPEG Compression - min: {img_lq.min()}, max: {img_lq.max()}")

        
        # Resize back to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        # img_lq = np.clip(img_lq, 0, 1)
        
        # print(f"After Resize Back - min: {img_lq.min()}, max: {img_lq.max()}")

        # # Apply contrast stretching
        # img_lq = stretch_image(img_lq)
        # print(f"After Stretching - min: {img_lq.min()}, max: {img_lq.max()}")

        # BGR to RGB, [0, 1]
        # img_lq = img_lq[..., ::-1].astype(np.float32)

        return img_lq


    def _preprocess_image(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array")
        
        image = self.preprocess(image)
        return image



    def __len__(self):
        return len(self.image_pairs)



 