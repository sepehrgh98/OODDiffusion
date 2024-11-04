import torchvision.transforms as transforms
import math

from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


from dataset.utils import center_crop_arr, random_crop_arr

from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class HybridDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        out_size: int,
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: list,
        kernel_prob: list,
        blur_sigma: list,
        downsample_range: list,
        noise_range: list,
        jpeg_range: list,
        valid_extensions: list = [".png", ".jpg", ".jpeg"],
    ) -> "HybridDataset":
        super(HybridDataset, self).__init__()

        print("[INFO] Initializing HybridDataset...")

        self.data_dir = data_dir
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"], f"Invalid crop_type: {self.crop_type}"

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
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])

        # Load images
        self._check_dir()
        self.image_pairs = self._load_image_pairs()

        # Generate the final dataset with either LQ or Synthetic, but not both
        self.final_images, self.final_labels = self._select_images()
        print("[INFO] HybridDataset initialization complete.\n")

    def _check_dir(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"[ERROR] The directory '{self.data_dir}' does not exist.")
    
        for dataset_name in os.listdir(self.data_dir):
            dataset_path = os.path.join(self.data_dir, dataset_name)
            
            if os.path.isdir(dataset_path):
                lq_path = os.path.join(dataset_path, 'lq')
                hq_path = os.path.join(dataset_path, 'hq')
                
                if not os.path.isdir(lq_path) or not os.path.isdir(hq_path):
                    raise FileNotFoundError(
                        f"[ERROR] The dataset '{dataset_name}' is missing required 'lq' or 'hq' directories.\n"
                        f"Valid structure:\n"
                        f"{self.data_dir}/\n"
                        f"  ├── {dataset_name}/\n"
                        f"  │   ├── lq/\n"
                        f"  │   └── hq/"
                    )
            else:
                raise NotADirectoryError(f"[ERROR] Expected a dataset directory but found '{dataset_name}', which is not a directory.")

        print("[INFO] Dataset structure verified successfully.")

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
                raise ValueError(f"[ERROR] Mismatch in the number of images in 'lq' and 'hq' folders for dataset '{dataset_name}'.")

            # Pair the images by their filenames
            for lq_image, hq_image in zip(lq_images, hq_images):
                lq_image_path = os.path.join(lq_path, lq_image)
                hq_image_path = os.path.join(hq_path, hq_image)
                image_pairs.append((lq_image_path, hq_image_path))

        print(f"[INFO] Loaded {len(image_pairs)} image pairs successfully.")
        return image_pairs

    def _select_images(self):
        """
        Generate synthetic images from HQ images and select either LQ or Syn images.
        """
        final_images = []
        final_labels = []

        print("[INFO] Generating synthetic images and selecting final dataset...")
        for idx, (lq_path, hq_path) in enumerate(self.image_pairs):
            # Load HQ image to generate synthetic image
            hq_image = Image.open(hq_path).convert("RGB")
            
            hq_image = self._clip(hq_image)


            # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            hq_image = (hq_image[..., ::-1] / 255.0).astype(np.float32)


            # Apply degradations to create a synthetic version
            synthetic_image = self._apply_degradations(hq_image)

            # Randomly select either LQ or synthetic image
            if random.choice([True, False]):
                # Select LQ image
                final_images.append(lq_path)
                final_labels.append(0)  # Label for Real
            else:
                # Select synthetic image (store as numpy array)
                final_images.append(synthetic_image)
                final_labels.append(1)  # Label for Synthetic

        # Ensure equal number of LQ and synthetic images
        lq_count = sum(1 for label in final_labels if label == 0)
        syn_count = len(final_labels) - lq_count

        print(f"[INFO] Number of Real images selected: {lq_count}")
        print(f"[INFO] Number of Synthetic images selected: {syn_count}")

        if lq_count > syn_count:
            excess = lq_count - syn_count
            self._balance_dataset(final_images, final_labels, excess, 0)
        elif syn_count > lq_count:
            excess = syn_count - lq_count
            self._balance_dataset(final_images, final_labels, excess, 1)

        return final_images, final_labels
    
    
    def _clip(self, img):
        if self.crop_type != "none":
            if img.height == self.out_size and img.width == self.out_size:
                img = np.array(img)
            else:
                if self.crop_type == "center":
                    img = center_crop_arr(img, self.out_size)
                elif self.crop_type == "random":
                    img = random_crop_arr(img, self.out_size, min_crop_frac=0.7)
        else:
            assert img.height == self.out_size and img.width == self.out_size
            img = np.array(img)
        return img

    
    def _balance_dataset(self, final_images, final_labels, excess, label):
        """
        Remove excess images to ensure equal representation.
        """
        indices_to_remove = [i for i, l in enumerate(final_labels) if l == label]
        indices_to_remove = indices_to_remove[:excess]

        for idx in sorted(indices_to_remove, reverse=True):
            del final_images[idx]
            del final_labels[idx]


    def __getitem__(self, index):
        # Load LQ or synthetic image based on final selection
        if isinstance(self.final_images[index], str):
            img_path = self.final_images[index]
            image = Image.open(img_path).convert("RGB")
        else:
            image = self.final_images[index]


        # Preprocess image
        image = self._preprocess_image(np.array(image))

        # Get label
        label = self.final_labels[index]
        return image, label


    def _apply_degradations(self, img_gt):
        """
        Apply degradation operations to create a synthetic version.
        """
        h, w, _ = img_gt.shape
        img_lq = img_gt.copy()

        # Apply blur
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


        # Downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)


        # Add noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)


        # JPEG compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)

        # Resize back to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, [0, 1]
        img_lq = img_lq[..., ::-1].astype(np.float32)

        return img_lq

    def _preprocess_image(self, image):
        if image.dtype != np.uint8:  # Assuming image values are in range [0, 1]
          image = (image * 255).astype(np.uint8)
        
        
        image = self.preprocess(image)
        return image

    def __len__(self):
        return len(self.final_images)
