import torchvision.transforms as transforms
import math

from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import torch


from dataset.utils import center_crop_arr, random_crop_arr, crop_image

from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# class HybridDataset(Dataset):
class OOD_Dataset(Dataset):
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
    ) -> "OOD_Dataset":
        super(OOD_Dataset, self).__init__()

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

        self.nReal = 0
        self.nSyn = 0
        
        # Define the preprocessing steps
        self.preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        self._run()



    def _run(self):
        print("[INFO] Initializing OOD_Dataset...")
        # Load images
        self._check_dir()
        self.image_pairs = self._load_image_pairs()

        # Generate the final dataset with either LQ or Synthetic, but not both
        # self.final_images, self.final_labels = self._select_images()
        self.final_image_pack = self._select_images()
        print("[INFO] OOD_Dataset initialization complete.\n")


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

            is_empty = not any(os.scandir(hq_path))

            if is_empty:
                hq_path = lq_path


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
        final_image_pack = []
        # final_syn_images = []
        # final_labels = []

        print("[INFO] Generating synthetic images and selecting final dataset...")
        for idx, (lq_path, hq_path) in enumerate(self.image_pairs):
            # Load HQ image to generate synthetic image

            try:
                hq_image = Image.open(hq_path).convert("RGB")
                lq_image = Image.open(lq_path).convert("RGB")

            except Exception as e:
                print(f"Corrupted file: {hq_path} or {lq_path}, Error: {e}")
                continue


            hq_image = Image.open(hq_path).convert("RGB")
            lq_image = Image.open(lq_path).convert("RGB")
            
            hq_image = self._clip(hq_image)
            lq_image = self._clip(lq_image)


            # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            hq_image = (hq_image[..., ::-1] / 255.0).astype(np.float32)
            lq_image = (lq_image[..., ::-1] / 255.0).astype(np.float32)


            # Apply degradations to create a synthetic version
            synthetic_image = self._apply_degradations(hq_image)

            final_image_pack.append((lq_image, synthetic_image, hq_image))

        return final_image_pack
    
    
    def _clip(self, img):
        if self.crop_type != "none":
            if img.height == self.out_size and img.width == self.out_size:
                img = np.array(img)
            else:
                if self.crop_type == "center":
                    img = center_crop_arr(img, self.out_size)
                elif self.crop_type == "random":
                    img = random_crop_arr(img, self.out_size, min_crop_frac=0.7)
            # img = crop_image(img, 1200, 700 ,self.out_size)
        else:
            #assert img.height == self.out_size and img.width == self.out_size
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

        real, syn, gt = self.final_image_pack[index]

        # Preprocess image
        real = self._preprocess_image(np.array(real))
        syn = self._preprocess_image(np.array(syn))
        gt = self._preprocess_image(np.array(gt))



        return real, syn, gt

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
        return len(self.final_image_pack)


# class HybridDataset_non_observe(OOD_Dataset):
#     def __getitem__(self, index):

#         real, syn, gt = self.final_image_pack[index]
#         final_image = real
#         label = 0


#         final_image = self._preprocess_image(np.array(final_image))

#         return final_image, label
    


# class TestDataset(OOD_Dataset):
class OOD_Inference(OOD_Dataset):

    def _run(self):
        print("[INFO] Initializing HybridDataset...")
        # Load images
        self._check_dir()

        self.image_pairs = self._load_image_pairs()

        # Generate the final dataset
        self.final_image_pack = self._select_images()
        print("[INFO] HybridDataset initialization complete.\n")

    def _check_dir(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"[ERROR] The directory '{self.data_dir}' does not exist.")
        
        
        dataset_types = ['IDDataset', 'OODDataset']
        
        for dataset_type in dataset_types:
            # Path to IDDataset and OODDataset
            type_path = os.path.join(self.data_dir, dataset_type)
            
            if not os.path.isdir(type_path):
                raise FileNotFoundError(f"[ERROR] Missing '{dataset_type}' directory under '{self.data_dir}'.")
            
            # Iterate over each dataset inside the IDDataset and OODDataset folders
            for dataset_name in os.listdir(type_path):
                dataset_path = os.path.join(type_path, dataset_name)

                if os.path.isdir(dataset_path):
                    lq_path = os.path.join(dataset_path, 'lq')
                    hq_path = os.path.join(dataset_path, 'hq')

                    if not os.path.isdir(lq_path) or not os.path.isdir(hq_path):
                        raise FileNotFoundError(
                            f"[ERROR] The dataset '{dataset_name}' in '{dataset_type}' is missing required 'lq' or 'hq' directories.\n"
                            f"Valid structure:\n"
                            f"{self.data_dir}/\n"
                            f"  ├── {dataset_type}/\n"
                            f"  │   ├── {dataset_name}/\n"
                            f"  │   │   ├── lq/\n"
                            f"  │   │   └── hq/"
                        )
                else:
                    raise NotADirectoryError(f"[ERROR] Expected a dataset directory but found '{dataset_name}' in '{dataset_type}', which is not a directory.")
                
        print("[INFO] Dataset structure verified successfully.")

    def _load_image_pairs(self) -> list:
        image_pairs = {
            'IDDataset': [],
            'OODDataset': []
        }

        dataset_types = ['IDDataset', 'OODDataset']
        
        for dataset_type in dataset_types:
            type_path = os.path.join(self.data_dir, dataset_type)
            
            if not os.path.isdir(type_path):
                raise FileNotFoundError(f"[ERROR] '{dataset_type}' directory not found in '{self.data_dir}'.")

            # Iterate through all datasets in either IDDataset or OODDataset
            for dataset_name in os.listdir(type_path):
                dataset_path = os.path.join(type_path, dataset_name)
                
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
                    raise ValueError(f"[ERROR] Mismatch in the number of images in 'lq' and 'hq' folders for dataset '{dataset_name}' in '{dataset_type}'.")

                # Pair the images by their filenames
                for lq_image, hq_image in zip(lq_images, hq_images):
                    lq_image_path = os.path.join(lq_path, lq_image)
                    hq_image_path = os.path.join(hq_path, hq_image)
                    image_pairs[dataset_type].append((lq_image_path, hq_image_path))

        print(f"[INFO] Loaded {len(image_pairs['IDDataset'])} image pairs for IDDataset successfully.")
        print(f"[INFO] Loaded {len(image_pairs['OODDataset'])} image pairs for OODDataset successfully.")
        return image_pairs
    
    def _apply_degradations(self, img, deg_type:str):
        """
        Apply degradation operations to create a synthetic version.
        """
        h, w, _ = img.shape
        img_lq = img.copy()

        if deg_type == "blur":
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
        elif deg_type == "Downsample":
            # Downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)

        elif deg_type == "JPEG_compression":
            # JPEG compression
            if self.jpeg_range is not None:
                img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        elif deg_type == "noise":
            # Add noise
            if self.noise_range is not None:
                img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
            

        # Resize back to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, [0, 1]
        img_lq = img_lq[..., ::-1].astype(np.float32)

        return img_lq


    def _select_images(self):
        """
        Generate synthetic images from IDDataset.
        """
        final_image_pack = []

        print("[INFO] Generating synthetic images and selecting final dataset...")

        degradations = ["blur", "Downsample", "JPEG_compression", "noise"]
        selected_degradations = random.sample(degradations, 3)


        ID_list = self.image_pairs["IDDataset"]
        OOD_list = self.image_pairs["OODDataset"]

        # In Distribution image

        for _ , hq_img in ID_list:
            # we just need hq image
            ID_hq_image = Image.open(hq_img).convert("RGB")
            ID_hq_image = self._clip(ID_hq_image)

            # shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            ID_hq_image = (ID_hq_image[..., ::-1] / 255.0).astype(np.float32)
            
            # apply in distribution degradation
            for deg in selected_degradations:

                ID_syn_image = self._apply_degradations(ID_hq_image, deg)

                # Label
                label = 1

                final_image_pack.append((ID_syn_image, ID_hq_image, label))


        # Out of Distribution image

        for lq_img, hq_img in OOD_list:
            OOD_lq_image = Image.open(lq_img).convert("RGB")
            OOD_lq_image = self._clip(OOD_lq_image)

            ID_hq_image = Image.open(hq_img).convert("RGB")
            ID_hq_image = self._clip(ID_hq_image)

            # shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            ID_hq_image = (ID_hq_image[..., ::-1] / 255.0).astype(np.float32)

            # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            OOD_lq_image = (OOD_lq_image[..., ::-1] / 255.0).astype(np.float32)

            # Label
            label = 0

            final_image_pack.append((OOD_lq_image, ID_hq_image,label))


        random.shuffle(final_image_pack)

        return final_image_pack
    
    def __getitem__(self, index):

        img, gt, label = self.final_image_pack[index]

        # Preprocess image
        img = self._preprocess_image(np.array(img))
        gt = self._preprocess_image(np.array(gt))

        return img, gt, label

   
# class CSDataset(OOD_Dataset):
# Comparitive Study dataset
class OOD_CS_Dataset(OOD_Dataset):
    def _run(self):
        print("[INFO] Initializing HybridDataset...")
        # Load images
        self._check_dir()

        self.image_pairs = self._load_image_pairs()

        # Generate the final dataset
        self.final_image_pack = self._select_images()
        print("[INFO] HybridDataset initialization complete.\n")

    def _check_dir(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"[ERROR] The directory '{self.data_dir}' does not exist.")
        
        
        dataset_types = ['IDDataset', 'OODDataset']
        
        for dataset_type in dataset_types:
            # Path to IDDataset and OODDataset
            type_path = os.path.join(self.data_dir, dataset_type)
            
            if not os.path.isdir(type_path):
                raise FileNotFoundError(f"[ERROR] Missing '{dataset_type}' directory under '{self.data_dir}'.")
            
            # Iterate over each dataset inside the IDDataset and OODDataset folders
            for dataset_name in os.listdir(type_path):
                dataset_path = os.path.join(type_path, dataset_name)

                if os.path.isdir(dataset_path):
                    lq_path = os.path.join(dataset_path, 'lq')
                    hq_path = os.path.join(dataset_path, 'hq')

                    if not os.path.isdir(lq_path) or not os.path.isdir(hq_path):
                        raise FileNotFoundError(
                            f"[ERROR] The dataset '{dataset_name}' in '{dataset_type}' is missing required 'lq' or 'hq' directories.\n"
                            f"Valid structure:\n"
                            f"{self.data_dir}/\n"
                            f"  ├── {dataset_type}/\n"
                            f"  │   ├── {dataset_name}/\n"
                            f"  │   │   ├── lq/\n"
                            f"  │   │   └── hq/"
                        )
                else:
                    raise NotADirectoryError(f"[ERROR] Expected a dataset directory but found '{dataset_name}' in '{dataset_type}', which is not a directory.")
                
        print("[INFO] Dataset structure verified successfully.")

    def _load_image_pairs(self) -> list:
        image_pairs = {
            'IDDataset': [],
            'OODDataset': []
        }

        dataset_types = ['IDDataset', 'OODDataset']
        
        for dataset_type in dataset_types:
            type_path = os.path.join(self.data_dir, dataset_type)
            
            if not os.path.isdir(type_path):
                raise FileNotFoundError(f"[ERROR] '{dataset_type}' directory not found in '{self.data_dir}'.")

            # Iterate through all datasets in either IDDataset or OODDataset
            for dataset_name in os.listdir(type_path):
                dataset_path = os.path.join(type_path, dataset_name)
                
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
                    raise ValueError(f"[ERROR] Mismatch in the number of images in 'lq' and 'hq' folders for dataset '{dataset_name}' in '{dataset_type}'.")

                # Pair the images by their filenames
                for lq_image, hq_image in zip(lq_images, hq_images):
                    lq_image_path = os.path.join(lq_path, lq_image)
                    hq_image_path = os.path.join(hq_path, hq_image)
                    image_pairs[dataset_type].append((lq_image_path, hq_image_path, dataset_name))

        print(f"[INFO] Loaded {len(image_pairs['IDDataset'])} image pairs for IDDataset successfully.")
        print(f"[INFO] Loaded {len(image_pairs['OODDataset'])} image pairs for OODDataset successfully.")
        return image_pairs
    
    def _select_images(self):
        """
        Generate synthetic images from IDDataset.
        """
        final_image_pack = []

        print("[INFO] Generating synthetic images and selecting final dataset...")


        ID_list = self.image_pairs["IDDataset"]
        OOD_list = self.image_pairs["OODDataset"]

        # In Distribution image

        for _ , hq_img, dataset_n in ID_list:
            # we just need hq image
            ID_hq_image = Image.open(hq_img).convert("RGB")
            ID_hq_image = self._clip(ID_hq_image)

            # shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            ID_hq_image = (ID_hq_image[..., ::-1] / 255.0).astype(np.float32)
            
            # apply in distribution degradation
            ID_syn_image = self._apply_degradations(ID_hq_image)
            
            # Label
            label = 1

            img_name = os.path.basename(hq_img)

            final_image_pack.append((ID_syn_image, ID_hq_image, label, dataset_n, img_name))


        # Out of Distribution image

        for lq_img, hq_img , dataset_n in OOD_list:
            OOD_lq_image = Image.open(lq_img).convert("RGB")
            OOD_lq_image = self._clip(OOD_lq_image)
            # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            
            OOD_lq_image = np.array(OOD_lq_image)
            OOD_lq_image = (OOD_lq_image[..., ::-1] / 255.0).astype(np.float32)

            OOD_hq_image = Image.open(hq_img).convert("RGB")
            OOD_hq_image = self._clip(OOD_hq_image)
            # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            OOD_hq_image = (OOD_hq_image[..., ::-1] / 255.0).astype(np.float32)

            # Label
            label = 0

            img_name = os.path.basename(lq_img)


            final_image_pack.append((OOD_lq_image, OOD_hq_image, label, dataset_n, img_name))


        # random.shuffle(final_image_pack)

        return final_image_pack

    def __getitem__(self, index):

        img, gt, label, dataset_n, img_name = self.final_image_pack[index]

        # Preprocess image
        img = self._preprocess_image(np.array(img))
        gt = self._preprocess_image(np.array(gt))

        # img = self._clip(img)
        # gt = self._clip(gt)


        return img, gt, label, dataset_n, img_name


