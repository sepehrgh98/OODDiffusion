from torch import Tensor
import pyiqa
from typing import Dict
import torch
import numpy as np



def prepare_image(img):
    # Convert numpy array to torch tensor if needed

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    
    if img.dim() == 3:  # If image is [C, H, W]
        img = img.unsqueeze(0)
    
    # Normalize to [0, 1] range if it's not already
    if img.max() > 1:
        img = img / 255.0

    return img

def psnr(batch1: Tensor, batch2: Tensor) -> Dict:
    
    # Ensure both batches are of the same size
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
    
    psnr_metric = pyiqa.create_metric('psnr')

    result = []

    for img1, img2 in zip(batch1, batch2):
        img1 = prepare_image(img1)
        img2 = prepare_image(img2)
        psnr_value = psnr_metric(img1, img2).item()
        result.append(torch.tensor(psnr_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def ssim(batch1: Tensor, batch2: Tensor) -> Dict:
    
    # Ensure both batches are of the same size
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
    
    ssim_metric = pyiqa.create_metric('ssim')

    result = []

    for img1, img2 in zip(batch1, batch2):
        img1 = prepare_image(img1)
        img2 = prepare_image(img2)
        ssim_value = ssim_metric(img1, img2).item()
        result.append(torch.tensor(ssim_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def lpips(batch1: Tensor, batch2: Tensor) -> Dict:
    
    # Ensure both batches are of the same size
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
    
    lpips_metric = pyiqa.create_metric('lpips')

    result = []

    for img1, img2 in zip(batch1, batch2):
        img1 = prepare_image(img1)
        img2 = prepare_image(img2)
        lpips_value = lpips_metric(img1, img2).item()
        result.append(torch.tensor(lpips_value, dtype=torch.float32))


    result = torch.stack(result)

    return result



def brisque(batch: Tensor) -> Dict:
    
    brisque_metric = pyiqa.create_metric('brisque')

    result = []

    for img1, img2 in batch:
        img1 = prepare_image(img1)
        img2 = prepare_image(img2)
        brisque_value = brisque_metric(img1, img2).item()
        result.append(torch.tensor(brisque_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def nima(batch: Tensor) -> Dict:
    
    nima_metric = pyiqa.create_metric('nima')

    result = []

    for img1, img2 in batch:
        img1 = prepare_image(img1)
        img2 = prepare_image(img2)
        nima_value = nima_metric(img1, img2).item()
        result.append(torch.tensor(nima_value, dtype=torch.float32))


    result = torch.stack(result)

    return result

def niqe(batch: Tensor) -> Dict:
    
    niqe_metric = pyiqa.create_metric('niqe')

    result = []

    for img1, img2 in batch:
        img1 = prepare_image(img1)
        img2 = prepare_image(img2)
        niqe_value = niqe_metric(img1, img2).item()
        result.append(torch.tensor(niqe_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def musiq(batch: Tensor) -> Dict:
    
    musiq_metric = pyiqa.create_metric('musiq')

    result = []

    for img1, img2 in batch:
        img1 = prepare_image(img1)
        img2 = prepare_image(img2)
        musiq_value = musiq_metric(img1, img2).item()
        result.append(torch.tensor(musiq_value, dtype=torch.float32))

    result = torch.stack(result)

    return result


