from torch import Tensor
import pyiqa
from typing import Dict
import torch
import numpy as np


def psnr(batch1: Tensor, batch2: Tensor) -> Dict:
    
    # Ensure both batches are of the same size
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
    
    psnr_metric = pyiqa.create_metric('psnr')

    result = []

    for img1, img2 in zip(batch1, batch2):
        psnr_value = psnr_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()
        result.append(torch.tensor(psnr_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def ssim(batch1: Tensor, batch2: Tensor) -> Dict:
    
    # Ensure both batches are of the same size
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
   
    ssim_metric = pyiqa.create_metric('ssim')
    result = []

    for img1, img2 in zip(batch1, batch2):
        ssim_value = ssim_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()
        result.append(torch.tensor(ssim_value, dtype=torch.float32))

    result = torch.stack(result)

    return result



def lpips(batch1: Tensor, batch2: Tensor) -> Dict:
    
    # Ensure both batches are of the same size
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
    
    lpips_metric = pyiqa.create_metric('lpips')
    result = []

    for img1, img2 in zip(batch1, batch2):
        lpips_value = lpips_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()
        result.append(torch.tensor(lpips_value, dtype=torch.float32))


    result = torch.stack(result)

    return result



def brisque(batch: Tensor) -> Dict:
    
    brisque_metric = pyiqa.create_metric('brisque')
    result = []

    for img in batch:
        # img = (img * 255).clamp(0, 255).to(torch.uint8)
        brisque_value = brisque_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(brisque_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def clipiqa(batch: Tensor) -> Dict:
    
    clipiqa_metric = pyiqa.create_metric('clipiqa')
    result = []

    for img1 in batch:
        clipiqa_value = clipiqa_metric(img1.unsqueeze(0)).item()
        result.append(torch.tensor(clipiqa_value, dtype=torch.float32))

    result = torch.stack(result)

    return result


def nima(batch: Tensor) -> Dict:

    nima_metric = pyiqa.create_metric('nima')
    result = []

    for img in batch:
        nima_value = nima_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(nima_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def niqe(batch: Tensor) -> Dict:
    
    niqe_metric = pyiqa.create_metric('niqe')
    result = []

    for img in batch:
        niqe_value = niqe_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(niqe_value, dtype=torch.float32))
        
    result = torch.stack(result)

    return result

def musiq(batch: Tensor) -> Dict:
   
    musiq_metric = pyiqa.create_metric('musiq')
    result = []

    for img in batch:
        musiq_value = musiq_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(musiq_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def musiq_koniq(batch: Tensor) -> Dict:
   
    musiq_koniq_metric = pyiqa.create_metric('musiq', dataset='koniq')
    result = []

    for img in batch:
        musiq_koniq_value = musiq_koniq_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(musiq_koniq_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def musiq_ava(batch: Tensor) -> Dict:
   
    musiq_ava_metric = pyiqa.create_metric('musiq', dataset='ava')
    result = []

    for img in batch:
        musiq_ava_value = musiq_ava_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(musiq_ava_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def maniqa_koniq(batch: Tensor) -> Dict:
   
    maniqa_koniq_metric = pyiqa.create_metric('maniqa', dataset='koniq')
    result = []

    for img in batch:
        maniqa_koniq_value = maniqa_koniq_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(maniqa_koniq_value, dtype=torch.float32))

    result = torch.stack(result)

    return result


def cnniqa(batch: Tensor) -> Dict:
   
    cnniqa_metric = pyiqa.create_metric('cnniqa')
    result = []

    for img in batch:
        cnniqa_value = cnniqa_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(cnniqa_value, dtype=torch.float32))

    result = torch.stack(result)

    return result


