from torch import Tensor
import pyiqa
from typing import Dict
import torch
import numpy as np


def psnr(batch1: Tensor, batch2: Tensor, psnr_metric) -> Dict:
    
    # Ensure both batches are of the same size
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
    

    result = []

    for img1, img2 in zip(batch1, batch2):
        psnr_value = psnr_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()
        result.append(torch.tensor(psnr_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def ssim(batch1: Tensor, batch2: Tensor, ssim_metric) -> Dict:
    
    # Ensure both batches are of the same size
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
   
    result = []

    for img1, img2 in zip(batch1, batch2):
        ssim_value = ssim_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()
        result.append(torch.tensor(ssim_value, dtype=torch.float32))

    result = torch.stack(result)

    return result



def lpips(batch1: Tensor, batch2: Tensor, lpips_metric) -> Dict:
    
    # Ensure both batches are of the same size
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
    
    result = []

    for img1, img2 in zip(batch1, batch2):
        lpips_value = lpips_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()
        result.append(torch.tensor(lpips_value, dtype=torch.float32))


    result = torch.stack(result)

    return result



def brisque(batch: Tensor, brisque_metric) -> Dict:
    
    result = []

    for img in batch:
        # img = (img * 255).clamp(0, 255).to(torch.uint8)
        brisque_value = brisque_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(brisque_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def clipiqa(batch: Tensor, clipiqa_metric) -> Dict:
    
    result = []

    for img1 in batch:
        clipiqa_value = clipiqa_metric(img1.unsqueeze(0)).item()
        result.append(torch.tensor(clipiqa_value, dtype=torch.float32))

    result = torch.stack(result)

    return result


def nima(batch: Tensor, nima_metric) -> Dict:

    result = []

    for img in batch:
        nima_value = nima_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(nima_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def niqe(batch: Tensor, niqe_metric) -> Dict:
    
    result = []

    for img in batch:
        niqe_value = niqe_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(niqe_value, dtype=torch.float32))
        
    result = torch.stack(result)

    return result

def musiq(batch: Tensor,  musiq_metric) -> Dict:
   
    result = []

    for img in batch:
        musiq_value = musiq_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(musiq_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def musiq_ava(batch: Tensor, musiq_ava_metric) -> Dict:
   
    result = []

    for img in batch:
        musiq_ava_value = musiq_ava_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(musiq_ava_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def maniqa(batch: Tensor, maniqa_metric) -> Dict:
   
    result = []

    for img in batch:
        maniqa_value = maniqa_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(maniqa_value, dtype=torch.float32))

    result = torch.stack(result)

    return result

def maniqa_kadid(batch: Tensor, maniqa_kadid_metric) -> Dict:

    result = []

    for img in batch:
        maniqa_kadid_value = maniqa_kadid_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(maniqa_kadid_value, dtype=torch.float32))

    result = torch.stack(result)

    return result




def cnniqa(batch: Tensor, cnniqa_metric) -> Dict:
   
    result = []

    for img in batch:
        cnniqa_value = cnniqa_metric(img.unsqueeze(0)).item()
        result.append(torch.tensor(cnniqa_value, dtype=torch.float32))

    result = torch.stack(result)

    return result


