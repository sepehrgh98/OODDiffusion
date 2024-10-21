import numpy as np
from scipy.ndimage import median_filter
import torch
import torch.nn.functional as F
import os
import json


def cosine_similarity(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape.")
    
    tensor1_flat = tensor1.reshape(-1)
    tensor2_flat = tensor2.reshape(-1)
    
    similarity = F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0))
    
    return similarity.item()



def median_filter_4d(tensor, size=3):
    """
    Apply a median filter to a 4D tensor.

    Parameters:
    - tensor: 4D numpy array
    - size: The size of the median filter (default is 3)

    Returns:
    - Smoothed tensor: 4D numpy array with the median filter applied
    """
    smoothed_tensor = median_filter(tensor, size=size)
    smoothed_tensor = torch.from_numpy(smoothed_tensor)

    return smoothed_tensor


def generate_file_list(image_paths, output_file):
    """
    Function to write the paths of all images from a list to a text file.
    
    Args:
        image_paths (list): A list of image paths.
        output_file (str): Path to the output text file where image paths will be saved.
    """
    with open(output_file, 'w') as file:
        for image_path in image_paths:
            file.write(image_path + '\n')

    print(f"Image paths have been written to {output_file}")



