import importlib
from typing import Mapping, Any, Dict
import torch
from scipy.ndimage import median_filter
import torch.nn.functional as F


def cosine_similarity(tensor1, tensor2):
    """
    Calculate cosine similarity between two batches of images.
    
    Args:
        tensor1 (torch.Tensor): A tensor of shape [batch_size, channels, height, width].
        tensor2 (torch.Tensor): A tensor of shape [batch_size, channels, height, width].

    Returns:
        torch.Tensor: A tensor of shape [batch_size], containing the cosine similarity
                      between corresponding images in tensor1 and tensor2.
    """
    # Ensure the two tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"
    
    # Flatten each image in the batch to a 1D vector
    batch_size = tensor1.shape[0]
    tensor1_flat = tensor1.reshape(batch_size, -1)  # Shape: [batch_size, channels * height * width]
    tensor2_flat = tensor2.reshape(batch_size, -1)  # Shape: [batch_size, channels * height * width]
    
    # Calculate cosine similarity between corresponding images in the two batches
    similarity = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
    
    return similarity



def median_filter_4d(tensor, size=3):
    # Initialize a list to store the filtered images
    smoothed_images = []

    # Loop through each image in the batch
    for i in range(tensor.shape[0]):
        # Detach the tensor and convert it to a NumPy array
        detached_image = tensor[i].detach().cpu().numpy()

        # Apply median filter to each channel of the image
        smoothed_image = median_filter(detached_image, size=size)

        # Convert the result back to a tensor and append to the list
        smoothed_images.append(torch.tensor(smoothed_image, dtype=torch.float32, device=tensor.device))

    # Stack all the filtered images along the batch dimension
    smoothed_tensor = torch.stack(smoothed_images, dim=0)
    return smoothed_tensor



def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    # get_pad = lambda x: (x // multiple + 1) * multiple - x
    get_pad = lambda x: (x // multiple + int(x % multiple != 0)) * multiple - x
    ph, pw = get_pad(h), get_pad(w)
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)



def load_model_from_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    sd = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in sd:
        sd = sd["state_dict"]

    if list(sd.keys())[0].startswith("module"):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    return sd


def get_obj_from_str(string: str, reload: bool=False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))