import importlib
from typing import Mapping, Any, Dict, Callable, Tuple, Optional
import torch
from scipy.ndimage import median_filter
import torch.nn.functional as F
import os
import numpy as np
import torch.nn as nn
import math
from einops import repeat
from inspect import isfunction
import warnings
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import tempfile
import uuid
import errno
import hashlib
import sys
import shutil
from torch import Tensor
from PIL import Image






_hub_dir = None
ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'

class _Faketqdm:  # type: ignore[no-redef]

    def __init__(self, total=None, disable=False,
                 unit=None, *args, **kwargs):
        self.total = total
        self.disable = disable
        self.n = 0
        # Ignore all extra *args and **kwargs lest you want to reinvent tqdm

    def update(self, n):
        if self.disable:
            return

        self.n += n
        if self.total is None:
            sys.stderr.write(f"\r{self.n:.1f} bytes")
        else:
            sys.stderr.write(f"\r{100 * self.n / float(self.total):.1f}%")
        sys.stderr.flush()

    # Don't bother implementing; use real tqdm if you want
    def set_description(self, *args, **kwargs):
        pass

    def write(self, s):
        sys.stderr.write(f"{s}\n")

    def close(self):
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return

        sys.stderr.write('\n')

try:
    from tqdm import tqdm  # If tqdm is installed use it, otherwise use the fake wrapper
except ImportError:
    tqdm = _Faketqdm

__all__ = [
    'download_url_to_file',
    'get_dir',
    'help',
    'list',
    'load',
    'load_state_dict_from_url',
    'set_dir',
]

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


def euclidean_distance(tensor1, tensor2):
    """
    Calculate Euclidean distance between two batches of images.
    
    Args:
        tensor1 (torch.Tensor): A tensor of shape [batch_size, channels, height, width].
        tensor2 (torch.Tensor): A tensor of shape [batch_size, channels, height, width].

    Returns:
        torch.Tensor: A tensor of shape [batch_size], containing the Euclidean distance
                      between corresponding images in tensor1 and tensor2.
    """
    # Ensure the two tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"
    
    # Flatten each image in the batch to a 1D vector
    batch_size = tensor1.shape[0]
    tensor1_flat = tensor1.reshape(batch_size, -1)  # Shape: [batch_size, channels * height * width]
    tensor2_flat = tensor2.reshape(batch_size, -1)  # Shape: [batch_size, channels * height * width]
    
    # Calculate Euclidean distance between corresponding images in the two batches
    distance = torch.norm(tensor1_flat - tensor2_flat, p=2, dim=1)
    
    return distance



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



def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME,
                                         DEFAULT_CACHE_DIR), 'torch')))
    return torch_home



def get_dir():
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_HUB'):
        warnings.warn('TORCH_HUB is deprecated, please use env TORCH_HOME instead')

    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_torch_home(), 'hub')


def download_url_to_file(url: str, dst: str, hash_prefix: Optional[str] = None,
                         progress: bool = True) -> None:
    r"""Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # xdoctest: +REQUIRES(POSIX)
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    # We deliberately do not use NamedTemporaryFile to avoid restrictive
    # file permissions being applied to the downloaded file.
    dst = os.path.expanduser(dst)
    for seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + '.' + uuid.uuid4().hex + '.partial'
        try:
            f = open(tmp_dst, 'w+b')
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, 'No usable temporary file name found')

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(f'invalid hash value (expected "{hash_prefix}", got "{digest}")')
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/download_util.py/
def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file



def load_model_from_url(url: str) -> Dict[str, torch.Tensor]:
    sd_path = load_file_from_url(url, model_dir="weights")
    sd = torch.load(sd_path, map_location="cpu")
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


def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> Tuple[int, int, int, int]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)
    
    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)
    
    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords


COUNT_VRAM = bool(os.environ.get("COUNT_VRAM", False))

def count_vram_usage(func: Callable) -> Callable:
    if not COUNT_VRAM:
        return func
    
    def wrapper(*args, **kwargs):
        peak_before = torch.cuda.max_memory_allocated() / (1024 ** 3)
        ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        peak_after = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"VRAM peak before {func.__name__}: {peak_before:.5f} GB, after: {peak_after:.5f} GB")
        return ret
    return wrapper



# https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)]
    weights = np.outer(y_probs, x_probs)
    return weights



def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)
    
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + [x for x in ctx.input_params if x.requires_grad],
            output_grads,
            allow_unused=True,
        )
        grads = list(grads)
        # Assign gradients to the correct positions, matching None for those that do not require gradients
        input_grads = []
        for tensor in ctx.input_tensors + ctx.input_params:
            if tensor.requires_grad:
                input_grads.append(grads.pop(0))  # Get the next computed gradient
            else:
                input_grads.append(None)  # No gradient required for this tensor
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + tuple(input_grads)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq


def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> Tuple[int, int, int, int]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)
    
    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)
    
    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords


# https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)]
    weights = np.outer(y_probs, x_probs)
    return weights


def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq


# Function to calculate noise levels based on user input
def calculate_noise_levels(num_timesteps, num_levels):
    step = num_timesteps // num_levels
    noise_levels = [step * (i + 1) for i in range(num_levels)]
    return noise_levels



def calculate_psnr(img1, img2, max_pixel_value=1.0):
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr

def batch_psnr(batch1, batch2, max_pixel_value=1.0):
    """
    Calculate PSNR between two batches of images one by one.

    Args:
        batch1 (torch.Tensor): A tensor of shape (batch_size, channels, height, width).
        batch2 (torch.Tensor): A tensor of shape (batch_size, channels, height, width).
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0 (normalized images).

    Returns:
        list: A list containing PSNR values for each pair of images in the batches.
    """
    # Ensure the input tensors have the same shape
    assert batch1.shape == batch2.shape, "Input batches must have the same shape."

    psnr_values = []
    batch_size = batch1.shape[0]

    for i in range(batch_size):
        img1 = batch1[i]
        img2 = batch2[i]
        psnr = calculate_psnr(img1, img2, max_pixel_value)
        psnr_values.append(psnr.item())
    
    return psnr_values


def save(img, out_dir, img_name):
    save_path = os.path.join(out_dir, img_name)
    Image.fromarray(img).save(save_path)





def save(img, out_dir, img_name):   

    img = img.squeeze(axis=0)   
    save_path = os.path.join(out_dir, img_name)
    Image.fromarray(img.cpu().numpy()).save(save_path)


def normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)


def bicubic_resize(images: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize a batch of images using bicubic interpolation for input in [B, C, H, W] format.
    
    Args:
        images (np.ndarray): A batch of images with shape [B, C, H, W].
        scale (float): The scaling factor for resizing.
    
    Returns:
        np.ndarray: A batch of resized images in [B, C, new_H, new_W] format.
    """
    resized_images = []
    batch_size, channels, height, width = images.shape
    
    for img in images:
        # Iterate over each channel, resize individually, and stack
        resized_channels = []
        for c in range(channels):
            pil = Image.fromarray(img[c])  # Convert each channel to PIL image
            res = pil.resize((int(width * scale), int(height * scale)), Image.BICUBIC)
            resized_channels.append(np.array(res))
        
        # Stack resized channels back into a single image
        resized_image = np.stack(resized_channels, axis=0)
        resized_images.append(resized_image)
    
    # Stack resized images back into a batch
    return np.stack(resized_images, axis=0)
