from functools import partial
from typing import Tuple

import torch
from torch import nn
import numpy as np

def gaussian_weights(center, n_timestep, std_dev=100):
    """
    Creates a Gaussian-like weight distribution centered around a specific timestep.
    
    Args:
        center (int): The timestep to center the distribution.
        n_timestep (int): Total number of timesteps.
        std_dev (float): Standard deviation controlling the spread of the distribution.
    
    Returns:
        np.ndarray: Normalized probabilities for each timestep.
    """
    timesteps = np.arange(n_timestep)
    weights = np.exp(-0.5 * ((timesteps - center) / std_dev) ** 2)
    return weights / weights.sum()  # Normalize to sum to 1

def make_beta_schedule(
    schedule,
    n_timestep, 
    center_weights=None,
    linear_start=1e-4, 
    linear_end=2e-2, 
    cosine_s=8e-3):

    if schedule == "linear":
        betas = (
            np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "sqrt":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64) ** 0.5
    elif schedule == "uniform":
        # Uniform schedule: all betas have the same value.
        uniform_value = (linear_start + linear_end) / 2  # Take an average value between start and end.
        betas = np.full((n_timestep,), uniform_value, dtype=np.float64)
    elif schedule == "weighted":
        if center_weights is None:
            raise ValueError("For 'weighted', you must specify a `center_weights`.")
        
        weights = gaussian_weights(center=center_weights, n_timestep=n_timestep, std_dev=50)
        base_betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        betas = base_betas * weights
        betas = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps"
    ):
        super().__init__()
        self.num_timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s
        assert parameterization in ["eps", "x0", "v"], "currently only supporting 'eps' and 'x0' and 'v'"
        self.parameterization = parameterization
        self.loss_type = loss_type

        
        
        high_betas = make_beta_schedule(beta_schedule, 
                                   timesteps, 
                                   center_weights=999,
                                   linear_start=linear_start, 
                                   linear_end=linear_end,
                                   cosine_s=cosine_s)
        
        mid_betas = make_beta_schedule(beta_schedule, 
                                   timesteps, 
                                   center_weights=666,
                                   linear_start=linear_start, 
                                   linear_end=linear_end,
                                   cosine_s=cosine_s)
        high_alphas = 1. - high_betas
        mid_alphas = 1. - mid_betas
        high_alphas_cumprod = np.cumprod(high_alphas, axis=0)
        mid_alphas_cumprod = np.cumprod(mid_alphas, axis=0)
        sqrt_high_alphas_cumprod = np.sqrt(high_alphas_cumprod)
        sqrt_one_minus_high_alphas_cumprod = np.sqrt(1. - high_alphas_cumprod)
        sqrt_mid_alphas_cumprod = np.sqrt(mid_alphas_cumprod)
        sqrt_one_minus_mid_alphas_cumprod = np.sqrt(1. - mid_alphas_cumprod)

        self.high_betas = high_betas
        self.mid_betas = mid_betas

        self.register("sqrt_high_alphas_cumprod", sqrt_high_alphas_cumprod)
        self.register("sqrt_one_minus_high_alphas_cumprod", sqrt_one_minus_high_alphas_cumprod)

        self.register("sqrt_mid_alphas_cumprod", sqrt_mid_alphas_cumprod)
        self.register("sqrt_one_minus_mid_alphas_cumprod", sqrt_one_minus_mid_alphas_cumprod)
    
    def register(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def q_sample(self, x_start, t, noise, OOD_res):

        if self.beta_schedule == "weighted":

            # Compute tensors for high (ID) and mid (OOD) alphas
            sqrt_high_alphas_cumprod = extract_into_tensor(self.sqrt_high_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_high_alphas_cumprod = extract_into_tensor(self.sqrt_one_minus_high_alphas_cumprod, t, x_start.shape)
                
            sqrt_mid_alphas_cumprod = extract_into_tensor(self.sqrt_mid_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_mid_alphas_cumprod = extract_into_tensor(self.sqrt_one_minus_mid_alphas_cumprod, t, x_start.shape)

            # Expand OOD_res to match x_start dimensions
            OOD_res_expanded = OOD_res.view(-1, 1, 1, 1)

            # Compute results for ID (high) and OOD (mid)
            high_result = (
                sqrt_high_alphas_cumprod * x_start +
                sqrt_one_minus_high_alphas_cumprod * noise
            )

            mid_result = (
                sqrt_mid_alphas_cumprod * x_start +
                sqrt_one_minus_mid_alphas_cumprod * noise
            )

            # Merge results using OOD_res
            final_result = OOD_res_expanded * high_result + (1 - OOD_res_expanded) * mid_result
            return final_result
        else:
            return (
                extract_into_tensor(self.sqrt_high_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_high_alphas_cumprod, t, x_start.shape) * noise
            )

    # def get_v(self, x, noise, t):
    #     return (
    #         extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
    #         extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
    #     )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # def p_losses(self, model, x_start, t, cond):
    #     noise = torch.randn_like(x_start)
    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     model_output = model(x_noisy, t, cond)

    #     if self.parameterization == "x0":
    #         target = x_start
    #     elif self.parameterization == "eps":
    #         target = noise
    #     elif self.parameterization == "v":
    #         target = self.get_v(x_start, noise, t)
    #     else:
    #         raise NotImplementedError()

    #     loss_simple = self.get_loss(model_output, target, mean=False).mean()
    #     return loss_simple
