from argparse import ArgumentParser, Namespace
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from torch.nn import functional as F
import numpy as np
from accelerate.utils import set_seed
import os
import csv




from utils import (pad_to_multiples_of
                   ,instantiate_from_config
                   , load_model_from_checkpoint
                   , load_model_from_url
                   , wavelet_decomposition
                   , wavelet_reconstruction
                   , calculate_noise_levels
                   , batch_psnr)

from model.SwinIR import SwinIR
from model.cldm import ControlLDM
from model.gaussian_diffusion import Diffusion
from model.cond_fn import MSEGuidance, WeightedMSEGuidance
from model.sampler import SpacedSampler
from dataset.HybridDataset import HybridDataset


MODELS = {
    ### stage_2 model weights
    "sd_v21": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth"
}

 

def run_stage1(swin_model, image, device='cpu'):
    # image to tensor
    image = torch.tensor((image / 255.).clip(0, 1), dtype=torch.float32, device=device)
    pad_image = pad_to_multiples_of(image, multiple=64)
    # run
    output, features = swin_model(pad_image)

    return output, features


def run_stage2(
    clean: torch.Tensor,
    cldm: ControlLDM,
    cond_fn,
    diffusion: Diffusion,
    steps: int,
    strength: float,
    tiled: bool,
    tile_size: int,
    tile_stride: int,
    pos_prompt: str,
    neg_prompt: str,
    cfg_scale: float,
    better_start: float,
    device,
    noise_levels: list
) -> torch.Tensor:
    
    ### preprocess
    bs, _, ori_h, ori_w = clean.shape
    
    
    # pad: ensure that height & width are multiples of 64
    pad_clean = pad_to_multiples_of(clean, multiple=64)
    h, w = pad_clean.shape[2:]
    
    
    # prepare conditon
    if not tiled:
        cond = cldm.prepare_condition(pad_clean, [pos_prompt] * bs)
        uncond = cldm.prepare_condition(pad_clean, [neg_prompt] * bs)
    else:
        cond = cldm.prepare_condition_tiled(pad_clean, [pos_prompt] * bs, tile_size, tile_stride)
        uncond = cldm.prepare_condition_tiled(pad_clean, [neg_prompt] * bs, tile_size, tile_stride)
    if cond_fn:
        cond_fn.load_target(pad_clean * 2 - 1)
    old_control_scales = cldm.control_scales
    cldm.control_scales = [strength] * 13
    if better_start:
        # using noised low frequency part of condition as a better start point of 
        # reverse sampling, which can prevent our model from generating noise in 
        # image background.
        _, low_freq = wavelet_decomposition(pad_clean)
        if not tiled:
            x_0 = cldm.vae_encode(low_freq)
        else:
            x_0 = cldm.vae_encode_tiled(low_freq, tile_size, tile_stride)
        
        # x_T = diffusion.q_sample(
        #     x_0,
        #     torch.full((bs, ), diffusion.num_timesteps - 1, dtype=torch.long, device=device),
        #     torch.randn(x_0.shape, dtype=torch.float32, device=device)
        # )

        # add noise
        x_T_s = add_diffusion_noise(x_0 = x_0
                                            , diffusion = diffusion
                                            , noise_levels = noise_levels
                                            , device = device) # [X_T1 , X_T2 ,...]


    else:
        x_T_s = [torch.randn((bs, 4, h // 8, w // 8), dtype=torch.float32, device=device) for _ in range(len(noise_levels))]
    
    ### run sampler
    sampler = SpacedSampler(diffusion.betas)

    clean_samples = []

    for x_T in x_T_s:
        z = sampler.sample(
            model=cldm, device=device, steps=steps, batch_size=bs, x_size=(4, h // 8, w // 8),
            cond=cond, uncond=uncond, cfg_scale=cfg_scale, x_T=x_T, progress=True,
            progress_leave=True, cond_fn=cond_fn, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        if not tiled:
            x = cldm.vae_decode(z)
        else:
            x = cldm.vae_decode_tiled(z, tile_size // 8, tile_stride // 8)
        ### postprocess
        cldm.control_scales = old_control_scales
        sample = x[:, :, :ori_h, :ori_w]
        clean_samples.append(sample)
    return clean_samples

# Function to add noise at different levels using diffusion process
def add_diffusion_noise(x_0, diffusion, noise_levels, device):
    noisy_versions = []
    bs = x_0.shape[0]
    for level in noise_levels:
        noise_scale = torch.full((bs,), level, dtype=torch.long, device=device)
        noise = torch.randn(x_0.shape, dtype=torch.float32, device=device)
        x_T = diffusion.q_sample(x_0, noise_scale, noise)
        noisy_versions.append(x_T)
    return noisy_versions

def main(args):
    
    print("[INFO] Start...")

    # config 
    cfg = OmegaConf.load(args.config)
    # device
    device = args.device


    # Initialize Stage 1
    print("[INFO] Load SwinIR...")

    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    # sd = load_model_from_checkpoint(cfg.test.swin_check_dir)
    # swinir.load_state_dict(sd, strict=True)
    swinir.eval().to(device)


    # Initialize Stage 2
    print("[INFO] Load ControlLDM...")
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    # # sd = load_model_from_url(MODELS["sd_v21"])
    # sd = load_model_from_checkpoint(cfg.test.diffusion_check_dir)
    # unused = cldm.load_pretrained_sd(sd)
    # print(f"[INFO] strictly load pretrained sd_v2.1, unused weights: {unused}")

    ### load controlnet
    print("[INFO] Load ControlNet...")
    # # control_sd = load_model_from_url(MODELS["v2"])
    # control_sd = load_model_from_checkpoint(cfg.test.controlnet_check_dir)
    # cldm.load_controlnet_from_ckpt(control_sd)
    print(f"[INFO] strictly load controlnet weight")
    cldm.eval().to(device)

    ### load diffusion
    print("[INFO] Load Diffusion...")
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)

    # Initialize Condition
    if not args.guidance:
        cond_fn = None
    else:
        if args.g_loss == "mse":
            cond_fn_cls = MSEGuidance
        elif args.g_loss == "w_mse":
            cond_fn_cls = WeightedMSEGuidance
        else:
            raise ValueError(args.g_loss)
        cond_fn = cond_fn_cls(
            scale=args.g_scale, t_start=args.g_start, t_stop=args.g_stop,
            space=args.g_space, repeat=args.g_repeat
        )


   # Initialize ResNet
    print("[INFO] Load ResNet...")
    resnet_model = instantiate_from_config(cfg.model.resnet)
    rd = load_model_from_checkpoint(cfg.test.res_check_dir)
    resnet_model.load_state_dict(rd, strict=True)
    resnet_model.eval().to(device)

    # Setup data
    print("[INFO] Setup Dataset...")
    dataset: HybridDataset = instantiate_from_config(cfg.dataset)
    test_loader = DataLoader(
    dataset=dataset, batch_size=cfg.test.batch_size,
    num_workers=cfg.test.num_workers,
    shuffle=True)


    # Noising Setup
    noise_levels = calculate_noise_levels(
                    num_timesteps = diffusion.num_timesteps
                    , num_levels = args.num_levels)


    # Setup result path
    result_path = os.path.join(cfg.test.test_result_dir, 'results.csv')

    with open(result_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method_1', 'PSNR', 'Label'])


    for data, label in tqdm(test_loader):

        # Stage 1
        data = torch.tensor((data / 255.).clip(0, 1), dtype=torch.float32, device=device)
        # data = rearrange(data, "n h w c -> n c h w").contiguous()

        # set pipeline output size
        h, w = data.shape[2:]
        final_size = (h, w)
        clean, _ = run_stage1(swin_model=swinir, image=data)


        # OOD Detection: Method 1
        with torch.no_grad():  
            output = resnet_model(data)
            pred_method1 = (output > 0.5).int()

        samples = run_stage2(
            clean = clean,
            cldm = cldm,
            cond_fn = cond_fn,
            diffusion = diffusion,
            steps = args.steps,
            strength = 1.0,
            tiled = args.tiled,
            tile_size = args.tile_size,
            tile_stride = args.tile_stride,
            pos_prompt = args.pos_prompt,
            neg_prompt = args.neg_prompt,
            cfg_scale = args.cfg_scale,
            better_start = args.better_start,
            device = device,
            noise_levels = noise_levels
        )

        batch_similarities = []
        for sample in samples:
            # colorfix (borrowed from StableSR, thanks for their work)
            sample = (sample + 1) / 2
            sample = wavelet_reconstruction(sample, clean)

            # resize to desired output size
            sample = F.interpolate(sample, size=final_size, mode="bicubic", antialias=True)
            
            # tensor to image
            sample = rearrange(sample * 255., "n c h w -> n h w c")
            sample = sample.contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()

            # Calculate PSNR
            batch_similarities.append(batch_psnr(sample, clean))

        batch_similarities = np.mean(np.array(batch_similarities), axis=0).tolist()

        # Write results to CSV file
        with open(result_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(data)):
                writer.writerow([pred_method1[i].item(), batch_similarities[i].item(), label[i].item()])

    


def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                  "built with CUDA enabled.")
            device = "cpu"
    else:
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                          "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                          "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f"using device {device}")
    return device



def parse_args() -> Namespace:
    parser = ArgumentParser()
    ### model parameters
    # parser.add_argument("--task", type=str, required=True, choices=["sr", "dn", "fr", "fr_bg"])
    parser.add_argument("--upscale", type=float, required=True)
    # parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"])
    ### sampling parameters
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--better_start", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=256)
    parser.add_argument("--pos_prompt", type=str, default="")
    parser.add_argument("--neg_prompt", type=str, default="low quality, blurry, low-resolution, noisy, unsharp, weird textures")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    ### input parameters
    # parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=1)
    ### guidance parameters
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--g_loss", type=str, default="w_mse", choices=["mse", "w_mse"])
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_start", type=int, default=1001)
    parser.add_argument("--g_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=1)
    ### output parameters
    # parser.add_argument("--output", type=str, required=True)
    ### common parameters
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    ### number of noise levels
    parser.add_argument("--num_levels", type=int, default=3)
    ### config
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.device = check_device(args.device)
    set_seed(args.seed)
    main(args)
    print("done!")
