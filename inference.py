from argparse import Namespace, ArgumentParser
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os
import csv
from tqdm import tqdm
from einops import rearrange
from PIL import Image
from accelerate.utils import set_seed


from model.SwinIR import SwinIR
from model.cldm import ControlLDM
from model.gaussian_diffusion import Diffusion
from model.cond_fn import MSEGuidance, WeightedMSEGuidance
from model.ResNet50 import ResNet50
from model.sampler import SpacedSampler

from dataset.HybridDataset import TestDataset

from Similarity import (psnr,
                        ssim,
                        clipiqa,
                        musiq,
                        musiq_koniq,
                        musiq_ava,
                        lpips,
                        niqe,
                        maniqa_koniq,
                        nima,
                        cnniqa,
                        brisque)


from utils import (instantiate_from_config
                   ,load_model_from_checkpoint
                   ,pad_to_multiples_of
                   ,cosine_similarity
                   ,median_filter_4d
                   ,wavelet_decomposition
                   ,normalize
                   ,wavelet_reconstruction)


class InferenceLoop:

    def __init__(self, args: Namespace) -> "InferenceLoop" :
        # initialization
        self.args = args
        self.cfg = OmegaConf.load(self.args.config)
        self.stage1_model = None
        self.cldm = None
        self.diffusion = None
        self.cond_fn = None
        self.OOD_detector = None
        

        # init
        self.init_stage1()
        self.init_stage2()
        self.init_OOD_detector()
        self.setup()

    def init_stage1(self):
        print("[INFO] Load SwinIR...")
        self.stage1_model: SwinIR = instantiate_from_config(self.cfg.model.swinir)
        sd = load_model_from_checkpoint(self.cfg.test.swin_check_dir)
        self.stage1_model.load_state_dict(sd, strict=True)
        self.stage1_model.eval().to(self.args.device)

    def init_stage2(self):
        print("[INFO] Load ControlLDM...")
        self.cldm: ControlLDM = instantiate_from_config(self.cfg.model.cldm)
        # sd = load_model_from_url(MODELS["sd_v21"])
        sd = load_model_from_checkpoint(self.cfg.test.diffusion_check_dir)
        unused = self.cldm.load_pretrained_sd(sd)
        print(f"[INFO] strictly load pretrained sd_v2.1, unused weights: {unused}")

        ### load controlnet
        print("[INFO] Load ControlNet...")
        # control_sd = load_model_from_url(MODELS["v2"])
        control_sd = load_model_from_checkpoint(self.cfg.test.controlnet_check_dir)
        self.cldm.load_controlnet_from_ckpt(control_sd)
        print(f"[INFO] strictly load controlnet weight")
        self.cldm.eval().to(self.args.device)

        ### load diffusion
        print("[INFO] Load Diffusion...")
        self.diffusion: Diffusion = instantiate_from_config(self.cfg.model.diffusion)
        self.diffusion.to(self.args.device)

        # Initialize Condition
        if not self.args.guidance:
            self.cond_fn = None
        else:
            if self.args.g_loss == "mse":
                cond_fn_cls = MSEGuidance
            elif self.args.g_loss == "w_mse":
                cond_fn_cls = WeightedMSEGuidance
            else:
                raise ValueError(self.args.g_loss)
            self.cond_fn = cond_fn_cls(
                scale=self.args.g_scale, t_start=self.args.g_start, t_stop=self.args.g_stop,
                space=self.args.g_space, repeat=self.args.g_repeat
            )

    def init_OOD_detector(self):
        # Initialize ResNet
        print("[INFO] Load OOD Detector...")
        self.OOD_detector: ResNet50 = instantiate_from_config(self.cfg.model.resnet)
        rd = load_model_from_checkpoint(self.cfg.test.res_check_dir)
        self.OOD_detector.load_state_dict(rd, strict=True)
        self.OOD_detector.eval().to(self.args.device)

    def init_dataset(self):
        # Setup data
        print("[INFO] Setup Dataset...")
        dataset: TestDataset = instantiate_from_config(self.cfg.dataset)
        loader = DataLoader(
        dataset=dataset, batch_size=self.cfg.test.batch_size,
        num_workers=self.cfg.test.num_workers,
        shuffle=True)

        return loader
    
    def setup(self):

        # Make dir
        self.output_dir = self.args.output
        os.makedirs(self.output_dir, exist_ok=True)
        self.quant_res_path = os.path.join(self.output_dir,"quantative")
        os.makedirs(self.quant_res_path, exist_ok=True)
        self.qual_res_path = os.path.join(self.output_dir,"qualitative")
        os.makedirs(self.qual_res_path, exist_ok=True)

        # Make csv output
        header = [
        "label", "detector_pred", "PSNR", "SSIM", "LPIPS",
        "BRISQUE", "CLIP-IQA", "NIMA", "NIQE", "MUSIQ", "MUSIQ-KONIQ", 
        "MUSIQ-AVA",  "MANIQA-KONIQ", "CNNIQA"]

        # Open CSV file in write mode and write the header
        with open(self.quant_res_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write the header

        print("[INFO] Result dir created...")

    def run_stage1(self, image):
        pad_image = pad_to_multiples_of(image, multiple=64)
        output, features = self.stage1_model(pad_image)
        return output, features
    
    @torch.no_grad()
    def OOD_detector(self, img, img_feature):
        epsilon = 1e-8
        threshold = 16
        output = self.OOD_detector(img).squeeze()
        sim  = cosine_similarity(img_feature, median_filter_4d(img_feature))
        real_final_prob = sim / (output + epsilon)
        result = (real_final_prob < threshold).int() # 1: in distribution 0: Out of distribution
        return result

        
    
    def run_stage2(self, clean, OOD_res):

        ### preprocess
        bs, _, ori_h, ori_w = clean.shape

        # pad: ensure that height & width are multiples of 64
        pad_clean = pad_to_multiples_of(clean, multiple=64)
        h, w = pad_clean.shape[2:]

        # prepare conditon
        if not self.args.tiled:
            cond = self.cldm.prepare_condition(pad_clean, [self.args.pos_prompt] * bs)
            uncond = self.cldm.prepare_condition(pad_clean, [self.args.neg_prompt] * bs)
        else:
            cond = self.cldm.prepare_condition_tiled(pad_clean, [self.args.pos_prompt] * bs, self.args.tile_size, self.args.tile_stride)
            uncond = self.cldm.prepare_condition_tiled(pad_clean, [self.args.neg_prompt] * bs, self.args.tile_size, self.args.tile_stride)
        if self.cond_fn:
            self.cond_fn.load_target(pad_clean * 2 - 1)

        old_control_scales = self.cldm.control_scales
        self.cldm.control_scales = [self.args.strength] * 13

        # number of steps
        num_timesteps = torch.where(OOD_res == 1, torch.tensor(999), torch.tensor(666))

        if self.args.better_start:
            # using noised low frequency part of condition as a better start point of 
            # reverse sampling, which can prevent our model from generating noise in 
            # image background.
            _, low_freq = wavelet_decomposition(pad_clean)

            if not self.args.tiled:
                x_0 = self.cldm.vae_encode(low_freq)
            else:
                x_0 = self.cldm.vae_encode_tiled(low_freq, self.args.tile_size, self.args.tile_stride)


            x_T = self.diffusion.q_sample(
               x_0,
               torch.full((bs, ), num_timesteps, dtype=torch.long, device=self.args.device),
               torch.randn(x_0.shape, dtype=torch.float32, device=self.args.device)
            )
        else:
            x_T = torch.randn((bs, 4, h // 8, w // 8), dtype=torch.float32, device=self.device)

        ### run sampler
        sampler = SpacedSampler(self.diffusion.betas)
        z = sampler.sample(
            model=self.cldm, device=self.args.device, steps=self.args.steps, batch_size=bs, x_size=(4, h // 8, w // 8),
            cond=cond, uncond=uncond, cfg_scale=self.args.cfg_scale, x_T=x_T, progress=True,
            progress_leave=True, cond_fn=self.cond_fn, tiled=self.args.tiled, tile_size=self.args.tile_size, tile_stride=self.args.tile_stride
        )

        if not self.args.tiled:
            x = self.cldm.vae_decode(z)
        else:
            x = self.cldm.vae_decode_tiled(z, self.args.tile_size // 8, self.args.tile_stride // 8)

        ### postprocess
        self.cldm.control_scales = old_control_scales
        sample = x[:, :, :ori_h, :ori_w]

        return sample



    @torch.no_grad()
    def run(self) -> None: 
        print("[INFO] Run...")
        self.setup()

        # data loader
        loader = self.init_dataset()

        # device
        device = self.args.device

        # Variables to track accuracy
        total_samples = 0
        correct_predictions = 0

        for batch_idx, (img, label) in enumerate(tqdm(loader)):
            img, label = img.to(device), label.to(device)

            # stage1
            img_clean, img_feature = self.run_stage1(img)
            torch.cuda.empty_cache()


            # OOD detection
            OOD_res = self.OOD_detector(img, img_feature)
            torch.cuda.empty_cache()

            # Calculate correct predictions
            correct_predictions += torch.sum(OOD_res == label).item()
            total_samples += label.size(0)


            # stage2
            sample = self.run_stage2(clean = img_clean, OOD_res=OOD_res)
            torch.cuda.empty_cache()

            # post process
            sample = normalize(sample)
            sample = wavelet_reconstruction(sample, img_clean)
            sample = normalize(sample)



            # Image Quality Metrics Calculation
            Q_metrics = {
                "psnr": psnr(sample, img_clean),
                "ssim": ssim(sample, img_clean),
                "clipiqa": clipiqa(sample, img_clean),
                "musiq": musiq(sample),
                "musiq_koniq": musiq_koniq(sample),
                "musiq_ava": musiq_ava(sample),
                "lpips": lpips(sample, img_clean),
                "niqe": niqe(sample),
                "maniqa_koniq": maniqa_koniq(sample),
                "nima": nima(sample),
                "cnniqa": cnniqa(sample),
                "brisque": brisque(sample, img_clean),
            }

            # Save Quality metrics in csv file
            with open(self.quant_res_path, mode='w', newline='') as file:
                writer = csv.writer(file)

                for i in range(len(img)):
                    writer.writerow([label[i].item()
                        ,OOD_res[i].item()
                        ,Q_metrics["psnr"][i]
                        ,Q_metrics["ssim"][i]
                        ,Q_metrics["lpips"][i]
                        ,Q_metrics["brisque"][i]
                        ,Q_metrics["clipiqa"][i]
                        ,Q_metrics["nima"][i]
                        ,Q_metrics["niqe"][i]
                        ,Q_metrics["musiq"][i]
                        ,Q_metrics["musiq_koniq"][i]
                        ,Q_metrics["musiq_ava"][i]
                        ,Q_metrics["maniqa_koniq"][i]
                        ,Q_metrics["cnniqa"][i]
                        ])

            if batch_idx % self.test.cfg.save_image_every == 0:       
                # Save sample images
                self.save(img[0], f"{batch_idx}_LR")
                self.save(sample[0], f"{batch_idx}_HR")

        
        # Calculate and print overall accuracy
        accuracy = correct_predictions / total_samples * 100
        print(f"[INFO] OOD Detection Accuracy: {accuracy:.2f}%")

    def save(self, img, prefix):
        img = rearrange(img * 255., "n c h w -> n h w c")
        img = img.contiguous().clamp(0, 255).to(torch.uint8).cpu()

        save_path = os.path.join(self.qual_res_path, prefix)
        Image.fromarray(img).save(save_path)



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
    InferenceLoop(args.seed).run()
    print("done!")
        





