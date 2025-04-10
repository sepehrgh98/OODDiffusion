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
import pyiqa



from model.SwinIR import SwinIR
from model.cldm import ControlLDM
from model.gaussian_diffusion import Diffusion
from model.cond_fn import MSEGuidance, WeightedMSEGuidance
from model.ResNet50 import ResNet50
from model.sampler import SpacedSampler


from utils import (instantiate_from_config
                   ,load_model_from_checkpoint
                   ,pad_to_multiples_of
                   ,cosine_similarity
                   ,median_filter_4d
                   ,wavelet_decomposition
                   ,normalize
                   ,wavelet_reconstruction)


def run_stage1(image, model):
    image = pad_to_multiples_of(image, multiple=64)
    output, features = model(image)
    return output, features


def main(args) -> None:
    cfg = OmegaConf.load(args.config)


    # Initialize ResNet
    print("[INFO] Load OOD Detector...")
    OOD_detector: ResNet50 = instantiate_from_config(cfg.model.resnet)
    rd = load_model_from_checkpoint(cfg.test.res_check_dir)
    OOD_detector.load_state_dict(rd, strict=True)
    OOD_detector.eval().to(args.device)

    #swin
    print("[INFO] Load SwinIR...")
    stage1_model: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = load_model_from_checkpoint(cfg.test.swin_check_dir)
    stage1_model.load_state_dict(sd, strict=True)
    stage1_model.eval().to(args.device)


    # Setup data
    print("[INFO] Setup Dataset...")
    dataset: OOD_Inference = instantiate_from_config(cfg.dataset)
    loader = DataLoader(
    dataset=dataset, batch_size=cfg.test.batch_size,
    num_workers=cfg.test.num_workers,
    shuffle=True)

    # Variables to track accuracy
    total_samples = 0
    correct_predictions = 0

    for batch_idx, (img, label) in enumerate(tqdm(loader)):
        img, label = img.to(args.device), label.to(args.device)
        torch.cuda.empty_cache()

        # stage1
        img_clean, img_feature = run_stage1(img, stage1_model)
        torch.cuda.empty_cache()

        # Method 1
        epsilon = 1e-8
        threshold = 16
        output = OOD_detector(img).squeeze()
        sim  = cosine_similarity(img_feature, median_filter_4d(img_feature))
        real_final_prob = sim / (output + epsilon)
        OOD_res = (real_final_prob < threshold).int() # 1: in distibution 0: Out of distribution    

        
        # Calculate correct predictions
        correct_predictions += torch.sum(OOD_res == label).item()
        total_samples += label.size(0)

    # Calculate and print overall accuracy
    accuracy = correct_predictions / total_samples * 100
    print(f"[INFO] OOD Detection Accuracy: {accuracy:.2f}%, {dataset.nReal}/{len(dataset)} real samples and {dataset.nSyn}/{len(dataset)} syn samples")
        




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    main(args)


