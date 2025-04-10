from argparse import ArgumentParser, Namespace
import torch
import pyiqa


from utils import (pad_to_multiples_of
                   , wavelet_decomposition
                   , wavelet_reconstruction
                   , calculate_noise_levels)

from model.sampler import SpacedSampler



class PSNRDetector:
    def __init__(self
                , args: Namespace
                , cldm: ControlLDM
                , cond_fn
                , diffusion: Diffusion
                , threshold: double):
        
        self.args = args
        self.cldm = cldm
        self.cond_fn = cond_fn
        self.diffusion = diffusion
        self.threshold = threshold

        # Setup Similarity Metrics
        self.psnr_metric = pyiqa.create_metric('psnr')

        # Noising Setup
        self.noise_levels = calculate_noise_levels(num_timesteps = self.diffusion.num_timesteps
                        , num_levels = self.args.num_levels)


    def add_diffusion_noise(self, x_0):
        noisy_versions = []
        bs = x_0.shape[0]
        for level in self.noise_levels:
            noise_scale = torch.full((bs,), level, dtype=torch.long, device=self.args.device)
            noise = torch.randn(x_0.shape, dtype=torch.float32, device=self.args.device)
            x_T = self.diffusion.q_sample(x_0, noise_scale, noise)
            noisy_versions.append(x_T)
        return noisy_versions

    def run(self, clean: torch.Tensor):

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

        if self.better_start:

            _, low_freq = wavelet_decomposition(pad_clean)

            if not self.args.tiled:
                x_0 = self.cldm.vae_encode(low_freq)
            else:
                x_0 = self.cldm.vae_encode_tiled(low_freq, self.args.tile_size, self.args.tile_stride)


            # add noise
            x_T_s = add_diffusion_noise(x_0 = x_0) # [X_T1 , X_T2 ,...]

        else:
            x_T_s = [torch.randn((bs, 4, h // 8, w // 8), dtype=torch.float32, device=self.args.device) for _ in range(len(noise_levels))]


        ### run sampler
        sampler = SpacedSampler(diffusion.betas)

        PSNRs = []
        for x_T in x_T_s:

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

            sample = normalize(sample)
            sample = wavelet_reconstruction(sample, clean)
            sample = normalize(sample)

            sim = psnr(sample, clean, self.psnr_metric)
            PSNRs.append(sim)


        mean_psnr = np.mean(np.array(PSNRs), axis=0).tolist()
        result = (mean_psnr > self.threshold).int() # 1: in distribution 0: Out of distribution

        return result





    


