import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import functools
from utils import *
from models.UNet import UNet
from ddpm_utils import *

@hydra.main(config_path="../configs", config_name="config_ddpm")
def main(cfg: DictConfig):
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")
    
    if cfg.reproducibility.seed is not None:
        torch.manual_seed(cfg.reproducibility.seed)
        np.random.seed(cfg.reproducibility.seed)
    if cfg.reproducibility.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cfg.system.cudnn_benchmark


    data_name = cfg.data.dataset_type
    bfgs_iters = cfg.data.bfgs_iters

    samples_path = cfg.sampling.samples_path.format(
        dataset=data_name,
        bfgs_iters=bfgs_iters,
        noise=num2str_deciaml(cfg.data.noise_level)
    )
    
    data = np.load(cfg.data.data_path)

    imgs_true = data["imgs_true"][cfg.sampling.start_ind:cfg.sampling.end_ind, ...]
    imgs_pred = data["imgs_pred"][cfg.sampling.start_ind:cfg.sampling.end_ind, ...]

    # Create N copies of each image for averaging
    if cfg.sampling.num_copies > 1:
        imgs_true = np.repeat(imgs_true, cfg.sampling.num_copies, axis=0)
        imgs_pred = np.repeat(imgs_pred, cfg.sampling.num_copies, axis=0)
        
    test_hat = torch.from_numpy(imgs_pred).float().reshape(-1, cfg.data.img_size, cfg.data.img_size, 1)
    test_true = torch.from_numpy(imgs_true).float().reshape(-1, cfg.data.img_size, cfg.data.img_size, 1)

    clip = torch.max(torch.abs(test_true)) * cfg.ddpm.clip_coeff

    unet_config = process_unet_config(cfg, cfg.model.c0, cfg.model.embed_dim)
    
    if cfg.model.name == 'UNet':
        model = UNet(cfg, unet_config).to(device)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, sigma = get_parameters(
        cfg.ddpm.beta_start, cfg.ddpm.beta_end, cfg.ddpm.Nt
    )

    chkpts_name = cfg.sampling.model_path
    print(f"Loading model from: {chkpts_name}")

    checkpoint = torch.load(chkpts_name, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tic_ddpm = time.time()
    pd, Process = solver_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, cfg.ddpm.Nt, test_hat[..., [0]])
    toc = time.time() - tic_ddpm
    avg_time_ddpm = toc / test_hat.shape[0]

    print(f"Average inference time per sample: {avg_time_ddpm:.4f} seconds", flush=True)

    # Average the results if multiple copies were created
    if cfg.sampling.num_copies > 1:
        original_num_images = (cfg.sampling.end_ind - cfg.sampling.start_ind)
        pd_averaged = np.zeros((original_num_images, N_pts, N_pts, 1))
        
        for i in range(original_num_images):
            start_idx = i * cfg.sampling.num_copies
            end_idx = start_idx + cfg.sampling.num_copies
            pd_averaged[i, ...] = np.mean(pd[start_idx:end_idx, ...], axis=0)
        
        print(f"Averaged {cfg.sampling.num_copies} samples for each of {original_num_images} original images", flush=True)
        
        # Save averaged results
        print(f"Saving averaged results to: {samples_path}")
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        np.save(samples_path, pd_averaged)
        
        if cfg.sampling.save_individual_samples:
            individual_samples_path = samples_path.replace('.npy', '_individual.npy')
            print(f"Saving individual samples to: {individual_samples_path}")
            np.save(individual_samples_path, pd)
    else:
        # Save results without averaging
        print(f"Saving results to: {samples_path}")
            os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        np.save(samples_path, pd)

if __name__ == "__main__":
    main() 