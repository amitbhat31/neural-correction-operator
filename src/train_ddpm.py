import os
import time
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import functools
import scipy.io as sio
import matplotlib.tri as tri

from models.UNet import UNet
from utils import *
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

    # Define grids for plotting and interpolation spanning [-1, 1] × [-1, 1] 
    N_pts = cfg.data.img_size
    lx, ly = cfg.data.lx, cfg.data.ly
    points_x = np.linspace(-lx/2, lx/2, N_pts)
    points_y = np.linspace(-ly/2, ly/2, N_pts)
    xx, yy = np.meshgrid(points_x, points_y)

    GCOORD = np.vstack([xx.ravel(), yy.ravel()]).T
    GCOORD = GCOORD.reshape((N_pts, N_pts, 2))
    GCOORD = np.flip(GCOORD, axis=0)
    GCOORD = GCOORD.reshape((-1, 2))


    # load the file containing the mesh
    mat_fname  = cfg.data.mesh_path
    mat_contents = sio.loadmat(mat_fname)

    p = mat_contents['p']
    t = mat_contents['t']-1 
    centroids = np.mean(p[t], axis=1)
    triangulation = tri.Triangulation(p[:,0], p[:,1], t)


    data_name = cfg.data.dataset_type
    bfgs_iters = cfg.data.bfgs_iters

    save_name = cfg.output.name_template.format(
        dataset=data_name,
        bfgs_iters=bfgs_iters,
        img_size=cfg.data.img_size,
        noise=num2str_deciaml(cfg.data.noise_level),
        train_samples=cfg.data.train_samples,
        lr_max=num2str_deciaml(cfg.training.lr_max),
        lr_min=num2str_deciaml(cfg.training.lr_min),
        Nt=cfg.ddpm.Nt
    )

    data = np.load(cfg.data.data_path)
    imgs_true = data["imgs_true"][:cfg.data.total_samples, ...]
    imgs_pred = data["imgs_pred"][:cfg.data.total_samples, ...]

    imgs_true = torch.from_numpy(imgs_true).float().reshape(cfg.data.total_samples, cfg.data.img_size, cfg.data.img_size, 1)
    imgs_pred = torch.from_numpy(imgs_pred).float().reshape(cfg.data.total_samples, cfg.data.img_size, cfg.data.img_size, 1)

    clip = torch.max(torch.abs(imgs_pred)) * cfg.ddpm.clip_coeff
    
    dataset = []
    for i in range(cfg.data.train_samples):
        tmp_ls = []
        tmp_ls.append(imgs_true[i, ...])
        tmp_ls.append(imgs_pred[i, ...])
        dataset.append(tmp_ls)
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.system.num_workers)

    val_pred = imgs_pred[cfg.data.train_samples:cfg.data.train_samples+cfg.data.val_samples, ...]
    val_true = imgs_true[cfg.data.train_samples:cfg.data.train_samples+cfg.data.val_samples, ...]

    unet_config = process_unet_config(cfg, cfg.model.c0, cfg.model.embed_dim)
    
    if cfg.model.name == 'UNet':
        model = UNet(cfg, unet_config).to(device)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr_max)
    if cfg.training.scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs, eta_min=cfg.training.lr_min)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.training.scheduler}")
    

    betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, sigma = get_parameters(
        cfg.ddpm.beta_start, cfg.ddpm.beta_end, cfg.ddpm.Nt
    )

    log_name = os.path.join(cwd, cfg.output.base_dir, cfg.output.logs_dir, f"{save_name}_log.txt")
    fig_name = os.path.join(cwd, cfg.output.base_dir, cfg.output.figs_dir, save_name)
    chkpts_name = os.path.join(cwd, cfg.output.base_dir, cfg.output.models_dir, save_name)

    os.makedirs(os.path.dirname(log_name), exist_ok=True)
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    os.makedirs(os.path.dirname(chkpts_name), exist_ok=True)

    content = f'Save name: {save_name}'
    mylogger(log_name, content)
    
    tic = time.time()
    for k in range(cfg.training.num_epochs + 1):
        model.train()
        for data in data_loader: 
            x, x_hat = data[0], data[1]
            x, x_hat = x.to(device), x_hat.to(device)

            # Calculate loss
            loss = get_loss(model, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, cfg.ddpm.Nt, x, x_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
        
        if k % cfg.training.record_epoch == 0 and k > 0:
            model.eval()
            elapsed_time = time.time() - tic
            content = f'at epoch {k} the total training time is {elapsed_time:.3f} and the empirical loss is: {loss:.3f}'
            mylogger(log_name, content)
            
            if cfg.validation.enabled:
                pd, Process = solver_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, cfg.ddpm.Nt, val_pred[..., [0]])

                get_plot_sample_ddpm(cfg, cfg.ddpm.Nt, xx, yy,  GCOORD, Process, pd, val_pred, val_true, fig_name, k, centroids, triangulation)

                error_pd = myRL2_np(tensor2nump(val_true), pd)  
                content = f'at step: {k}, Relative L2 error of ddpm is: {error_pd:.3f}'
                mylogger(log_name, content)
            
            if cfg.checkpoint.save and k % cfg.training.save_frequency == 0:
                current_lr = scheduler.get_last_lr()[0]
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'current_lr': current_lr,
                    'epoch': k,
                    'loss': loss,
                    'config': cfg,
                }
                torch.save(checkpoint, f"{chkpts_name}_{k}.pth")

if __name__ == "__main__":
    main() 