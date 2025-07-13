import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import scipy.io as sio
import matplotlib.tri as tri

from utils import *
from ddpm_utils import *
from models.resnet import ResNet18, BasicBlock

@hydra.main(config_path="../configs", config_name="config_resnet")
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

    # loading the file containing the mesh
    mat_fname  = cfg.data.mesh_path
    mat_contents = sio.loadmat(mat_fname)

    p = mat_contents['p']
    t = mat_contents['t']-1 
    centroids = np.mean(p[t], axis=1)
    triangulation = tri.Triangulation(p[:,0], p[:,1], t)

    data_name = cfg.data.dataset_type
    
    save_name = cfg.output.name_template.format(
        dataset=data_name,
        bfgs_iters=cfg.data.bfgs_iters,
        img_size=cfg.data.img_size,
        noise=num2str_deciaml(cfg.data.noise_level),
        lr=num2str_deciaml(cfg.training.lr),
        train_samples=cfg.data.train_samples
    )

    data = np.load(cfg.data.data_path)
    imgs_true = data["imgs_true"][:cfg.data.total_samples, ...]
    imgs_pred = data["imgs_pred"][:cfg.data.total_samples, ...]

    imgs_true = torch.from_numpy(imgs_true).float().reshape(cfg.data.total_samples, cfg.data.img_size, cfg.data.img_size, 1)
    imgs_pred = torch.from_numpy(imgs_pred).float().reshape(cfg.data.total_samples, cfg.data.img_size, cfg.data.img_size, 1)


    dataset = []
    for i in range(cfg.data.train_samples):
        tmp_ls = []
        tmp_ls.append(imgs_true[i, ...])  # x (fine solution)
        tmp_ls.append(imgs_pred[i, ...])  # x_hat (rough solution)
        dataset.append(tmp_ls)
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.system.num_workers)

    val_data = []
    for i in range(cfg.data.train_samples, cfg.data.train_samples + cfg.data.val_samples):
        tmp_ls = []
        tmp_ls.append(imgs_true[i, ...]) 
        tmp_ls.append(imgs_pred[i, ...])  
        val_data.append(tmp_ls)
    val_loader = DataLoader(val_data, batch_size=cfg.training.batch_size, num_workers=cfg.system.num_workers)
    
    if cfg.model.name == "resNet":
        model = ResNet18(cfg).to(device)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    if cfg.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.training.scheduler_step_size, 
            gamma=cfg.training.scheduler_gamma
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.training.scheduler}")

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

            loss = myL2L(x, model(x_hat))

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
                val_loss = 0.
                with torch.no_grad():
                    for data in val_loader:
                        x, x_hat = data[0], data[1] 
                        x, x_hat = x.to(device), x_hat.to(device)
                        outputs = model(x_hat)
                        loss_val = myL2L(x, outputs)
                        val_loss += loss_val.item()
                    avg_loss = val_loss / len(val_loader)

                content = f'at epoch {k} the validation loss is: {avg_loss:.3f}'    
                mylogger(log_name, content)

                data_1 = next(iter(val_loader))
                x, x_hat = data_1[0], data_1[1] 
                x, x_hat = x.to(device), x_hat.to(device)
                x_pred = model(x_hat)

                plot_resnet_results(cfg, xx, yy, GCOORD, x_hat, x, x_pred, fig_name, k, centroids, triangulation)
            
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