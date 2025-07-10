import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from utils import *
from ddpm_utils import *

# Import the ResNet model
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

    # Define grid - spanning [-0.5, 0.5] × [-0.5, 0.5]
    Nx_f = cfg.data.img_size
    dx = 1 / (Nx_f + 1)
    points_x = np.linspace(-0.5 + dx, 0.5 - dx, Nx_f).T
    xx, yy = np.meshgrid(points_x, points_x)

    cwd = os.getcwd()
    print(f"Working directory: {cwd}", flush=True)

    # Determine dataset name
    data_name = cfg.data.dataset_type
    
    # Create save name using template
    save_name = cfg.output.name_template.format(
        dataset=data_name,
        bfgs_iters=cfg.data.bfgs_iters,
        img_size=cfg.data.img_size,
        noise=num2str_deciaml(cfg.data.noise_level),
        lr=num2str_deciaml(cfg.training.lr),
        train_samples=cfg.data.train_samples
    )
    print(f"Save name: {save_name}")

    # Load training data
    data = np.load(cfg.data.data_path)
    imgs_true = data["imgs_true"][:cfg.data.total_samples, ...]
    imgs_pred = data["imgs_pred"][:cfg.data.total_samples, ...]

    imgs_true = torch.from_numpy(imgs_true).float().reshape(cfg.data.total_samples, cfg.data.img_size, cfg.data.img_size, 1)
    imgs_pred = torch.from_numpy(imgs_pred).float().reshape(cfg.data.total_samples, cfg.data.img_size, cfg.data.img_size, 1)

    print(f"Data shapes: {imgs_true.shape}, {imgs_pred.shape}", flush=True)

    # Create training dataset
    dataset = []
    for i in range(cfg.data.train_samples):
        tmp_ls = []
        tmp_ls.append(imgs_true[i, ...])  # x (fine solution)
        tmp_ls.append(imgs_pred[i, ...])  # x_hat (coarse solution)
        dataset.append(tmp_ls)
    
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.system.num_workers)

    print('Loaded training data', flush=True)

    # Create validation dataset
    val_data = []
    for i in range(cfg.data.train_samples, cfg.data.train_samples + cfg.data.val_samples):
        tmp_ls = []
        tmp_ls.append(imgs_true[i, ...])  # x (fine solution)
        tmp_ls.append(imgs_pred[i, ...])  # x_hat (coarse solution)
        val_data.append(tmp_ls)

    val_loader = DataLoader(val_data, batch_size=cfg.training.batch_size, num_workers=cfg.system.num_workers)
    

    # Create model
    if cfg.model.name == "resNet":
        model = ResNet18(cfg).to(device)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{cfg.model.name} number of parameters: {model_trainable_params}', flush=True)

    # Setup optimizer and scheduler
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

    print(f"Log: {log_name}")
    print(f"Figs: {fig_name}")
    print(f"Checkpoints: {chkpts_name}", flush=True)

    content = 'start training'
    mylogger(log_name, content)
    
    # Training loop
    tic = time.time()
    for k in range(cfg.training.num_epochs + 1):
        model.train()
        for data in data_loader: 
            x, x_hat = data[0], data[1]  # x is fine solution, x_hat is coarse solution
            x, x_hat = x.to(device), x_hat.to(device)

            # Calculate loss
            loss = myL2L(x, model(x_hat))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        if k % cfg.training.record_epoch == 0 and k > 0:
            model.eval()
            # Record time and loss
            elapsed_time = time.time() - tic
            content = f'at epoch {k} the total training time is {elapsed_time:.3f} and the empirical loss is: {loss:.3f}'
            print(content, flush=True)
            mylogger(log_name, content)

            # Validation step
            if cfg.validation.enabled:
                val_loss = 0.
                with torch.no_grad():
                    for data in val_loader:
                        x, x_hat = data[0], data[1]  # x is fine solution, x_hat is coarse solution
                        x, x_hat = x.to(device), x_hat.to(device)
                        outputs = model(x_hat)
                        loss_val = myL2L(x, outputs)
                        val_loss += loss_val.item()
                    avg_loss = val_loss / len(val_loader)

                print(f'at epoch {k} the validation loss is: {avg_loss:.3f}', flush=True)

                # Plot sample results
                data_1 = next(iter(val_loader))
                x, x_hat = data_1[0], data_1[1]  # x is fine solution, x_hat is coarse solution
                x, x_hat = x.to(device), x_hat.to(device)
                x_pred = model(x_hat)

                plot_resnet_results(xx, yy, x_hat, x, x_pred, fig_name, k)
            
            # Save checkpoint
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