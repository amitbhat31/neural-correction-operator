import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from utils import *
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

    test_hat = torch.from_numpy(imgs_pred).float().reshape(-1, cfg.data.img_size, cfg.data.img_size, 1)
    test_true = torch.from_numpy(imgs_true).float().reshape(-1, cfg.data.img_size, cfg.data.img_size, 1)

    clip = torch.max(torch.abs(test_true)) * cfg.sampling.clip_coeff

    test_data = []
    for i in range(test_true.shape[0]):
        tmp_ls = []
        tmp_ls.append(test_true[i, ...])   
        tmp_ls.append(test_hat[i, ...])
        test_data.append(tmp_ls)
    test_loader = DataLoader(test_data, batch_size=cfg.sampling.batch_size, num_workers=cfg.system.num_workers)

    if cfg.model.name == "resNet":
        model = ResNet18(cfg).to(device)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")


    chkpts_name = cfg.sampling.model_path
    print(f"Loading model from: {chkpts_name}")

    checkpoint = torch.load(chkpts_name, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    res = np.zeros((test_true.shape[0], N_pts, N_pts, 1))

    tic_start = time.time()
    for i, (x, x_hat) in enumerate(test_loader):
        x, x_hat = x.to(device), x_hat.to(device)
        x_pred = model(x_hat)
        res[i, ...] = x_pred.detach().cpu().numpy()
    
    tic_total = time.time() - tic_start
    avg_time = tic_total / test_true.shape[0]

    print(f"Average inference time per sample: {avg_time:.4f} seconds")

    print(f"Saving results to: {samples_path}")
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)
    np.save(samples_path, res)

if __name__ == "__main__":
    main() 