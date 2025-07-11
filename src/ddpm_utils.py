import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from matplotlib import cm
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import CubicSpline
from scipy import sparse
from utils import *

from data.fem import Mesh, V_h, dtn_map
from data.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mylogger(filename, content):
    with open(filename, 'a') as fw:
        print(content, file=fw)

def myfunc(L, d0, d1, x):
    return d0 * L.dot(x) + np.power(x, 3) * d1

def myjac(L, d0, d1, x):
    return d0 * L + 3 * sparse.diags(np.square(x)) * d1

def nump2tensor(x):
    return torch.from_numpy(x).float()

def tensor2nump(x):
    return x.cpu().detach().numpy()


def myL2L(a, b):
    return torch.mean(torch.square(a - b))

def myRL2_np(a, b):
    error = np.mean( np.sqrt( np.sum((a-b)**2, axis=(1, 2, 3)) / np.sum((a)**2, axis=(1, 2, 3)) ) )
    return error

def process_unet_config(config, c0, embed_dim):
    """
    Process UNet configuration templates by substituting c0 and embed_dim values.
    """
    def process_layer_config(layer_config_template):
        processed_config = []
        for layer in layer_config_template:
            processed_layer = []
            for param in layer:
                if isinstance(param, str):
                    param = param.replace("c0", str(c0))
                    param = param.replace("embed_dim", str(embed_dim))
                    if "*" in param or "+" in param:
                        param = eval(param)
                    else:
                        param = int(param)
                processed_layer.append(param)
            processed_config.append(tuple(processed_layer))
        return processed_config
    
    unet_config = {
        'down_config': process_layer_config(config.model.unet.down_config_template),
        'mid_config': process_layer_config(config.model.unet.mid_config_template),
        'up_config': process_layer_config(config.model.unet.up_config_template)
    }
    
    return unet_config

#downsample dtn
def downsample_dtn_data(x,v_h):
    
    dtn_data, _ = dtn_map(v_h, x)
    
    dtn_torch = torch.from_numpy(dtn_data)[None, None, :, :].float()
    down_dtn = torch.nn.functional.interpolate(dtn_torch, scale_factor=0.5, mode='nearest')[0, 0].numpy()
    
    return down_dtn

#DDPM utils

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def get_parameters(beta_start, beta_end, Nt):
    betas = torch.linspace(beta_start, beta_end, steps=Nt)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, 0)
    alphas_bar_sqrt = torch.sqrt(alphas_bar)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)
    sigma = torch.sqrt(betas)
    return betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, sigma

def rand_noise(x):
    return torch.randn_like(x)

def q_x_t_cond_x_0(alphas_bar_sqrt, one_minus_alphas_bar_sqrt, x_0, t):
    # compute q(x_t|x_0) = N (mean = sqrt(alpha bar))
    noise = rand_noise(x_0)
    x_0_coeff = extract(alphas_bar_sqrt, t, x_0)
    noise_ceoff = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return x_0_coeff.to(device) * x_0.to(device) + noise_ceoff.to(device) * noise.to(device), noise.to(device)

def get_loss(model, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, Nt, x_0, x_c):
    # we sample x_t at any specific t
    bs = x_0.shape[0]
    t = torch.randint(0, Nt, size=(bs // 2,))
    t = torch.cat([t, Nt - t - 1], dim=0)
    #t = torch.randint(0, Nt, (bs,), device=device).long()

    # use q_x_t_cond_x_0 tp generate x_t from x_0
    x_noise, noise = q_x_t_cond_x_0(alphas_bar_sqrt, one_minus_alphas_bar_sqrt, x_0, t)

    noise_approx = model(x_noise.to(device), x_c.to(device), t.to(device))

    return myL2L(noise, noise_approx)

def get_parameters(beta_start, beta_end, Nt):
    betas = torch.linspace(beta_start, beta_end, steps=Nt)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, 0)
    alphas_bar_sqrt = torch.sqrt(alphas_bar)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)
    sigma = torch.sqrt(betas)
    return betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, sigma

def get_L(Nx, dim):
    dx = 1 / (1 + Nx)
    Lap = np.zeros((Nx, Nx))
    for i in range(Nx):
        Lap[i][i] = -2

    for i in range(Nx - 1):
        Lap[i + 1][i] = 1
        Lap[i][i + 1] = 1

    L = Lap / dx ** 2
    Ls = sparse.csr_matrix(L)

    if dim==2:
        Lx = sparse.kron(Ls, sparse.eye(Nx))
        Ly = sparse.kron(sparse.eye(Nx), Ls)
        Lap = Lx + Ly
    elif dim==3:
        Lx = sparse.kron(Ls, sparse.kron(sparse.eye(Nx), sparse.eye(Nx)))
        Ly = sparse.kron(sparse.eye(Nx), sparse.kron(Ls, sparse.eye(Nx)))
        Lz = sparse.kron(sparse.eye(Nx), sparse.kron(sparse.eye(Nx), Ls))
        Lap = Lx + Ly + Lz
    return Lap


def up_sample_1d(Nx_c, Nx_f, x_c, x_f, f_c):
    x_c = np.concatenate([-1.1*np.ones((1, 1)), x_c, np.ones((1, 1))*1.1], axis=0)
    x_f = np.concatenate([-1.1*np.ones((1, 1)), x_f, np.ones((1, 1))*1.1], axis=0)
    f_c = np.concatenate([np.zeros((1)), f_c, np.zeros((1))], axis=0)
    func = CubicSpline(x_c[:, 0], f_c)
    f_f = func(x_f[1:-1, 0])
    f_f = f_f[:Nx_f]

    return f_f


def up_sample_2d(Nx_c, Nx_f, x_c, x_f, f_c):

    f_f_1 = np.zeros((Nx_c, Nx_f))
    f_f_2 = np.zeros((Nx_f, Nx_f))

    for i in range(Nx_c):
        f_f_1[i, :] = up_sample_1d(Nx_c, Nx_f, x_c, x_f, f_c[i, :])

    for j in range(Nx_f):
        f_f_2[:, j] = up_sample_1d(Nx_c, Nx_f, x_c, x_f, f_f_1[:, j])

    f_f = f_f_2[:Nx_f, :Nx_f]

    return f_f



def solver_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, u_c_mat):
    bs = u_c_mat.shape[0]
    N = u_c_mat.shape[1]
    DM_mat = np.zeros((bs, N, N, 1))
    P_ls = []

    for i in range(bs):
        # a = a_mat[[i], ...]
        x_c = u_c_mat[[i], ...]

        x_generate_process = sample_backward_loop_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, x_c)
        DM_mat[i, ...] = x_generate_process[-1, ...]
        P_ls.append(x_generate_process)

    return DM_mat, P_ls

def sample_backward_ddpm_step(model, sigma, alphas, one_minus_alphas_bar_sqrt, x_t, x_c, t):
    t = torch.tensor([t])
    sigma_t = extract(sigma, t, x_t).to(device)
    xt_coeff = (1 / extract(alphas, t, x_t)).sqrt().to(device)
    eps_coeff = -(1 / extract(alphas, t, x_t).sqrt() * (
            1 - extract(alphas, t, x_t)) / extract(one_minus_alphas_bar_sqrt, t,
                                                   x_t)).to(device)
    with torch.no_grad():
        eps_theta = model(x_t.to(device), x_c.to(device), t.to(device))
    z = rand_noise(x_t).to(device)

    mean = (xt_coeff * x_t + eps_coeff * eps_theta).to(device)

    if t == 0:
        return mean
    else:
        torch.cuda.empty_cache()
        return mean + sigma_t * z

def sample_backward_loop_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, x_c):
    # x_T from guidance info
    x_T = torch.randn_like(x_c)
    x_seq = [x_T.detach().cpu()]
    x_tmp_cpu = x_T

    for i in range(Nt)[::-1]: # check here

        x_tmp = sample_backward_ddpm_step(model, sigma, alphas, one_minus_alphas_bar_sqrt, x_tmp_cpu.to(device), x_c.to(device), i)

        x_tmp_cpu = x_tmp.detach().cpu()

        x_tmp_cpu = torch.clamp(x_tmp_cpu, -clip, clip)

        del x_tmp

        torch.cuda.empty_cache()

        x_seq.append(x_tmp_cpu)

    #x_seq = np.concatenate(x_seq, axis=0)
    x_seq = torch.cat(x_seq, dim=0)

    return x_seq

def get_plot_sample_ddpm(config, Nt, xx, yy, GCOORD, Process, pd, test_hat, test_true, fig_name, step, centroids, triangulation):
    idx = 0

    md_hat = max_grad(test_hat[idx, ..., 0])
    md_true = max_grad(test_true[idx, ..., 0])
    md_ddpm = max_grad(Process[idx][-1, ..., 0])

    curr_hat = interpolate_pts(GCOORD, test_hat[idx, ..., 0].detach().cpu().numpy().flatten(), centroids)
    curr_true = interpolate_pts(GCOORD, test_true[idx, ..., 0].detach().cpu().numpy().flatten(), centroids)
    curr_ddpm = interpolate_pts(GCOORD, pd[idx, ..., 0].flatten(), centroids)
    print("interpolated shapes", curr_hat.shape, curr_true.shape, curr_ddpm.shape)

    print('max grad', md_hat, md_true, md_ddpm)

    fig, ax = plt.subplots(1, 11, figsize=(20, 3))

    ax[0].contourf(xx, yy, Process[idx][-1, ..., 0], 36, cmap=cm.jet)
    ax[0].set_title(r'final')

    for i in range(10):
        ax[i+1].contourf(xx, yy, Process[idx][int(Nt / 10) * i, ..., 0], 36, cmap=cm.jet)
        ax[i+1].set_title(r'$sample$' + f"-{int(Nt / 10) * i}")

    for axs in ax.flat:
        axs.set_xticks([])
        axs.set_yticks([])

    plt.tight_layout()
    fig.savefig(fig_name + '_epoch_step_' + str(step) + '_generate.jpg')

    fig2, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].tripcolor(triangulation, curr_hat, edgecolors='none')
    ax[0, 0].axis('off')
    ax[0, 0].set_title('x_hat')

    cp = ax[0, 1].tripcolor(triangulation, curr_true, edgecolors='none')
    ax[0, 1].axis('off')
    ax[0, 1].set_title('x_true')
    plt.colorbar(cp)

    cp = ax[1, 0].tripcolor(triangulation, curr_ddpm, edgecolors='none')
    ax[1, 0].axis('off')
    ax[1, 0].set_title('x_pred')
    plt.colorbar(cp)

    cp = ax[1, 1].tripcolor(triangulation, np.abs(curr_ddpm - curr_true), edgecolors='none')
    ax[1, 1].axis('off')
    ax[1, 1].set_title('error')
    plt.colorbar(cp)
    fig2.savefig(fig_name + '_epoch_step_' + str(step) + '_pd.jpg')

    plt.show()

def max_grad(f):
    #f = f.detach().cpu()
    N = f.shape[0]
    max_d = 1e-4 * torch.ones(1,).to(device)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            dx = torch.abs(f[i, j] - f[i - 1, j]) * N
            dy = torch.abs(f[i, j] - f[i, j - 1]) * N

            max_d = torch.max(max_d, dx)
            max_d = torch.max(max_d, dy)

            # max_d = np.maximum(max_d, dx)
            # max_d = np.maximum(max_d, dy)

    return max_d

def plot_resnet_results(config, xx, yy, GCOORD, x_hat, x_true, x_pred, fig_name, step, centroids, triangulation):
    idx = 0

    curr_hat = interpolate_pts(GCOORD, x_hat[idx, ..., 0].detach().cpu().numpy().flatten(), centroids)
    curr_true = interpolate_pts(GCOORD, x_true[idx, ..., 0].detach().cpu().numpy().flatten(), centroids)
    curr_pred = interpolate_pts(GCOORD, x_pred[idx, ..., 0].detach().cpu().numpy().flatten(), centroids)
    
    # Calculate error
    error = np.abs(curr_pred - curr_true)
    
    # Create figure with 2x2 subplots
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    
    # Plot coarse solution
    cp1 = ax[0, 0].tripcolor(triangulation, curr_hat, edgecolors='none')
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Coarse Solution')
    plt.colorbar(cp1, ax=ax[0, 0])
    
    # Plot fine solution (ground truth)
    cp2 = ax[0, 1].tripcolor(triangulation, curr_true, edgecolors='none')
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Fine Solution (Ground Truth)')
    plt.colorbar(cp2, ax=ax[0, 1])
    
    # Plot predicted solution
    cp3 = ax[1, 0].tripcolor(triangulation, curr_pred, edgecolors='none')
    ax[1, 0].axis('off')
    ax[1, 0].set_title('ResNet Prediction')
    plt.colorbar(cp3, ax=ax[1, 0])
    
    # Plot error
    cp4 = ax[1, 1].tripcolor(triangulation, error, edgecolors='none')
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Absolute Error')
    plt.colorbar(cp4, ax=ax[1, 1])
    
    # Remove ticks for cleaner visualization
    for axs in ax.flat:
        axs.set_xticks([])
        axs.set_yticks([])
    
    plt.tight_layout()
    fig.savefig(f'{fig_name}_step_{step}_results.jpg')
    plt.show()