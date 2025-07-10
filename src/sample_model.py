import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from utils_eit import *
# from data_utils import *
# from train_utils import *
import functools
from UNet import UNet
import os
from config_eit import *

if __name__ == "__main__":
    # define grid
    Nx_f = 64
    dx = 1 / (Nx_f + 1)
    points_x = np.linspace(dx, 1 - dx, Nx_f).T
    xx, yy = np.meshgrid(points_x, points_x)

    cwd = os.getcwd()

    # chosen_nums = [4100 + i for i in range(4)]
    
    
    #load testing data
    
    npy_name = "/blue/chunmei.wang/amit.bhat/my_score_sde/downscaling_DDPM/Generative-downsscaling-PDE-solvers/eit_data/sl_images_5_150_n_01.npz"
    # npy_name = "/blue/chunmei.wang/amit.bhat/my_score_sde/downscaling_DDPM/Generative-downsscaling-PDE-solvers/eit_data/eit_images_405_350.npz"
    data = np.load(npy_name)

    imgs_true = data["imgs_true"][Ntr+Nval:, ...]
    imgs_pred = data["imgs_pred"][Ntr+Nval:, ...]



    # imgs_true = data["imgs_true"][chosen_nums, ...]
    # imgs_pred = data["imgs_pred"][chosen_nums, ...]

    # imgs_true = np.repeat(imgs_true, 10, axis=0)
    # imgs_pred = np.repeat(imgs_pred, 10, axis=0)


    test_c = torch.from_numpy(imgs_pred).float().reshape(-1, 64, 64, 1)
    test_f = torch.from_numpy(imgs_true).float().reshape(-1, 64, 64, 1)

    print(imgs_true.shape, imgs_pred.shape, flush=True)

    clip = torch.max(torch.abs(test_f)) * clip_coeff

    # test_c, test_f = imgs_pred[Ntr:Ntr+Nval, ...], imgs_true[Ntr:Ntr+Nval, ...]

    print(clip, test_c.shape, test_f.shape, flush=True)
    
    
     #################### load UNet ##############################################
    load_iter = 20000
    if model_name=='UNet':
        model = UNet(Nt, embed_dim, Down_config, Up_config, Mid_config).to(device)
    elif model_name == 'UNet_attn':
        model = UNet_attn(Nt, embed_dim, Down_config, Up_config, Mid_config).to(device)

    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('UNet num of parameters', model_trainable_params)

    ### define diffusion parameters
    betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, sigma = get_parameters(beta_start, beta_end, Nt)

    L_f = get_L(Nx_f, dim=2)
    myfunc_f, myjac_f = functools.partial(myfunc, L_f, d0, d1), functools.partial(myjac, L_f, d0, d1)

    generate_method = 'ddpm'

    ### load name
    save_name = 'DM_P_2D_' + model_name + '_c0_' + str(c0) + '_Nt_' + str(Nt) + '_Ntr_' + str(Ntr) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
        m) + '_d0_' + num2str_deciaml(
        d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(d2) + '_alp_' + num2str_deciaml(
        alpha) + '_tau_' + num2str_deciaml(tau)
    
    print(save_name)

    chkpts_name = cwd + '/mdls/' + save_name + str(load_iter) + '.pth'
    fig_name = cwd + '/figs/' + save_name

    checkpoint = torch.load(chkpts_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    ###############################################################################################
    
    ############# Diffusion model result ####################################

    ### ddpm ###
    tic_ddpm = time.time()
    pd, _ = solver_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, test_c[..., [0]])
    toc = time.time() - tic_ddpm
    avg_time_ddpm = toc / test_f.shape[0]

    npy_name = "/blue/chunmei.wang/amit.bhat/my_score_sde/downscaling_DDPM/Generative-downsscaling-PDE-solvers/eit_data/ddpm_samples_sl_150_n01_orig_weights"
    # npy_name = "/blue/chunmei.wang/amit.bhat/my_score_sde/downscaling_DDPM/Generative-downsscaling-PDE-solvers/eit_data/ddpm_samples_circs_350"
    # # print(pd.shape)
    np.save(npy_name, pd)
    # pd = np.load(npy_name)

    # ### ddim ###
    # tic_ddim = time.time()
    # pd_ddim, _ = solver_ddim(model, clip, alphas_bar, test_f[..., [0]], test_c[..., [0]], my_t_list_skip)
    # toc = time.time() - tic_ddim
    # avg_time_ddim = toc / test_f.shape[0]

    error_pd = myRL2_np(tensor2nump(test_f), pd)
    # error_pd_ddim = myRL2_np(tensor2nump(test_r[..., [1]]), pd_ddim)
    # error_fno = myRL2_np(tensor2nump(test_r[..., [1]]), fno_pd)
    error_c = myRL2_np(tensor2nump(test_f), tensor2nump(test_c))
    # error_f = myRL2_np(tensor2nump(test_r[..., [1]]), tensor2nump(test_f[..., [1]]))

    content = 'compute time of coarse solver is: %3f, Relative L2 error of coarse solver is: %3f, ddpm is: %3f'  % (avg_time_ddpm, error_c, error_pd)
    
    
    # content1 = 'compute time of coarse solver is: %3f, fine solver is: %3f, FNO is: %3f, ddpm is: %3f, ddim is: %3f' % (
    #     tc, tf, avg_time_fno, avg_time_ddpm, avg_time_ddim)

    # content2 = 'compute time of coarse+ft is is: %3f, FNO+ft is: %3f, PGDM is: %3f' % (
    #     tc + avg_ft_time, avg_time_fno + avg_ft_time, avg_time_ddim + avg_ft_time)

    # content3 = 'Relative L2 error of coarse solver is: %3f, fine solver is: %3f, FNO is: %3f, ddpm is: %3f, ddim is: %3f' % (
    #     error_c, error_f, error_fno, error_pd, error_pd_ddim)
    


    # print(content1)
    # print(content2)
    # print(content3)
    # print(content, flush=True)
    # content = 'compute time of coarse solver is: %3f' % (avg_time_ddpm)
    print(content)

    # #write a new plot function for result
    im_arr_true = []
    im_arr_coarse = []
    im_arr_sample = []
    start = 0
    num_images = 10
    for i in range(start, start+num_images):
        img_true = test_f[i, ...]
        img_coarse = test_c[i, ...] 
        image =  pd[i, ...]
        # image = sample["samples"].reshape((128, 128, 1))
        # image = datasets.central_crop(image, 180)
        # image = datasets.resize_small(image, 64)
        im_arr_true.append(img_true)
        im_arr_coarse.append(img_coarse)
        im_arr_sample.append(image)

    # batch_size = 8, lr = 2e-5

    vmin = np.min(im_arr_true) 
    vmax = np.max(im_arr_true)

    print(vmin, vmax)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    last_mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    last_mappable.set_array([])  # Dummy array
     
    rows = 2
    cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    fig.suptitle('True Centroid Images', fontsize=16)

    for i in range(num_images):
        ax = axes[i // cols, i % cols]
        im = ax.imshow(im_arr_true[i])
        ax.axis('off')
        
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
    cbar = fig.colorbar(last_mappable, cax=cax)
    cbar.set_label('Colorbar Label', fontsize=12)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig('true_images_sl_150_n01_orig_weights.png', dpi=300, bbox_inches='tight')
    plt.show()

    vmin = np.min(im_arr_coarse) 
    vmax = np.max(im_arr_coarse) 

    print(vmin, vmax)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    last_mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    last_mappable.set_array([])

     # batch_size = 8, lr = 2e-5
    rows = 2
    cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    fig.suptitle('EIT Initial Solutions', fontsize=16)

    for i in range(num_images):
        ax = axes[i // cols, i % cols]
        im = ax.imshow(im_arr_coarse[i])
        ax.axis('off')
        
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
    cbar = fig.colorbar(last_mappable, cax=cax)
    cbar.set_label('Colorbar Label', fontsize=12)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig('eit_initial_solutions_sl_150_n01_orig_weights.png', dpi=300, bbox_inches='tight')
    plt.show()


    vmin = np.min(im_arr_sample)
    vmax = np.max(im_arr_sample)

    print(vmin, vmax)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    last_mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    last_mappable.set_array([])

    # batch_size = 8, lr = 2e-5
    rows = 2
    cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    fig.suptitle('DDPM Centroids Samples, 400 noise scales', fontsize=16)

    for i in range(num_images):
        ax = axes[i // cols, i % cols]
        im = ax.imshow(im_arr_sample[i])
        ax.axis('off')
        
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
    cbar = fig.colorbar(last_mappable, cax=cax)
    cbar.set_label('Colorbar Label', fontsize=12)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig('ddpm_samples_sl_150_n01_orig_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    
    
    