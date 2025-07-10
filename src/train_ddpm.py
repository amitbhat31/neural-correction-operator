import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from utils_eit import *
import functools

from UNet import UNet

import os


@click.command()
@click.option('--img-size', type=int, required=True, help='size of output image')
@click.option('--dataset', type=int, required=True, help='name of dataset: 0:circs or 1:sl')
@click.option('--noise', type=float, required=True, help='noise level')
@click.option('--lr-max', type=float, required=True, help='maximal learning rate for cosine scheduler')
@click.option('--lr-min', type=float, required=True, help='minimal learning rate for cosine scheduler')
@click.option('--data-path', type=str, required=True, help='path to the data')
def main(
    img_size: int,
    dataset: int,
    noise: float,
    lr_max: float,
    lr_min: float,
    data_path: str,
):
    # define grid
    Nx_f = img_size
    dx = 1 / (Nx_f + 1)
    points_x = np.linspace(dx, 1 - dx, Nx_f).T
    xx, yy = np.meshgrid(points_x, points_x)

    cwd = os.getcwd()
    print(cwd, flush=True)

    torch.backends.cudnn.benchmark = True

    data_name = 'circs'
    if dataset == 1:
        data_name = 'sl'
    

    save_name = 'DDPM_' + data_name + '_res_'+ str(img_size) + '_noise_' + num2str_deciaml(noise) + '_Ntr_' + str(Ntr) +  '_maxLR_' + num2str_deciaml(lr_max) + '_minLR_' + num2str_deciaml(lr_min) + '_timesteps_' + str(Nt)

    print(save_name)

    ### load training data ####
    #convert numpy to torch
    # npy_name = "/blue/chunmei.wang/amit.bhat/my_score_sde/downscaling_DDPM/Generative-downsscaling-PDE-solvers/eit_data/sl_images_5_150_n_05.npz"
    # npy_name = "/blue/chunmei.wang/amit.bhat/my_score_sde/downscaling_DDPM/Generative-downsscaling-PDE-solvers/eit_data/eit_images_405_350.npz"
    data = np.load(data_path)

    imgs_true = data["imgs_true"][:Nim, ...]
    imgs_pred = data["imgs_pred"][:Nim, ...]


    imgs_true = torch.from_numpy(imgs_true).float().reshape(Nim, 64, 64, 1)
    imgs_pred = torch.from_numpy(imgs_pred).float().reshape(Nim, 64, 64, 1)

    
    print(imgs_true.shape, imgs_pred.shape, flush=True)

    clip = torch.max(torch.abs(imgs_pred)) * clip_coeff
    
    # this is fine
    dataset = []
    for i in range(Ntr):
        #contains u_c and u_f
        tmp_ls = []
        tmp_ls.append(imgs_true[i, ...])
        tmp_ls.append(imgs_pred[i, ...])
        
        dataset.append(tmp_ls)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    content = 'loaded training data'
    print(content, flush=True)
    
    
#     insert testing data here
# slice original eit_images file in train/test/val; may need to include 
    test_c, test_f = imgs_pred[Ntr:Ntr+Nval, ...], imgs_true[Ntr:Ntr+Nval, ...]

    print(test_c.shape, test_f.shape, flush=True)
    

    if model_name=='UNet':
        model = UNet(Nt, embed_dim, Down_config, Up_config, Mid_config).to(device)
    elif model_name == 'UNet_attn':
        model = UNet_attn(Nt, embed_dim, Down_config, Up_config, Mid_config).to(device)

    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('UNet num of parameters', model_trainable_params, flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.75)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)
    
     ### define diffusion parameters ##################################################
    betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, sigma = get_parameters(beta_start, beta_end, Nt)

    # L_f = get_L(Nx_f, dim=2)
    # myfunc_f, myjac_f = functools.partial(myfunc, L_f, d0, d1), functools.partial(myjac, L_f, d0, d1)
    ################################################################################################
    log_name = cwd + '/runs/logs/' + save_name + '_log.txt'
    fig_name = cwd + '/runs/figs/' + save_name
    chkpts_name = cwd + '/runs/mdls/' + save_name

    print(log_name, fig_name, chkpts_name, flush=True)

    content = 'start training'
    mylogger(log_name, content)
    
    ### training loop
    tic = time.time()
    for k in range(num_epoch+1):
        model.train()
        for data in data_loader: 
            x, x_c = data[0], data[1]
            x, x_c = x.to(device), x_c.to(device)

            #have to rewrite loss function
            loss = get_loss(model, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, Nt, x, x_c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
        
        if k % record_epoch == 0 and k>0:
            model.eval()
            ### record time and loss ###
            elapsed_time = time.time() - tic
            content = 'at epoch %d the total training time is %3f and the empirical loss is: %3f' % (k, elapsed_time, loss)
            print(content, flush=True)
            mylogger(log_name, content)
            
            #insert validation here by solving at intermediate ddpm step
        
            pd, Process = solver_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, test_c[..., [0]])

            get_plot_sample_ddpm(Nt, xx, yy, Process, pd, test_c, test_f, fig_name, k)

            error_pd = myRL2_np(tensor2nump(test_f), pd)
            error_c = myRL2_np(tensor2nump(test_f), tensor2nump(test_c))
            error_f = myRL2_np(tensor2nump(test_f), tensor2nump(test_f))

            content = 'at step: %d, Relative L2 error of coarse solver is: %3f, fine solver is: %3f, ddim is: %3f' % (
                k, error_c, error_f,  error_pd)

            mylogger(log_name, content)

            print(content, flush=True)
            
            if k%2000 == 0:
                ### save model ###
                current_lr = scheduler.get_last_lr()[0]
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # Saving scheduler state is optional but can be useful
                    'current_lr': current_lr,
                    'epoch': k,  # Optional, add if you want to keep track of epochs
                    'loss': loss,
                    # ... include any other things you want to save
                }
                torch.save(checkpoint, chkpts_name +'_'+ str(k) + '.pth')
        

if __name__ == "__main__":
    main()

        
    
    