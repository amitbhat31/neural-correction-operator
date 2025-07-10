import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):

    def __init__(self, config, unet_config):
        # comments here

        super().__init__()

        self.Nt = config.ddpm.Nt
        #extract config params
        embed_dim = config.model.embed_dim
        down = unet_config['down_config']
        mid = unet_config['mid_config']
        up = unet_config['up_config']
        
        self.time_embed = nn.Sequential(TimeEmbedding(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))

        self.trunk = nn.ModuleDict()

        for b in range(config.model.num_blocks):
            self.trunk[f'down_conv_{b}'] = nn.Conv2d(down[b][0], down[b][1], kernel_size=down[b][2], stride=down[b][3], bias=False)
            self.trunk[f'down_fc_{b}'] = nn.Linear(embed_dim, down[b][1])
            self.trunk[f'down_groupnorm_{b}'] = nn.GroupNorm(down[b][4], num_channels = down[b][1])

        self.trunk[f'mid_conv'] = nn.Conv2d(mid[0][0], mid[0][1], mid[0][2], mid[0][3], bias=False)
        self.trunk[f'mid_fc'] = nn.Linear(embed_dim, mid[0][1])
        self.trunk[f'mid_groupnorm'] = nn.GroupNorm(mid[0][4], num_channels = mid[0][1])

        for b, i in zip(range(config.model.num_blocks-1, 0, -1), range(config.model.num_blocks-1)):
            self.trunk[f'up_conv_{b}'] = nn.ConvTranspose2d(up[i][0], up[i][1], kernel_size=up[i][2], stride=up[i][3], output_padding=up[i][5], bias=False)
            self.trunk[f'up_fc_{b}'] = nn.Linear(embed_dim, up[i][1])
            self.trunk[f'up_groupnorm_{b}'] = nn.GroupNorm(up[i][4], num_channels = up[i][1])
        
        self.trunk['up_conv_0'] = nn.ConvTranspose2d(up[2][0], up[2][1], kernel_size=up[2][2], stride=up[2][3])

        self.act = nn.ReLU()

    def forward(self, x, x_hat, t):

        embed = self.act(self.time_embed(t/self.Nt))

        x = torch.cat([x, x_hat], dim=-1)
        x = x.permute(0, 3, 1, 2)

        # down layers
        x0 = self.trunk['down_conv_0'](x)
        x0 += self.trunk['down_fc_0'](embed)[..., None, None]
        x0 = self.trunk['down_groupnorm_0'](x0)
        x0 = self.act(x0)

        x1 = self.trunk['down_conv_1'](x0)
        x1 += self.trunk['down_fc_1'](embed)[..., None, None]
        x1 = self.trunk['down_groupnorm_1'](x1)
        x1 = self.act(x1)

        x2 = self.trunk['down_conv_2'](x1)
        x2 += self.trunk['down_fc_2'](embed)[..., None, None]
        x2 = self.trunk['down_groupnorm_2'](x2)
        x2 = self.act(x2)

        x3 = self.trunk['down_conv_3'](x2)
        x3 += self.trunk['down_fc_3'](embed)[..., None, None]
        x3 = self.trunk['down_groupnorm_3'](x3)
        x3 = self.act(x3)

        # mid layer
        x_mid = self.trunk['mid_conv'](x3)
        x_mid += self.trunk['mid_fc'](embed)[..., None, None]
        x_mid = self.trunk['mid_groupnorm'](x_mid)

        # up layers
        x_up = self.trunk['up_conv_3'](x_mid)
        x_up += self.trunk['up_fc_3'](embed)[..., None, None]
        x_up = self.trunk['up_groupnorm_3'](x_up)
        x_up = self.act(x_up)

        x_up = self.trunk['up_conv_2'](torch.cat([x_up, x2], dim=1))
        x_up += self.trunk['up_fc_2'](embed)[..., None, None]
        x_up = self.trunk['up_groupnorm_2'](x_up)
        x_up = self.act(x_up)

        x_up = self.trunk['up_conv_1'](torch.cat([x_up, x1], dim=1))
        x_up += self.trunk['up_fc_1'](embed)[..., None, None]
        x_up = self.trunk['up_groupnorm_1'](x_up)
        x_up = self.act(x_up)

        x_up = self.trunk['up_conv_0'](torch.cat([x_up, x0], dim=1))
        
        return x_up.permute(0, 2, 3, 1)
    
class TimeEmbedding(nn.Module):

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Weights randomly sampled and not optimized.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        
        

        



        
