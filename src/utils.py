import torch
import math

def num2str_deciaml(x):
    s = str(x)
    c = ''
    for i in range(len(s)):
        if s[i] == '0':
            c = c + 'z'
        elif s[i] == '.':
            c = c + 'p'
        elif s[i] == '-':
            c = c + 'n'
        else:
            c = c + s[i]

    return c

def tensor2nump(x):
    return x.cpu().detach().numpy()

def make_tensor(*args):
    return [torch.from_numpy(arg).float().to(device) for arg in args]
