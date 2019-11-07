import numpy as np
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json

def save_args(args, path):
    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    print('Saved args to {}'.format(f.name))

def load_args(path, verbose=False):
    with open(path, 'r') as f:
        args = json.load(f)
    if verbose: print(args)
    return args

def get_gt_densities(flows, opt):
    inp = flows[0] * opt.scaler_dict[opt.scales[0]]
    scale0 = opt.scales[0]
    out, masks = [], []

    for i, f in enumerate(flows[1:]):
        scale_ = opt.scales[i+1]
        inp_ = F.upsample(inp, scale_factor=scale_//scale0)
        masks.append(inp_ != 0)
        f0 = inp_ + 1e-9
        f_ = f*opt.scaler_dict[opt.scales[i+1]]
        out.append(f_/f0)
    return out, masks

def read_raw_data(datapath):
    files = [f for f in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, f))]
    files.sort()
    data = []
    for file in files:
        print('read %s' % file)
        temp = np.load(os.path.join(datapath, file))
        data_filtered = []
        for d in temp:
            occupied_ratio = (np.prod(d.shape) - (d==0).sum()) / np.prod(d.shape)
            if occupied_ratio > 0:
                data_filtered.append(d)
        data.append(np.asarray(data_filtered))
    return data

def get_lapprob_dataloader(datapath, args, batch_size=2, mode='train', test=False):
    datapath = os.path.join(datapath, mode)
    if test: cuda = False
    else: cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    Xs = list()
    for scale in args.scales:
        Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X_%d.npy'%scale)), 1)) / args.scaler_dict[scale])
    ext = Tensor(np.load(os.path.join(datapath, 'ext.npy')))
    Xs.append(ext)
    
    data = torch.utils.data.TensorDataset(*Xs)
    print('{} samples, shapes are {}'.format(len(data), [tensor.shape for tensor in data.tensors]))
    for scale in args.scales:
        if mode == 'train':
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    return dataloader

def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))
    return total_num
