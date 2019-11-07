import os
import sys
import warnings
import numpy as np
import random
import argparse
import warnings
from datetime import datetime
from PIL import Image
import pickle
import json
import time
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .utils.metrics import get_RMSE, get_MAE, get_MSE, get_MRE
from .utils.data_process import print_model_parm_nums, get_lapprob_dataloader, save_args, get_gt_densities
from .model import UrbanPy, weights_init_normal
from .layers import batch_kl, reverse_density
from .args import get_args

tk = time.time()
opt = get_args(sys.argv[1:])
print(opt)
torch.manual_seed(opt.seed)
random.seed(0)
np.random.seed(0)

warnings.filterwarnings('ignore')
# path for saving model
res_str = ','.join([str(i) for i in opt.n_residuals])
islocal = '_local' if opt.islocal else '_nonlocal'
save_path = 'saved_model/{}/{}-{}/urbanpy/{}-{}-{}-r{}-{}-{}-c{}{}{}'.format(opt.dataset,
                                                     opt.from_reso,
                                                     opt.to_reso,                                   
                                                     res_str,
                                                     opt.base_channels,
                                                     opt.ext_flag,
                                                     opt.n_res_propnet,
                                                     opt.loss_weights,
                                                     opt.to_reso,
                                                     opt.compress,
                                                     islocal,
                                                     opt.name)
os.makedirs(save_path, exist_ok=True)

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# initial model
model = UrbanPy(in_channels=opt.channels,
                out_channels=opt.channels,
                img_width=opt.from_reso,
                img_height=opt.from_reso,
                n_residual_blocks=opt.n_residuals,
                base_channels=opt.base_channels,
                ext_dim=opt.ext_dim,
                ext_flag=opt.ext_flag,
                scales=opt.scalers,
                N=opt.N,
                n_res=opt.n_res_propnet,
                islocal=opt.islocal,
                compress=opt.compress)
if opt.resume:  #resume training from breakpoint
    for epoch in range(opt.start_from):
        if epoch % opt.harved_epoch == 0 and epoch != 0:
            opt.lr /= 2
    save_args(opt, save_path)
    model.load_state_dict(torch.load('{}/final_model.pt'.format(save_path)))
else:
    model.apply(weights_init_normal)
    
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
opt.param_num = print_model_parm_nums(model, 'UrbanPy')
save_args(opt, save_path)
criterion = nn.MSELoss()

if cuda:
    model.cuda()
    criterion.cuda()

# Load training set and validation set
datapath = os.path.join('data', opt.dataset)
train_dataloader = get_lapprob_dataloader(
    datapath, opt, batch_size=opt.batch_size, mode='train')
valid_dataloader = get_lapprob_dataloader(
    datapath, opt, batch_size=16, mode='valid')

geo_dict = dict()
with h5py.File('data/embedded_geos.h5', 'r') as hf:
    for key in opt.scales:
        geo_dict[key] = hf.get(str(key)).value
geo_dict = {key:Tensor(geo_data).expand(opt.batch_size,-1,-1,-1) for key, geo_data in geo_dict.items()}

# Optimizers
lr = opt.lr
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, betas=(opt.b1, opt.b2))

# Loss functions
def compute_loss(predicts, ys, weights=[1,1,1]):
    batch_size = len(predicts[0])
    assert len(outs) == len(ys),\
            'out len: {}, flow len: {}'.format(len(outs), len(ys))
    losses = [criterion(yhat, y)*weights[i] 
              for i, (yhat, y) in enumerate(zip(predicts, ys))]
    return sum(losses), torch.sqrt(torch.stack(losses)).data.cpu().numpy()

def compute_kl_loss(predicts, ys, masks, scales, weights=[1,1,1,1]):
    losses = [batch_kl(yhat, y, scale, mask)*weights[i]
              for i, (yhat, y, scale, mask) in enumerate(zip(predicts, ys, scales, masks))]
    return sum(losses), torch.stack(losses).detach().cpu().numpy()

print('INITIALIZATION USED TIME {}'.format(time.time()-tk))

# Training phase
iter = 0
tmploss = []
list_rmses = [[np.inf]*opt.N]
list_maes = [[np.inf]*opt.N]
list_mres = [[np.inf]*opt.N]
history = {'rmse':[], 'maes':[], 'mres':[], 'loss':[], 'loss_v':[]}
scales = [2**(i+1) for i in range(opt.N)]
best_ep = 0
for epoch in range(opt.n_epochs):
    train_loss = 0
    ep_time = datetime.now()
    for i, flow_ext in enumerate(train_dataloader): #(flow_from, ..., flow_to, ext)
        flows = flow_ext[:-1]; ext = flow_ext[-1]
        gt_dens, gt_masks = get_gt_densities(flows, opt)
        model.train()
        optimizer.zero_grad()
        
        # generate images with high resolution
        densities, outs = model(flows[0], ext, geo_dict)
        loss_mse, losses = compute_loss(predicts=outs, ys=flows[1:], weights=opt.loss_weights)
        loss_kl, losses_kl = compute_kl_loss(predicts=densities[1:], ys=gt_dens, 
                                             scales=scales, masks=gt_masks)
        loss = (1-opt.alpha)*loss_mse + opt.alpha*loss_kl
        loss.backward()
        optimizer.step()
        tmploss.append(losses)
        print("[Epoch {}/{}] [Batch {}/{}] [Batch Loss: {:.6f}] [KL: {:.6f}, MSE: {:.6f}]".format(epoch,
                                                                    opt.n_epochs,
                                                                    i,
                                                                    len(train_dataloader),
                                                                    loss.item(),
                                                                    loss_kl.item(),
                                                                    loss_mse.item()))
        
        # counting training mse
        train_loss += loss.item() * len(flows[0])
        iter += 1
        # validation phase
        if iter % opt.sample_interval == 0:
            model.eval()
            valid_time = datetime.now()
            total_mses = [0 for i in range(opt.N)]
            total_maes = [0 for i in range(opt.N)]
            total_mres = [0 for i in range(opt.N)]
            tmploss_v = []    
            for j, flow_ext_val in enumerate(valid_dataloader):
                flows_v = flow_ext_val[:-1]; ext_v = flow_ext_val[-1]
                geo_dict_v = {key: geo_data[:len(ext_v)] for key, geo_data in geo_dict.items()}
                densities, outs = model(flows_v[0], ext_v, geo_dict_v)
                loss_v, losses_v = compute_loss(predicts=outs, ys=flows_v[1:], weights=opt.loss_weights)
                tmploss_v.append(losses_v)
                
                preds = [out.cpu().detach().numpy() * opt.scalers[j+1] for j, out in enumerate(outs)]
                labels = [flow.cpu().detach().numpy() * opt.scalers[j] for j, flow in enumerate(flows_v)] 
                for j, (pred, label) in enumerate(zip(preds, labels[1:])):
                    total_mses[j] += get_MSE(pred, label) * len(pred)
                    total_maes[j] += get_MAE(pred, label) * len(pred)
                    total_mres[j] += get_MRE(pred, label) * len(pred)
            rmses = [np.sqrt(total_mse / len(valid_dataloader.dataset)) for total_mse in total_mses]
            maes = [total_mae / len(valid_dataloader.dataset) for total_mae in total_maes]
            mres = [total_mre / len(valid_dataloader.dataset) for total_mre in total_mres]                
            
            if rmses[-1] < np.min(np.stack(list_rmses), axis=0)[-1]:
                str_pattern = '|'.join(['{:.5f}']*opt.N)
                log = ('iter\t{}\tRMSE\t['+str_pattern+']\tMAE\t['+str_pattern+']').format(iter, *rmses, *maes)
                print("{}\ttime\t{}".format(log, datetime.now()-valid_time))
                print('saved to {}'.format(save_path))
                torch.save(model.state_dict(),
                           '{}/final_model.pt'.format(save_path))
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("epoch\t{}\t{}\n".format(epoch, log))
                f.close()
                best_ep = epoch
            
            #update and record
            list_rmses.append(rmses)
            list_maes.append(maes)
            list_mres.append(mres)
            history['loss'].append(np.mean(tmploss, axis=0))
            history['loss_v'].append(np.mean(tmploss_v, axis=0))
            tmploss, tmploss_v = [], []
            history['rmse'] = list_rmses[1:]
            history['mae'] = list_maes[1:]
            history['mre'] = list_mres[1:]
            pickle.dump(history, open(os.path.join(save_path, 'history.pkl'), 'wb'))
            
    # halve the learning rate
    if epoch % opt.harved_epoch == 0 and epoch != 0:
        lr /= 2
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
        f = open('{}/results.txt'.format(save_path), 'a')
        f.write("half the learning rate!\n")
        f.close()
    print('=================time cost: {}==================='.format(
        datetime.now()-ep_time))
    if epoch-best_ep > 50:
        with open('{}/results.txt'.format(save_path), 'a') as f:
            f.write('end at epoch {}'.format(epoch))
        break
    
used_time = (time.time()-tk)/3600
opt.used_time = used_time
save_args(opt, save_path)