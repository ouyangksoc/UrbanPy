import os
import h5py
import numpy as np
import argparse
import sys
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from .utils.metrics import get_RMSE, get_MAE, get_MSE, get_MRE
from .utils.data_process import get_lapprob_dataloader, print_model_parm_nums
from .args import get_args
from .model import UrbanPy, weights_init_normal

warnings.filterwarnings("ignore")
opt = get_args(sys.argv[1:])
print(opt)
res_str = ','.join([str(i) for i in opt.n_residuals])
# islocal = '' if opt.islocal else '_nonlocal'
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
print('LOADING FROM {}'.format(save_path))
# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

geo_dict = dict()
with h5py.File('data/embedded_geos.h5', 'r') as hf:
    for key in opt.scales:
        geo_dict[key] = hf.get(str(key)).value
geo_dict = {key:Tensor(geo_data).expand(opt.batch_size,-1,-1,-1) for key, geo_data in geo_dict.items()}


# load model
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

model.load_state_dict(torch.load('{}/final_model.pt'.format(save_path)))
model.eval()
if cuda:
    model.cuda()

# load training set and validation set
datapath = os.path.join('data', opt.dataset)
test_dataloader = get_lapprob_dataloader(
    datapath, opt, batch_size=16, mode='test')

total_mses = [0 for i in range(opt.N)]
total_maes = [0 for i in range(opt.N)]
total_mres = [0 for i in range(opt.N)]
for i, flow_ext in enumerate(test_dataloader):
    flows = flow_ext[:-1]; ext = flow_ext[-1]
    geo_dict = {key: geo_data[:len(ext)] for key, geo_data in geo_dict.items()}
    densities, outs = model(flows[0], ext, geo_dict)
    
    preds = [out.cpu().detach().numpy() * opt.scalers[j+1] for j, out in enumerate(outs)]
    test_labels = [flow.cpu().detach().numpy() * opt.scalers[j] for j, flow in enumerate(flows)] 

    for j, (pred, label) in enumerate(zip(preds, test_labels[1:])):
        total_mses[j] += get_MSE(pred, label) * len(pred)
        total_maes[j] += get_MAE(pred, label) * len(pred)
        total_mres[j] += get_MRE(pred, label) * len(pred)
rmses = [np.sqrt(total_mse / len(test_dataloader.dataset)) for total_mse in total_mses]
maes = [total_mae / len(test_dataloader.dataset) for total_mae in total_maes]
mres = [total_mre / len(test_dataloader.dataset) for total_mre in total_mres]

with open('{}/test_results.txt'.format(save_path), 'w') as f:
    f.write("RMSE\n{}\tMAE\n{}\nMAPE{}\n".format(rmses, maes, mres))

print('Test RMSE = {}\nMAE = {}\nMRE = {}'.format(rmses, maes, mres))
