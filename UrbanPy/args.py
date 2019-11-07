import argparse
import numpy as np

def get_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='adam: decay of second order momentum of gradient')
    parser.add_argument('--n_residuals', type=str, default='16,16',
                        help='number of residual units')
    parser.add_argument('--base_channels', type=int,
                        default=64, help='number of feature maps')
    parser.add_argument('--channels', type=int, default=1,
                        help='number of flow image channels')
    parser.add_argument('--sample_interval', type=int, default=20,
                        help='interval between validation')
    parser.add_argument('--harved_epoch', type=int, default=50,
                        help='halved at every x epoch')
    parser.add_argument('--seed', type=int, default=2017, help='random seed')
    parser.add_argument('--ext_dim', type=int, default=7,
                        help='external factor dimension')
    parser.add_argument('--ext_flag', action='store_true',
                        help='whether to use external factors')
    parser.add_argument('--dataset', type=str, default='TaxiBJ/P1',
                        help='which dataset to use')
    ###############################################################
    parser.add_argument('--resume', default=False, action='store_true',
                        help='continue training')
    parser.add_argument('--start_from', type=int,
                        help='return training at this epoch')
    parser.add_argument('--simple', default=False, action='store_true',
                        help='use simple proposal')
    parser.add_argument('--name', type=str, default='',
                        help='training annotation')
    parser.add_argument('--n_res_propnet', type=int, default=1,
                        help='number of res_layer in proposal net')
    parser.add_argument('--from_reso', type=int, choices=[8, 16, 32, 64, 128],
                        help='coarse-grained input resolution')
    parser.add_argument('--to_reso', type=int, choices=[16, 32, 64, 128],
                        help='fine-grained input resolution')
    parser.add_argument('--islocal', action='store_true',
                        help='whether to use external factors')
    parser.add_argument('--loss_weights', type=str, default='1,1,1', 
                        help='weights on multiscale loss')
    parser.add_argument('--alpha', type=float, default=1e-2,
                        help='alpha')
    parser.add_argument('--debug', action='store_true',
                        help='whether to use external factors')
    parser.add_argument('--compress', type=int, default=2,
                        help='compress channels')
    parser.add_argument('--coef', type=float, default=0.5,
                        help='the weight of proposal net')
    parser.add_argument('--save_diff', default=False, action='store_true',
                        help='save differences')
    opt = parser.parse_args(args)
    
    opt.scaler_dict = {8: 8000, 16:3000, 32:1500, 64:400, 128:100}
    if opt.resume:
        assert opt.start_from > 0 , 'start_from is {}'.format(opt.start_from)
    opt.n_residuals = [int(_) for _ in opt.n_residuals.split(',')]
    opt.loss_weights = [float(_) for _ in opt.loss_weights.split(',')]
    opt.N = int(np.log(opt.to_reso / opt.from_reso)/np.log(2))

#     assert opt.N == len(opt.n_residuals),\
#         'upsampling step {}, len(n_residuals) {}'.format(opt.N, len(opt.n_residuals))
    opt.n_residuals = opt.n_residuals[:opt.N]
    assert opt.from_reso < opt.to_reso, 'invalid resolution, from {} to {}'.format(opt.from_reso, opt.to_reso)
    opt.scales = [opt.from_reso*2**i for i in range(opt.N+1)]
    opt.scalers= [opt.scaler_dict[key] for key in opt.scales]
    
    return opt