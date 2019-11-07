import torch.nn as nn
import torch.nn.functional as F
import torch
from .layers import LocalConv
import math

def n2_normalization_func(x, scale_factor):
    out = F.avg_pool2d(x, scale_factor) * scale_factor ** 2
    out = F.upsample(out, scale_factor=scale_factor)
    return torch.div(x, out + 1e-5)

def recover_func(dens, lr_inp, scale_factor):
    assert lr_inp.size()[-1] * scale_factor == dens.size()[-1], \
        'lr shape {}, dens shape {}, scale {}'.format(lr_inp.size(), dens.size(), scale_factor)
    hr_inp = F.upsample(lr_inp, scale_factor=scale_factor)
    return torch.mul(hr_inp, dens)

def process_featmaps(feat_list):
    len_feat = len(feat_list)
    if len_feat == 1:
        feat_maps = feat_list[0]
    else:
        feat_maps = feat_list[-1]
        for i in range(len_feat-2, -1, -1):
            scale = 2 ** (len_feat-i-1)
            feat_prev = F.upsample(feat_list[i], scale_factor=scale)
            feat_maps = feat_prev + feat_maps
        feat_maps = feat_maps/len_feat
    return feat_maps

def embed_ext(self, ext):
    ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
    ext_out2 = self.embed_hour(
        ext[:, 5].long().view(-1, 1)).view(-1, 3)
    ext_out3 = self.embed_weather(
        ext[:, 6].long().view(-1, 1)).view(-1, 3)
    ext_out4 = ext[:, :4]
    return torch.cat([ext_out1, ext_out2, ext_out3, ext_out4], dim=1)
    
class ProposalNetGlobal(nn.Module):
    def __init__(self, scale_factor, in_channels, compress, w=None, h=None, n_res=1, islocal=False):
        super(ProposalNetGlobal, self).__init__()
        self.compress = compress 
        self.scale = scale_factor
        
        base_channels = 1 + in_channels 
        res_blocks = [ResidualBlock(base_channels) for _ in range(n_res)]
        self.res_blocks = nn.Sequential(*res_blocks)    

        if islocal:
            compress_chn = compress
            self.conv_upsample = nn.Sequential(
                nn.Conv2d(base_channels, compress_chn * 4, 3, 1, 1),
                nn.BatchNorm2d(compress_chn * 4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                LocalConv(width=w, block_size=self.scale, in_chn=compress_chn, out_chn=1),
                nn.ReLU(inplace=True),)
        else:
            self.conv_upsample = nn.Sequential(
                nn.Conv2d(base_channels, 1 * 4, 3, 1, 1),
                nn.BatchNorm2d(1 * 4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True))

    def forward(self, prev_dens, feat=None):
        """
        prev_dens: (None, 1, N/2, N/2)
        feat: (None, F, N/2, N/2)
        """
        assert not feat is None, 'feature should be provided'
        feat_new = torch.cat([prev_dens, feat], dim=1)
        feat_new = self.res_blocks(feat_new)
        density_propo_ = self.conv_upsample(feat_new)
        density_propo = n2_normalization_func(density_propo_, self.scale)
        return density_propo

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class UrbanPy(nn.Module): #use proposalnet   
    def __init__(self, scales, N, in_channels=1, out_channels=1, n_residual_blocks=(16, 16), base_channels=64, ext_dim=7, img_width=32, img_height=32, ext_flag=True, n_res=1, islocal=False, compress=2):
        super(UrbanPy, self).__init__()
        self.ext_flag = ext_flag
        self.img_width = img_width
        self.img_height = img_height
        self.scales = scales
        self.N = N
        self.n_res = n_res
        self.ext_flag = ext_flag
        self.islocal=islocal

        if ext_flag:
            # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_day = nn.Embedding(8, 2)
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
            self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

            self.ext2lr = nn.Sequential(
                nn.Linear(12, 128),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.Linear(128, img_width * img_height),
                nn.ReLU(inplace=True)
            )

            self.ext2hr = nn.Sequential(
                nn.Conv2d(1, 4, 3, 1, 1),
                nn.BatchNorm2d(4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True)
            )
        if ext_flag:
            conv_inp_in = in_channels + 1
            conv_out_in = base_channels + 1
            propo_channels = base_channels + 1
            residual_channels = base_channels + 1
        else:
            conv_inp_in = in_channels
            conv_out_in = base_channels
            propo_channels = base_channels
            residual_channels = base_channels

        self.res_blocks = nn.ModuleList()
        self.post_res_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        self.correction_blocks = nn.ModuleList()
        self.proposal_blocks = nn.ModuleList()
        self.bottlenecks  = nn.ModuleList()
        
        self.conv_inp = nn.Sequential(
            nn.Conv2d(conv_inp_in, base_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )
    
        for i in range(N):

            #res blocks
            res_blocks = nn.ModuleList([])
            for _ in range(n_residual_blocks[i]):
                res_blocks.append(ResidualBlock(base_channels))
            self.res_blocks.append( nn.Sequential(*res_blocks) )

            #upsampling blocks
            self.upsampling_blocks.append(nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True)))

            geo_dim = 5 # 0 if geo_feature is not available
            block_size = 2**(i+1)
            
            self.correction_blocks.append(nn.Sequential(
                nn.Conv2d(residual_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
                
            self.proposal_blocks.append(
                  ProposalNetGlobal(2**(i+1), propo_channels+geo_dim, compress, img_width*2**(i+1), img_height*2**(i+1), n_res=n_res, islocal=islocal))
            
    def forward(self, x, ext, geo_dict):
        geo_keys = list(geo_dict.keys())
        if torch.cuda.is_available():
            densities = [torch.ones(x.shape).cuda()]
        else:
            densities = [torch.ones(x.shape)]
            
        inp = x
        if self.ext_flag:
            ext_emb = embed_ext(self, ext)
            ext_out = self.ext2lr(ext_emb).view(-1, 1, self.img_width, self.img_height)    
            inp = torch.cat([x, ext_out], dim=1)
        
        outs = []
        feat_list = []      
        
        inp = self.conv_inp(inp)
        for i in range(self.N):
            current_scale = 2**(i+1)
            feat_maps = self.res_blocks[i](inp) #feature extraction
            feat_maps_up = self.upsampling_blocks[i](feat_maps) #upsample using subpixel
            feat_list.append(feat_maps)
            if self.ext_flag:
                ext_out_up = self.ext2hr(ext_out)
                feat = torch.cat([feat_maps_up, ext_out_up], dim=1)
                density_res_ = self.correction_blocks[i](feat) #convert feature to residual
            else:
                density_res_ = self.correction_blocks[i](feat_maps_up)

            #mode == global
            density_res = n2_normalization_func(density_res_, current_scale)
            feat_maps = process_featmaps(feat_list)
            
            if self.ext_flag:
                feat_maps = torch.cat([feat_maps, ext_out, geo_dict[geo_keys[i]]], dim=1)
            density_propo = self.proposal_blocks[i](densities[-1], feat_maps)
            density_ = density_propo + density_res
            density_glb = n2_normalization_func(density_, current_scale) #renormalize

            densities.append(density_glb)
            outs.append(recover_func(density_glb, x*self.scales[0]/self.scales[i+1], scale_factor=current_scale))
            inp = feat_maps_up
            if self.ext_flag: ext_out = ext_out_up
        return densities, outs