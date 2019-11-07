import torch.nn as nn
import torch
import torch.nn.functional as F
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
import numpy as np

class LocalConv(nn.Module):
    def __init__(self, width, block_size, in_chn, out_chn):
        super(LocalConv, self).__init__()
        self.width = width
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.bs = block_size
        self.n_blk = width // self.bs
        self.weight = nn.Parameter(torch.randn(self.n_blk*self.n_blk, self.bs**2*in_chn, self.bs**2*out_chn)) #32*32 x 4*4*2 x 4*4*1
        
    def forward(self, x):
        # x is of shape NCWH
        N, C, W, H = x.shape
        assert C*W*H == self.in_chn*self.width**2, \
            'shape error x: {} vs defined {}'.format(x.shape, (self. in_chn, self.width))
        x = x.view(N, C, self.n_blk, self.bs, self.n_blk, self.bs).permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, 1, C*self.bs*self.bs, self.n_blk*self.n_blk).permute(0, 3, 1, 2).contiguous()
        tmp = [x_.bmm(self.weight) for x_ in x]
        tmp = torch.stack(tmp) # N, n_blk*n_blk, 1, bs*bs*out_C
        tmp = tmp.view(N, self.n_blk, self.n_blk, self.bs, self.bs, self.out_chn).permute(0, 5, 1, 3, 2, 4).contiguous()
        tmp = tmp.view(N, self.out_chn, self.bs*self.n_blk, self.bs*self.n_blk)
        return tmp
    
def test_local_conv():
    N, C, W, H, out_chn = 2,3,8,8,3
    n_blk, bs = 4, 2
    weight = torch.eye(C*bs*bs).expand(n_blk*n_blk, -1, -1)
    a = torch.randn(2,3,8,8)
    b = a.view(N, C, n_blk, bs, n_blk, bs).permute(0, 3, 5, 1, 2, 4).contiguous()
    b = b.view(N, 1, C*bs*bs, n_blk*n_blk).permute(0, 3, 1, 2).contiguous()
    tmp = [b_.bmm(weight) for b_ in b]
    b = torch.stack(tmp)
    b = b.view(N, n_blk, n_blk, bs, bs, out_chn).permute(0, 5, 1, 3, 2, 4).contiguous()
    b = b.view(N, out_chn, bs*n_blk, bs*n_blk)
    print(a.eq(b).sum() == 2*3*8*8)
    
#############################################################
def batch_kl(yhat, y, scale, mask):
    """
    yhat, y of shape [bs, 1, w, h]
    """
    assert len(y.shape) == 4
    assert yhat.shape == y.shape, 'yhat shape {}, y shape {}'.format(yhat.shape, y.shape)
    yhat = torch.log(yhat+1e-9)
    y    = y+1e-9
    def reshape(x, scale):
        bs, _, w, h = x.shape
        x = x.squeeze(1).reshape(bs, w//scale, scale, h//scale, scale) #[bs, w/s, s, h/s, s]
        x = x.permute(0,1,3,2,4) #[bs, w/s, h/s, s, s]
        x = x.reshape(bs, -1, scale**2)
        return x
    
    yhat_ = reshape(yhat, scale)
    y_    = reshape(y, scale)
    mask_ = reshape(mask, scale)

    loss = F.kl_div(yhat_, y_, reduction='none') #[bs, (w/s)*(h/s), s**2]
    loss = torch.where(mask_>0, loss, Tensor(np.zeros(1))) # [bs, n_blk, blk_size], sum on blk_size, mean on bs and n_blk
    loss = loss.sum([1,2])
    mask_ = mask_.sum([1,2])/(scale**2)
    loss /= mask_.float()
    return loss.mean()

def reverse_density(hr, lr, upscale, hr_scaler, lr_scaler):
    hr_o = F.upsample(lr*lr_scaler, scale_factor=upscale)
    dis = (hr*hr_scaler/hr_o).numpy()
    return dis, 1-np.isnan(dis)