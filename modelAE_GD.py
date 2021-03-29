import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable



#cell = 4
#input 256
#output 120 (128-4-4)
#receptive field = 18

#            0  18
#conv 4x4 s1 4  15
#conv 3x3 s2 6  7
#conv 3x3 s1 10 5
#conv 3x3 s1 14 3
#conv 3x3 s1 18 1
#conv 1x1 s1 1  1
class discriminator(nn.Module):
    def __init__(self, d_dim, z_dim):
        super(discriminator, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim

        self.conv_1 = nn.Conv3d(1,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.conv_6 = nn.Conv3d(self.d_dim*16, self.z_dim,    1, stride=1, padding=0, bias=True)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = torch.sigmoid(out)

        return out

#64 -> 256
class generator(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(generator, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        style_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

        self.conv_0 = nn.Conv3d(1+self.z_dim,             self.g_dim,    5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  5, stride=1, dilation=2, padding=4, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  5, stride=1, dilation=2, padding=4, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*4,  5, stride=1, dilation=1, padding=2, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim,    1,             3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_8(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out

#32 -> 128
class generator_halfsize(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(generator_halfsize, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        style_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

        self.conv_0 = nn.Conv3d(1+self.z_dim,             self.g_dim,    3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=1, padding=1, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim,    1,             3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_8(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out

#32 -> 256
class generator_halfsize_x8(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(generator_halfsize_x8, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        style_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

        self.conv_0 = nn.Conv3d(1+self.z_dim,             self.g_dim,    3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*8,  self.g_dim*4, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*4,  self.g_dim*4,  3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_9 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim,   1,             3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _,_,dimx,dimy,dimz = out.size()
        zs = z.repeat(1,1,dimx,dimy,dimz)
        out = torch.cat([out,zs],axis=1)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_10(out)
        #out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        #out = out.clamp(max=1.0)
        out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        #out = torch.sigmoid(out)

        out = out*mask

        return out
