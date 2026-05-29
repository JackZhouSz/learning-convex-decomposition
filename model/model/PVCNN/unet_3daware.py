# https://github.com/mit-han-lab/pvcnn
'''
MIT License

Copyright (c) 2019 MIT HAN Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


import einops

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

class ConvTriplane3dAware(nn.Module):
    """ 3D aware triplane conv (as described in RODIN) """
    # TODO unit test this (can check dims since can run on non-cube)
    def __init__(self, internal_conv_f, in_channels, out_channels, order='xz'):
        """
        Args:
            internal_conv_f: function that should return a 2D convolution Module 
                given in and out channels
            order: if triplane input is in 'xz' order
        """
        super(ConvTriplane3dAware, self).__init__()
        # Need 3 seperate convolutions
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert order in ['xz', 'zx']
        self.order = order
        # Going to stack from other planes
        self.plane_convs =  nn.ModuleList([
            internal_conv_f(3*self.in_channels, self.out_channels) for _ in range(3)])
    
    def forward(self, triplanes_list):
        """
        Args:
            triplanes_list: [(B,Ci,H,W)]*3 in xy,yz,(zx or xz) depending on order
        Returns:
            out_triplanes_list: [(B,Co,H,W)]*3 in xy,yz,(zx or xz) depending on order
        """
        inps = list(triplanes_list)
        xp = 1 #(yz)
        yp = 2 #(zx)
        zp = 0 #(xy)

        if self.order == 'xz':
            # get into zx order
            inps[yp] = einops.rearrange(inps[yp], 'b c x z -> b c z x')


        oplanes = [None]*3 
        # order shouldn't matter
        for iplane in [zp, xp, yp]:
            # i_plane -> (j,k)

            # need to average out i and convert to (j,k)
            # j_plane -> (k,i)
            # k_plane -> (i,j)
            jplane = (iplane+1)%3
            kplane = (iplane+2)%3

            ifeat = inps[iplane]
            # need to average out nonshared dim
            # Average pool across

            # j_plane -> (k,i) -> (k,1) -> (1,k) -> (j,k)
            # b c k i -> b c k 1
            jpool = torch.mean(inps[jplane], dim=3 ,keepdim=True)
            jpool = einops.rearrange(jpool, 'b c k 1 -> b c 1 k')
            jpool = einops.repeat(jpool, 'b c 1 k -> b c j k', j=ifeat.size(2))

            # k_plane -> (i,j) -> (1,j) -> (j,1) -> (j,k)
            # b c i j -> b c 1 j
            kpool = torch.mean(inps[kplane], dim=2 ,keepdim=True)
            kpool = einops.rearrange(kpool, 'b c 1 j -> b c j 1')
            kpool = einops.repeat(kpool, 'b c j 1 -> b c j k', k=ifeat.size(3))

            # b c h w
            # jpool = jpool.expand_as(ifeat)
            # kpool = kpool.expand_as(ifeat)

            # concat and conv on feature dim
            catfeat = torch.cat([ifeat, jpool, kpool], dim=1)
            oplane = self.plane_convs[iplane](catfeat)
            oplanes[iplane] = oplane

        if self.order == 'xz':
            # get back into xz order
            oplanes[yp] = einops.rearrange(oplanes[yp], 'b c z x -> b c x z')

        return oplanes

def roll_triplanes(triplanes_list):
    # B, C, tri, h, w
    tristack = torch.stack((triplanes_list),dim=2)
    return einops.rearrange(tristack, 'b c tri h w -> b c (tri h) w', tri=3)

def unroll_triplanes(rolled_triplane):
    # B, C, tri*h, w
    tristack = einops.rearrange(rolled_triplane, 'b c (tri h) w -> b c tri h w', tri=3)
    return torch.unbind(tristack, dim=2)

# Replace internal convs
# def conv3x3triplane3daware(in_channels, out_channels, order='xz', **kwargs):    
#     return ConvTriplane3dAware(lambda inp, out: conv3x3(inp,out,**kwargs), 
#                                in_channels, out_channels,order=order)

# def upconv2x2triplane3daware(in_channels, out_channels, **kwargs):    
#     return ConvTriplane3dAware(lambda inp, out: upconv2x2(inp,out,**kwargs))

def conv1x1triplane3daware(in_channels, out_channels, order='xz', **kwargs):    
    return ConvTriplane3dAware(lambda inp, out: conv1x1(inp,out,**kwargs), 
                               in_channels, out_channels,order=order)

def Normalize(in_channels, num_groups=32):
    num_groups = min(in_channels, num_groups)  # avoid error if in_channels < 32
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # return F.relu(x)
    # Swish
    return x*torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock3dAware(nn.Module):
    def __init__(self, in_channels, out_channels=None):
            #, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        # self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv3x3(self.in_channels, self.out_channels)

        self.norm_mid = Normalize(out_channels)
        self.conv_3daware = conv1x1triplane3daware(self.out_channels, self.out_channels)

        self.norm2 = Normalize(out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.in_channels != self.out_channels:
            # if self.use_conv_shortcut:
            #     self.conv_shortcut = torch.nn.Conv2d(in_channels,
            #                                          out_channels,
            #                                          kernel_size=3,
            #                                          stride=1,
            #                                          padding=1)
            # else:
            self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)

    def forward(self, x):
        # 3x3 plane comm
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # 1x1 3d aware, crossplane comm
        h = self.norm_mid(h)
        h = nonlinearity(h)
        h = unroll_triplanes(h)
        h = self.conv_3daware(h)
        h = roll_triplanes(h)

        # 3x3 plane comm
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            # if self.use_conv_shortcut:
            #     x = self.conv_shortcut(x)
            # else:
            x = self.nin_shortcut(x)

        return x+h

class DownConv3dAware(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, downsample=True, with_conv=False):
        super(DownConv3dAware, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.pooling = pooling

        # roll
        # self.norm1 = Normalize(in_channels)
        # self.conv1 = conv3x3(self.in_channels, self.out_channels)
        # # followed by 3daware
        # self.norm2 = Normalize(out_channels)
        # self.conv2 = conv3x3triplane3daware(self.out_channels, self.out_channels)
        self.block = ResnetBlock3dAware(in_channels=in_channels, 
            out_channels=out_channels)

        self.do_downsample = downsample
        self.downsample = Downsample(out_channels, with_conv=with_conv)

        # if self.pooling:
        #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # if self.in_channels != self.out_channels:
        #     if self.use_conv_shortcut:
        #         self.conv_shortcut = torch.nn.Conv2d(in_channels,
        #                                              out_channels,
        #                                              kernel_size=3,
        #                                              stride=1,
        #                                              padding=1)
        #     else:
        #         self.nin_shortcut = torch.nn.Conv2d(in_channels,
        #                                             out_channels,
        #                                             kernel_size=1,
        #                                             stride=1,
        #                                             padding=0)

    def forward(self, x):
        """
        rolled input, rolled output
        Args:
            x: rolled (b c (tri*h) w)
        """
        x = self.block(x)
        before_pool = x
        # if self.pooling:
        #     x = self.pool(x)
        if self.do_downsample:
            # unroll and cat channel-wise (to prevent pooling across triplane boundaries)
            x = einops.rearrange(x, 'b c (tri h) w -> b (c tri) h w', tri=3)
            x = self.downsample(x)
            # undo
            x = einops.rearrange(x, 'b (c tri) h w -> b c (tri h) w', tri=3)
        return x, before_pool

class UpConv3dAware(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', with_conv=False): #up_mode='transpose', ):
        super(UpConv3dAware, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        # self.up_mode = up_mode

        # self.upconv = upconv2x2(self.in_channels, self.out_channels, 
        #     mode=self.up_mode)
            
        self.upsample = Upsample(in_channels, with_conv)

        if self.merge_mode == 'concat':
            self.norm1 = Normalize(in_channels+out_channels)
            self.block = ResnetBlock3dAware(in_channels=in_channels+out_channels, 
                out_channels=out_channels)
        else:
            self.norm1 = Normalize(in_channels)
            self.block = ResnetBlock3dAware(in_channels=in_channels, 
                out_channels=out_channels)
            # self.block = ResnetBlock3dAware(in_channels=out_channels, 
            #     out_channels=out_channels)
            # num of input channels to conv2 is same
        

    def forward(self, from_down, from_up):
        """ Forward pass
        rolled inputs, rolled output
        rolled (b c (tri*h) w)
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        # from_up = self.upconv(from_up)
        from_up = self.upsample(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down

        x = self.norm1(x)
        x = self.block(x)
        return x

class UNetTriplane3dAware(nn.Module):
    def __init__(self, out_channels, in_channels=3, depth=5, 
                 start_filts=64,# up_mode='transpose', 
                 use_initial_conv=False,
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
        """
        super(UNetTriplane3dAware, self).__init__()
    
        # if merge_mode in ('concat', 'add'):
        #     self.merge_mode = merge_mode
        # else:
        #     raise ValueError("\"{}\" is not a valid mode for"
        #                      "merging up and down paths. "
        #                      "Only \"concat\" and "
        #                      "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        # if self.up_mode == 'upsample' and self.merge_mode == 'add':
        #     raise ValueError("up_mode \"upsample\" is incompatible "
        #                      "with merge_mode \"add\" at the moment "
        #                      "because it doesn't make sense to use "
        #                      "nearest neighbour to reduce "
        #                      "depth channels (by half).")

        self.out_channels = out_channels 
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        # TODO make initial conv a configurable?
        self.use_initial_conv = use_initial_conv
        if use_initial_conv:
            self.conv_initial = conv1x1(self.in_channels, self.start_filts)

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            if i == 0:
                ins = self.start_filts if use_initial_conv else self.in_channels
            else:
                ins = outs
            outs = self.start_filts*(2**i)
            downsamp_it = True if i < depth-1 else False

            down_conv = DownConv3dAware(ins, outs, downsample = downsamp_it)
            self.down_convs.append(down_conv)

        # Actually mid block is already built into above 
        # self.block_mid = ResnetBlock3dAware(outs, outs)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv3dAware(ins, outs,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.norm_out = Normalize(outs)
        self.conv_final = conv1x1(outs, self.out_channels)

        # self.bias = torch.nn.Parameter(torch.zeros([out_channels], device=device))
        # imitating to rgb??
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            # init.xavier_normal_(m.weight, gain=0.1)
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        """
        Args:
            x: Stacked triplane expected to be in (B,3,C,H,W)
        """
        # Roll
        x = einops.rearrange(x, 'b tri c h w -> b c (tri h) w', tri=3)

        if self.use_initial_conv:
            x = self.conv_initial(x)

        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        
        # Spend a block in the middle
        # x = self.block_mid(x)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        x = self.norm_out(x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(nonlinearity(x))

        # Unroll
        x = einops.rearrange(x, 'b c (tri h) w -> b tri c h w', tri=3)
        return x

    
def setup_unet(output_channels, input_channels, unet_cfg):
    if unet_cfg['use_3d_aware']:
        assert(unet_cfg['rolled'])
        unet = UNetTriplane3dAware(
                                        out_channels=output_channels, 
                                        in_channels=input_channels, 
                                        depth=unet_cfg['depth'], 
                                        use_initial_conv=unet_cfg['use_initial_conv'],
                                        start_filts=unet_cfg['start_hidden_channels'],)
    else:
        raise NotImplementedError
    # else:
    #     unet = UNetTriplane(rolled=unet_cfg['rolled'],
    #                                     out_channels=output_channels, 
    #                                     in_channels=input_channels, 
    #                                     depth=unet_cfg['depth'], 
    #                                     start_filts=unet_cfg['start_hidden_channels'],)
    return unet


if __name__ == "__main__":
    # Do extreme unit test for ConvTriplane3dAware by feeding different shapes
    # xy, yz, zx

    B = 1
    Cin = 4
    # sx = 10
    # sy = 12
    # sz = 14

    sx = 128
    sy = 128
    sz = 128

    #TODO update with order

    triplane_list = [
        torch.zeros(((B, Cin, sx, sy))),
        torch.zeros(((B, Cin, sy, sz))),
        torch.ones(((B, Cin, sz, sx))),]

    ctrip = conv3x3triplane3daware(Cin, Cin)

    out_list = ctrip(triplane_list)
    print([out_.size() for out_ in out_list])

    unet = UNetTriplane3dAware(Cin, Cin,depth=2, start_filts=16)

    # from torchview import draw_graph
    # draw_graph(unet, [torch.stack(triplane_list,dim=1)], device='cpu', save_graph=True, expand_nested=True, depth=1,
    #     hide_inner_tensors=True)

