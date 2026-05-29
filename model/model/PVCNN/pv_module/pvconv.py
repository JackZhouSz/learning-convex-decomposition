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

import torch.nn as nn

from . import functional as F
from .voxelization import Voxelization
from .shared_mlp import SharedMLP
from .se import SE3d
import torch

__all__ = ['PVConv']


# def my_voxelization(features, coords, resolution):
#     import ipdb
#     ipdb.set_trace()
#     features = features.contiguous().float()
#     coords = coords.int().contiguous()
#     b, c, _ = features.shape
#     result = torch.zeros(b, c, resolution * resolution * resolution, device=features.device, dtype=torch.float)
#     r = resolution
#     r2 = resolution * resolution
#     indices = coords[:, 0] * r2 + coords[:, 1] * r + coords[:, 2]
#     indices = indices.unsqueeze(dim=1)
#     result.scatter_(index=indices.long(), src=features, dim=2, reduce='add')
#     torch.scatter_reduce(input=result, index=indices, value=features, dim=2, reduce='sum')
#     return result.view(b, c, resolution, resolution, resolution)


class PVConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0, scale_pvcnn=False,
            device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        # print('==> PVConv device: ', device)
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps, scale_pvcnn=scale_pvcnn)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, device=device),
            nn.InstanceNorm3d(out_channels, eps=1e-4, device=device),
            # nn.GroupNorm(8, out_channels, device=device),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, device=device),
            nn.InstanceNorm3d(out_channels, eps=1e-4, device=device),
            # nn.GroupNorm(8, out_channels, device=device),
            nn.LeakyReLU(0.1, True),
        ]
        #####################################
        # if with_se:
        #     voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels, device=device)

    def forward(self, inputs):
        features, coords = inputs
        # import ipdb
        # ipdb.set_trace()
        voxel_features, voxel_coords = self.voxelization(features, coords)
        # voxel_features, voxel_coords = my_voxelization(features, coords, self.resolution)
        ###################
        # import ipdb
        # ipdb.set_trace()
        # print('Voxel feature in: ', voxel_features.device)
        voxel_features = self.voxel_layers(voxel_features)
        devoxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = devoxel_features + self.point_features(features)
        return fused_features, coords, voxel_features
