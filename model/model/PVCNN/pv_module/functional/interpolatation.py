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

from torch.autograd import Function

from .backend import _backend

__all__ = ['nearest_neighbor_interpolate']


class NeighborInterpolation(Function):
    @staticmethod
    def forward(ctx, points_coords, centers_coords, centers_features):
        """
        :param ctx:
        :param points_coords: coordinates of points, FloatTensor[B, 3, N]
        :param centers_coords: coordinates of centers, FloatTensor[B, 3, M]
        :param centers_features: features of centers, FloatTensor[B, C, M]
        :return:
            points_features: features of points, FloatTensor[B, C, N]
        """
        centers_coords = centers_coords.contiguous()
        points_coords = points_coords.contiguous()
        centers_features = centers_features.contiguous()
        points_features, indices, weights = _backend.three_nearest_neighbors_interpolate_forward(
            points_coords, centers_coords, centers_features
        )
        ctx.save_for_backward(indices, weights)
        ctx.num_centers = centers_coords.size(-1)
        return points_features

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights = ctx.saved_tensors
        grad_centers_features = _backend.three_nearest_neighbors_interpolate_backward(
            grad_output.contiguous(), indices, weights, ctx.num_centers
        )
        return None, None, grad_centers_features


nearest_neighbor_interpolate = NeighborInterpolation.apply
