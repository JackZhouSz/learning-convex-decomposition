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

__all__ = ['grouping']


class Grouping(Function):
    @staticmethod
    def forward(ctx, features, indices):
        """
        :param ctx:
        :param features: features of points, FloatTensor[B, C, N]
        :param indices: neighbor indices of centers, IntTensor[B, M, U], M is #centers, U is #neighbors
        :return:
            grouped_features: grouped features, FloatTensor[B, C, M, U]
        """
        features = features.contiguous().float()
        indices = indices.contiguous()
        ctx.save_for_backward(indices)
        ctx.num_points = features.size(-1)
        return _backend.grouping_forward(features, indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_features = _backend.grouping_backward(grad_output.contiguous(), indices, ctx.num_points)
        return grad_features, None


grouping = Grouping.apply
