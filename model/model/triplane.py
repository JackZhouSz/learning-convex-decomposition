#https://github.com/3DTopia/OpenLRM/blob/main/openlrm/models/modeling_lrm.py
'''
Under Apache 2.0
License file found here: https://github.com/3DTopia/OpenLRM/blob/main/LICENSE

                               Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from functools import partial

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    plane_axes = torch.tensor([[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]],
                                [[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]],
                                [[0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 0]]], dtype=torch.float32).cuda()
    
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    #coordinates = (2/box_warp) * coordinates # add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def get_grid_coord(grid_size = 256, align_corners=False):
    if align_corners == False:
        coords = torch.linspace(-1 + 1/(grid_size), 1 - 1/(grid_size), steps=grid_size)
    else:
        coords = torch.linspace(-1, 1, steps=grid_size)
    i, j, k = torch.meshgrid(coords, coords, coords, indexing='ij')
    coordinates = torch.stack((i, j, k), dim=-1).reshape(-1, 3)
    return coordinates

class BasicBlock(nn.Module):
    """
    Transformer block that is in its simplest form.
    Designed for PF-LRM architecture.
    """
    # Block contains a self-attention layer and an MLP
    def __init__(self, inner_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )
        # self.attn_bias = attn_bias
        # self.num_heads = num_heads

    def forward(self, x):
        # x: [N, L, D]
        before_sa = self.norm1(x)
        # print()
        # print(self.attn_bias)
        # print(self.num_heads)
        x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x
        
class ConditionBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """
    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = x + self.cross_attn(self.norm1(x), cond, cond, need_weights=False)[0]
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, block_type: str,
                 num_layers: int, num_heads: int,
                 inner_dim: int, cond_dim: int = None,
                 eps: float = 1e-6):
        super().__init__()
        self.block_type = block_type
        self.layers = nn.ModuleList([
            self._block_fn(inner_dim, cond_dim)(
                num_heads=num_heads,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)

    @property
    def block_type(self):
        return self._block_type

    @block_type.setter
    def block_type(self, block_type):
        assert block_type in ['cond', 'basic'], \
            f"Unsupported block type: {block_type}"
        self._block_type = block_type

    def _block_fn(self, inner_dim, cond_dim):
        assert inner_dim is not None, f"inner_dim must always be specified"
        if self.block_type == 'basic':
            # assert cond_dim is None and mod_dim is None, \
            #     f"Condition and modulation are not supported for BasicBlock"
            # # from .block import BasicBlock
            # # logger.debug(f"Using BasicBlock")
            return partial(BasicBlock, inner_dim=inner_dim)
        elif self.block_type == 'cond':
            assert cond_dim is not None, f"Condition dimension must be specified for ConditionBlock"
            return partial(ConditionBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        else:
            raise ValueError(f"Unsupported block type during runtime: {self.block_type}")


    def forward_layer(self, layer: nn.Module, x: torch.Tensor, cond: torch.Tensor,):
        if self.block_type == 'basic':
            return layer(x)
        elif self.block_type == 'cond':
            return layer(x, cond)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond] or None
        for layer in self.layers:
            x = self.forward_layer(layer, x, cond)
        x = self.norm(x)
        return x

class Voxel2Triplane(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self, transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 triplane_low_res: int, triplane_high_res: int, triplane_dim: int, voxel_feat_dim: int, normalize_vox_feat=False, voxel_dim=16):
        super().__init__()
        
        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim
        self.voxel_feat_dim = voxel_feat_dim

        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, transformer_dim) * (1. / transformer_dim) ** 0.5)
        self.transformer = TransformerDecoder(
            block_type='cond',
            num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=transformer_dim, cond_dim=voxel_feat_dim
        )
        self.upsampler = nn.ConvTranspose2d(transformer_dim, triplane_dim, kernel_size=8, stride=8, padding=0)

        self.normalize_vox_feat = normalize_vox_feat
        if normalize_vox_feat:
            self.vox_norm = nn.LayerNorm(voxel_feat_dim, eps=1e-6)
            self.vox_pos_embed = nn.Parameter(torch.randn(1, voxel_dim * voxel_dim * voxel_dim, voxel_feat_dim) * (1. / voxel_feat_dim) ** 0.5)

    def forward_transformer(self, voxel_feats):
        N = voxel_feats.shape[0]
        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        if self.normalize_vox_feat:
            vox_pos_embed = self.vox_pos_embed.repeat(N, 1, 1)  # [N, L, D]
            #print(vox_pos_embed.shape, voxel_feats.shape)
            voxel_feats = self.vox_norm(voxel_feats + vox_pos_embed)
        x = self.transformer(
            x,
            cond=voxel_feats
        )
        return x

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.upsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x

    def forward(self, voxel_feats):
        N = voxel_feats.shape[0]

        # encode image
        assert voxel_feats.shape[-1] == self.voxel_feat_dim, \
            f"Feature dimension mismatch: {voxel_feats.shape[-1]} vs {self.voxel_feat_dim}"

        # transformer generating planes
        tokens = self.forward_transformer(voxel_feats)
        planes = self.reshape_upsample(tokens)
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"

        return planes


class TriplaneTransformer(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self, input_dim: int, transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 triplane_low_res: int, triplane_high_res: int, triplane_dim: int):
        super().__init__()
        
        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, transformer_dim) * (1. / transformer_dim) ** 0.5)
        self.transformer = TransformerDecoder(
            block_type='basic',
            num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=transformer_dim,
        )
      
        self.downsampler = nn.Sequential(
            nn.Conv2d(input_dim, transformer_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces size from 128x128 to 64x64
            
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces size from 64x64 to 32x32
        )

        self.upsampler = nn.ConvTranspose2d(transformer_dim, triplane_dim, kernel_size=4, stride=4, padding=0)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, triplane_dim),
            nn.ReLU(),
            nn.Linear(triplane_dim, triplane_dim)
        )

    def forward_transformer(self, triplanes):
        N = triplanes.shape[0]
        tokens = torch.einsum('nidhw->nihwd', triplanes).reshape(N, self.pos_embed.shape[1], -1) # [N, L, D]
        x = self.pos_embed.repeat(N, 1, 1) + tokens # [N, L, D]
        x = self.transformer(x)
        return x

    def reshape_downsample(self, triplanes):
        N = triplanes.shape[0]
        H = W = self.triplane_high_res
        x = triplanes.view(N, 3, -1, H, W)
        x = torch.einsum('nidhw->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.downsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.upsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x
    
    def forward(self, triplanes):
        downsampled_triplanes = self.reshape_downsample(triplanes)
        tokens = self.forward_transformer(downsampled_triplanes)
        residual = self.reshape_upsample(tokens)
        
        triplanes = triplanes.permute(0, 1, 3, 4, 2).contiguous()
        triplanes = self.mlp(triplanes)
        triplanes = triplanes.permute(0, 1, 4, 2, 3).contiguous()
        planes = triplanes + residual
        return planes
