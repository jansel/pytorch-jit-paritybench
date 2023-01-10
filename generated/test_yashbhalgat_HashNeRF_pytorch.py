import sys
_module = sys.modules[__name__]
del sys
camera_pose_visualizer = _module
hash_encoding = _module
load_LINEMOD = _module
load_blender = _module
load_deepvoxels = _module
load_llff = _module
loss = _module
optimizer = _module
radam = _module
ray_utils = _module
run_nerf = _module
run_nerf_helpers = _module
make_gif = _module
plot_losses = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from math import exp


from math import log


from math import floor


from torch import nn


from torch.optim import Optimizer


from functools import reduce


from torch.optim import AdamW


import math


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import random


import time


from torch.distributions import Categorical


import matplotlib.pyplot as plt


def hash(coords, log2_hashmap_size):
    """
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    """
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]
    return torch.tensor((1 << log2_hashmap_size) - 1) & xor_result


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    """
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    """
    box_min, box_max = bounding_box
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        pdb.set_trace()
        xyz = torch.clamp(xyz, min=box_min, max=box_max)
    grid_size = (box_max - box_min) / resolution
    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0]) * grid_size
    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)
    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices


class HashEmbedder(nn.Module):

    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level
        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels - 1))
        self.embeddings = nn.ModuleList([nn.Embedding(2 ** self.log2_hashmap_size, self.n_features_per_level) for i in range(n_levels)])
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        """
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        """
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)
        c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 4] * weights[:, 0][:, None]
        c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 5] * weights[:, 0][:, None]
        c10 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 6] * weights[:, 0][:, None]
        c11 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 7] * weights[:, 0][:, None]
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]
        return c

    def forward(self, x):
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b ** i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(x, self.bounding_box, resolution, self.log2_hashmap_size)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)
        return torch.cat(x_embedded_all, dim=-1)


class SHEncoder(nn.Module):

    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5
        self.out_dim = degree ** 2
        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
        self.C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
        self.C4 = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601, -0.6690465435572892, 0.10578554691520431, -0.6690465435572892, 0.47308734787878004, -1.7701307697799304, 0.6258357354491761]

    def forward(self, input, **kwargs):
        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)
        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
        return result


class NeRF(nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [(nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)) for i in range(D - 1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, 'Not implemented if use_viewdirs=False'
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


class NeRFSmall(nn.Module):

    def __init__(self, num_layers=3, hidden_dim=64, geo_feat_dim=15, num_layers_color=4, hidden_dim_color=64, input_ch=3, input_ch_views=3):
        super(NeRFSmall, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim
            else:
                out_dim = hidden_dim
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = nn.ModuleList(sigma_net)
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            if l == num_layers_color - 1:
                out_dim = 3
            else:
                out_dim = hidden_dim
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.color_net = nn.ModuleList(color_net)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sigma, geo_feat = h[..., 0], h[..., 1:]
        h = torch.cat([input_views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        color = h
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)
        return outputs

