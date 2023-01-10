import sys
_module = sys.modules[__name__]
del sys
main = _module
cfgs = _module
dataset = _module
models = _module
cascade = _module
ecg = _module
grnet = _module
msn = _module
pcn = _module
topnet = _module
vrcnet = _module
test = _module
train = _module
dist_chamfer_2D = _module
setup = _module
dist_chamfer_3D = _module
setup = _module
dist_chamfer_5D = _module
setup = _module
chamfer_python = _module
fscore = _module
unit_test = _module
MDS_module = _module
setup = _module
pointnet2_modules = _module
pointnet2_utils = _module
pytorch_utils = _module
setup = _module
_init_path = _module
dataset = _module
kitti_utils = _module
pointnet2_msg = _module
train_and_eval = _module
utils = _module
cubic_feature_sampling = _module
setup = _module
test = _module
emd_module = _module
setup = _module
expansion_penalty_module = _module
setup = _module
gridding = _module
setup = _module
test = _module
model_utils = _module
train_utils = _module
vis_utils = _module

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


import numpy as np


import torch.utils.data as data


import math


import torch.nn as nn


import torch.nn.parallel


import torch.utils.data


import torch.nn.functional as F


from torch.autograd import Variable


from copy import deepcopy


import logging


import torch.optim as optim


import random


from torch import nn


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import time


from typing import List


from typing import Tuple


import torch.utils.data as torch_data


import torch.optim.lr_scheduler as lr_sched


from torch.nn.utils import clip_grad_norm_


from torch.utils.data import DataLoader


from torch.autograd import gradcheck


class MLP(nn.Module):

    def __init__(self, dims, bn=None):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-2]):
            self.model.add_module('fc_%d' % (i + 1), nn.Linear(num_channels, dims[i + 1]))
        self.bn = bn
        if self.bn:
            self.batch_norm = nn.BatchNorm1d(dims[-2])
        self.output_layer = nn.Linear(dims[-2], dims[-1])

    def forward(self, features):
        features = self.model(features)
        if self.bn:
            features = self.batch_norm(features)
        features = F.relu(features)
        outputs = self.output_layer(features)
        return outputs


class MLPConv(nn.Module):

    def __init__(self, dims, bn=None):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-2]):
            self.model.add_module('conv1d_%d' % (i + 1), nn.Conv1d(num_channels, dims[i + 1], kernel_size=1))
        self.bn = bn
        if self.bn:
            self.batch_norm = nn.BatchNorm1d(dims[-2])
        self.output_layer = nn.Conv1d(dims[-2], dims[-1], kernel_size=1)

    def forward(self, inputs):
        inputs = self.model(inputs)
        if self.bn:
            self.batch_norm
            inputs = self.batch_norm(inputs)
        inputs = F.relu(inputs)
        outputs = self.output_layer(inputs)
        return outputs


class ContractExpandOperation(nn.Module):

    def __init__(self, num_input_channels, up_ratio):
        super().__init__()
        self.up_ratio = up_ratio
        self.conv2d_1 = nn.Conv2d(num_input_channels, 64, kernel_size=(1, self.up_ratio), stride=(1, 1))
        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inputs):
        net = inputs.view(inputs.shape[0], inputs.shape[1], self.up_ratio, -1)
        net = net.permute(0, 1, 3, 2).contiguous()
        net = F.relu(self.conv2d_1(net))
        net = F.relu(self.conv2d_2(net))
        net = net.permute(0, 2, 3, 1).contiguous()
        net = net.view(net.shape[0], -1, self.up_ratio, 64)
        net = net.permute(0, 3, 1, 2).contiguous()
        net = F.relu(self.conv2d_3(net))
        net = net.view(net.shape[0], 64, -1)
        return net


class Encoder(nn.Module):

    def __init__(self, embed_size=1024):
        super().__init__()
        self.conv1 = MLPConv([3, 128, 256])
        self.conv2 = MLPConv([512, 512, embed_size])

    def forward(self, inputs):
        """
        :param inputs: B * C * N
        :return: B * C
        """
        features = self.conv1(inputs)
        features_global, _ = torch.max(features, 2, keepdim=True)
        features_global_tiled = features_global.repeat(1, 1, inputs.shape[2])
        features = torch.cat([features, features_global_tiled], dim=1)
        features = self.conv2(features)
        features, _ = torch.max(features, 2)
        return features


def gen_grid_up(up_ratio, grid_size=0.2):
    sqrted = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if up_ratio % i == 0:
            num_x = i
            num_y = up_ratio // i
            break
    grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
    grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)
    x, y = torch.meshgrid(grid_x, grid_y)
    grid = torch.stack([x, y], dim=-1).view(-1, 2).transpose(0, 1).contiguous()
    return grid


def symmetric_sample(points, num=512):
    p1_idx = pn2.furthest_point_sample(points, num)
    input_fps = pn2.gather_operation(points.transpose(1, 2).contiguous(), p1_idx).transpose(1, 2).contiguous()
    x = torch.unsqueeze(input_fps[:, :, 0], dim=2)
    y = torch.unsqueeze(input_fps[:, :, 1], dim=2)
    z = torch.unsqueeze(-input_fps[:, :, 2], dim=2)
    input_fps_flip = torch.cat([x, y, z], dim=2)
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)
    return input_fps


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.coarse_mlp = MLP([1024, 1024, 1024, 512 * 3])
        self.mean_fc = nn.Linear(1024, 128)
        self.up_branch_mlp_conv_mf = MLPConv([1157, 128, 64])
        self.up_branch_mlp_conv_nomf = MLPConv([1029, 128, 64])
        self.contract_expand = ContractExpandOperation(64, 2)
        self.fine_mlp_conv = MLPConv([64, 512, 512, 3])

    def forward(self, code, inputs, step_ratio, num_extract=512, mean_feature=None):
        """
        :param code: B * C
        :param inputs: B * C * N
        :param step_ratio: int
        :param num_extract: int
        :param mean_feature: B * C
        :return: coarse(B * N * C), fine(B, N, C)
        """
        coarse = torch.tanh(self.coarse_mlp(code))
        coarse = coarse.view(-1, 512, 3)
        coarse = coarse.transpose(2, 1).contiguous()
        inputs_new = inputs.transpose(2, 1).contiguous()
        input_fps = symmetric_sample(inputs_new, int(num_extract / 2))
        input_fps = input_fps.transpose(2, 1).contiguous()
        level0 = torch.cat([input_fps, coarse], 2)
        if num_extract > 512:
            level0_flipped = level0.transpose(2, 1).contiguous()
            level0 = pn2.gather_operation(level0, pn2.furthest_point_sample(level0_flipped, 1024))
        for i in range(int(math.log2(step_ratio))):
            num_fine = 2 ** (i + 1) * 1024
            grid = gen_grid_up(2 ** (i + 1)).contiguous()
            grid = torch.unsqueeze(grid, 0)
            grid_feat = grid.repeat(level0.shape[0], 1, 1024)
            point_feat = torch.unsqueeze(level0, 3).repeat(1, 1, 1, 2)
            point_feat = point_feat.view(-1, 3, num_fine)
            global_feat = torch.unsqueeze(code, 2).repeat(1, 1, num_fine)
            if mean_feature is not None:
                mean_feature_use = F.relu(self.mean_fc(mean_feature))
                mean_feature_use = torch.unsqueeze(mean_feature_use, 2).repeat(1, 1, num_fine)
                feat = torch.cat([grid_feat, point_feat, global_feat, mean_feature_use], dim=1)
                feat1 = F.relu(self.up_branch_mlp_conv_mf(feat))
            else:
                feat = torch.cat([grid_feat, point_feat, global_feat], dim=1)
                feat1 = F.relu(self.up_branch_mlp_conv_nomf(feat))
            feat2 = self.contract_expand(feat1)
            feat = feat1 + feat2
            fine = self.fine_mlp_conv(feat) + point_feat
            level0 = fine
        return coarse.transpose(1, 2).contiguous(), fine.transpose(1, 2).contiguous()


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor=None, new_xyz=None) ->(torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)).transpose(1, 2).contiguous() if self.npoint is not None else None
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool=True, use_xyz: bool=True, pool_method='max_pool', instance_norm=False, features: torch.Tensor=None):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if features is not None:
                if use_xyz:
                    mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
        self.pool_method = pool_method


class Discriminator(nn.Module):

    def __init__(self, args, divide_ratio=2):
        super(Discriminator, self).__init__()
        self.num_points = args.num_points
        self.pointnet_sa_module = PointnetSAModuleMSG(npoint=int(self.num_points / 8), radii=[0.1, 0.2, 0.4], nsamples=[16, 32, 128], mlps=[[3, 32 // divide_ratio, 32 // divide_ratio, 64 // divide_ratio], [3, 64 // divide_ratio, 64 // divide_ratio, 128 // divide_ratio], [3, 64 // divide_ratio, 96 // divide_ratio, 128 // divide_ratio]])
        self.patch_mlp_conv = MLPConv([64 // divide_ratio + 128 // divide_ratio + 128 // divide_ratio, 1])

    def forward(self, xyz):
        _, l1_points = self.pointnet_sa_module(xyz, features=None)
        patch_values = self.patch_mlp_conv(l1_points)
        return patch_values


class Linear_ResBlock(nn.Module):

    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)
        self.af = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, minus_center=True):
    idx = knn(x, k=k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if minus_center:
        feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)
    else:
        feature = torch.cat((x, feature), dim=3).permute(0, 3, 1, 2)
    return feature


class EF_expansion(nn.Module):

    def __init__(self, input_size, output_size=64, step_ratio=2, k=4):
        super(EF_expansion, self).__init__()
        self.step_ratio = step_ratio
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = nn.Conv2d(input_size * 2, output_size, 1)
        self.conv2 = nn.Conv2d(input_size * 2 + output_size, output_size * step_ratio, 1)
        self.conv3 = nn.Conv2d(output_size, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        input_edge_feature = get_graph_feature(x, self.k, minus_center=False).permute(0, 1, 3, 2).contiguous()
        edge_feature = self.conv1(input_edge_feature)
        edge_feature = F.relu(torch.cat((edge_feature, input_edge_feature), 1))
        edge_feature = F.relu(self.conv2(edge_feature))
        edge_feature = edge_feature.permute(0, 2, 3, 1).contiguous().view(batch_size, self.k, num_points * self.step_ratio, self.output_size).permute(0, 3, 1, 2)
        edge_feature = self.conv3(edge_feature)
        edge_feature, _ = torch.max(edge_feature, 2)
        return edge_feature


class Folding(nn.Module):

    def __init__(self, input_size, output_size, step_ratio, global_feature_size=1024, num_models=1):
        super(Folding, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.step_ratio = step_ratio
        self.num_models = num_models
        self.conv = nn.Conv1d(input_size + global_feature_size + 2, output_size, 1, bias=True)
        sqrted = int(math.sqrt(step_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if step_ratio % i == 0:
                num_x = i
                num_y = step_ratio // i
                break
        grid_x = torch.linspace(-0.2, 0.2, steps=num_x)
        grid_y = torch.linspace(-0.2, 0.2, steps=num_y)
        x, y = torch.meshgrid(grid_x, grid_y)
        self.grid = torch.stack([x, y], dim=-1).view(-1, 2)

    def forward(self, point_feat, global_feat):
        batch_size, num_features, num_points = point_feat.size()
        point_feat = point_feat.transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, self.step_ratio, 1).view(batch_size, -1, num_features).transpose(1, 2).contiguous()
        global_feat = global_feat.unsqueeze(2).repeat(1, 1, num_points * self.step_ratio).repeat(self.num_models, 1, 1)
        grid_feat = self.grid.unsqueeze(0).repeat(batch_size, num_points, 1).transpose(1, 2).contiguous()
        features = torch.cat([global_feat, point_feat, grid_feat], axis=1)
        features = F.relu(self.conv(features))
        return features


def get_edge_features(x, idx):
    batch_size, num_points, k = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.squeeze(2)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims).permute(0, 3, 2, 1)
    return feature


class SA_module(nn.Module):

    def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=16):
        super(SA_module, self).__init__()
        self.share_planes = share_planes
        self.k = k
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)
        self.conv_w = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(rel_planes * (k + 1), mid_planes // share_planes, kernel_size=1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(mid_planes // share_planes, k * mid_planes // share_planes, kernel_size=1))
        self.activation_fn = nn.ReLU(inplace=False)
        self.conv_out = nn.Conv2d(mid_planes, out_planes, kernel_size=1)

    def forward(self, input):
        x, idx = input
        batch_size, _, _, num_points = x.size()
        identity = x
        x = self.activation_fn(x)
        xn = get_edge_features(x, idx)
        x1, x2, x3 = self.conv1(x), self.conv2(xn), self.conv3(xn)
        x2 = x2.view(batch_size, -1, 1, num_points).contiguous()
        w = self.conv_w(torch.cat([x1, x2], 1)).view(batch_size, -1, self.k, num_points)
        w = w.repeat(1, self.share_planes, 1, 1)
        out = w * x3
        out = torch.sum(out, dim=2, keepdim=True)
        out = self.activation_fn(out)
        out = self.conv_out(out)
        out += identity
        return [out, idx]


class SK_SA_module(nn.Module):

    def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=[10, 20], r=2, L=32):
        super(SK_SA_module, self).__init__()
        self.num_kernels = len(k)
        d = max(int(out_planes / r), L)
        self.sams = nn.ModuleList([])
        for i in range(len(k)):
            self.sams.append(SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k[i]))
        self.fc = nn.Linear(out_planes, d)
        self.fcs = nn.ModuleList([])
        for i in range(len(k)):
            self.fcs.append(nn.Linear(d, out_planes))
        self.softmax = nn.Softmax(dim=1)
        self.af = nn.ReLU(inplace=False)

    def forward(self, input):
        x, idxs = input
        assert self.num_kernels == len(idxs)
        for i, sam in enumerate(self.sams):
            fea, _ = sam([x, idxs[i]])
            fea = self.af(fea)
            fea = fea.unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return [fea_v, idxs]


class SKN_Res_unit(nn.Module):

    def __init__(self, input_size, output_size, k=[10, 20], layers=1):
        super(SKN_Res_unit, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, 1, bias=False)
        self.sam = self._make_layer(output_size, output_size // 16, output_size // 4, output_size, int(layers), 8, k=k)
        self.conv2 = nn.Conv2d(output_size, output_size, 1, bias=False)
        self.conv_res = nn.Conv2d(input_size, output_size, 1, bias=False)
        self.af = nn.ReLU(inplace=False)

    def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
        layers = []
        for _ in range(0, blocks):
            layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
        return nn.Sequential(*layers)

    def forward(self, feat, idx):
        x, _ = self.sam([self.conv1(feat), idx])
        x = self.conv2(self.af(x))
        return x + self.conv_res(feat)


def knn_point(pk, point_input, point_output):
    m = point_output.size()[1]
    n = point_input.size()[1]
    inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
    pairwise_distance = -xx - inner - yy
    dist, idx = pairwise_distance.topk(k=pk, dim=-1)
    return dist, idx


def edge_preserve_sampling(feature_input, point_input, num_samples, k=10):
    batch_size = feature_input.size()[0]
    feature_size = feature_input.size()[1]
    num_points = feature_input.size()[2]
    p_idx = pn2.furthest_point_sample(point_input, num_samples)
    point_output = pn2.gather_operation(point_input.transpose(1, 2).contiguous(), p_idx).transpose(1, 2).contiguous()
    pk = int(min(k, num_points))
    _, pn_idx = knn_point(pk, point_input, point_output)
    pn_idx = pn_idx.detach().int()
    neighbor_feature = pn2.gather_operation(feature_input, pn_idx.view(batch_size, num_samples * pk)).view(batch_size, feature_size, num_samples, pk)
    neighbor_feature, _ = torch.max(neighbor_feature, 3)
    center_feature = pn2.grouping_operation(feature_input, p_idx.unsqueeze(2)).view(batch_size, -1, num_samples)
    net = torch.cat((center_feature, neighbor_feature), 1)
    return net, p_idx, pn_idx, point_output


def three_nn_upsampling(target_points, source_points):
    dist, idx = pn2.three_nn(target_points, source_points)
    dist = torch.max(dist, torch.ones(1) * 1e-10)
    norm = torch.sum(1.0 / dist, 2, keepdim=True)
    norm = norm.repeat(1, 1, 3)
    weight = 1.0 / dist / norm
    return idx, weight


class SA_SKN_Res_encoder(nn.Module):

    def __init__(self, input_size=3, k=[10, 20], pk=16, output_size=64, layers=[2, 2, 2, 2], pts_num=[3072, 1536, 768, 384]):
        super(SA_SKN_Res_encoder, self).__init__()
        self.init_channel = 64
        c1 = self.init_channel
        self.sam_res1 = SKN_Res_unit(input_size, c1, k, int(layers[0]))
        c2 = c1 * 2
        self.sam_res2 = SKN_Res_unit(c2, c2, k, int(layers[1]))
        c3 = c2 * 2
        self.sam_res3 = SKN_Res_unit(c3, c3, k, int(layers[2]))
        c4 = c3 * 2
        self.sam_res4 = SKN_Res_unit(c4, c4, k, int(layers[3]))
        self.conv5 = nn.Conv2d(c4, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.conv6 = nn.Conv2d(c4 + 1024, c4, 1)
        self.conv7 = nn.Conv2d(c3 + c4, c3, 1)
        self.conv8 = nn.Conv2d(c2 + c3, c2, 1)
        self.conv9 = nn.Conv2d(c1 + c2, c1, 1)
        self.conv_out = nn.Conv2d(c1, output_size, 1)
        self.dropout = nn.Dropout()
        self.af = nn.ReLU(inplace=False)
        self.k = k
        self.pk = pk
        self.rate = 2
        self.pts_num = pts_num

    def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
        layers = []
        for _ in range(0, blocks):
            layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
        return nn.Sequential(*layers)

    def _edge_pooling(self, features, points, rate=2, k=16, sample_num=None):
        features = features.squeeze(2)
        if sample_num is None:
            input_points_num = int(features.size()[2])
            sample_num = input_points_num // rate
        ds_features, p_idx, pn_idx, ds_points = edge_preserve_sampling(features, points, sample_num, k)
        ds_features = ds_features.unsqueeze(2)
        return ds_features, p_idx, pn_idx, ds_points

    def _edge_unpooling(self, features, src_pts, tgt_pts):
        features = features.squeeze(2)
        idx, weight = three_nn_upsampling(tgt_pts, src_pts)
        features = pn2.three_interpolate(features, idx, weight)
        features = features.unsqueeze(2)
        return features

    def forward(self, features):
        batch_size, _, num_points = features.size()
        pt1 = features[:, 0:3, :]
        idx1 = []
        for i in range(len(self.k)):
            idx = knn(pt1, self.k[i])
            idx1.append(idx)
        pt1 = pt1.transpose(1, 2).contiguous()
        x = features.unsqueeze(2)
        x = self.sam_res1(x, idx1)
        x1 = self.af(x)
        x, _, _, pt2 = self._edge_pooling(x1, pt1, self.rate, self.pk, self.pts_num[1])
        idx2 = []
        for i in range(len(self.k)):
            idx = knn(pt2.transpose(1, 2).contiguous(), self.k[i])
            idx2.append(idx)
        x = self.sam_res2(x, idx2)
        x2 = self.af(x)
        x, _, _, pt3 = self._edge_pooling(x2, pt2, self.rate, self.pk, self.pts_num[2])
        idx3 = []
        for i in range(len(self.k)):
            idx = knn(pt3.transpose(1, 2).contiguous(), self.k[i])
            idx3.append(idx)
        x = self.sam_res3(x, idx3)
        x3 = self.af(x)
        x, _, _, pt4 = self._edge_pooling(x3, pt3, self.rate, self.pk, self.pts_num[3])
        idx4 = []
        for i in range(len(self.k)):
            idx = knn(pt4.transpose(1, 2).contiguous(), self.k[i])
            idx4.append(idx)
        x = self.sam_res4(x, idx4)
        x4 = self.af(x)
        x = self.conv5(x4)
        x, _ = torch.max(x, -1)
        x = x.view(batch_size, -1)
        x = self.dropout(self.af(self.fc2(self.dropout(self.af(self.fc1(x))))))
        x = x.unsqueeze(2).repeat(1, 1, self.pts_num[3]).unsqueeze(2)
        x = self.af(self.conv6(torch.cat([x, x4], 1)))
        x = self._edge_unpooling(x, pt4, pt3)
        x = self.af(self.conv7(torch.cat([x, x3], 1)))
        x = self._edge_unpooling(x, pt3, pt2)
        x = self.af(self.conv8(torch.cat([x, x2], 1)))
        x = self._edge_unpooling(x, pt2, pt1)
        x = self.af(self.conv9(torch.cat([x, x1], 1)))
        x = self.conv_out(x)
        x = x.squeeze(2)
        return x


class MSAP_SKN_decoder(nn.Module):

    def __init__(self, num_coarse_raw, num_fps, num_coarse, num_fine, layers=[2, 2, 2, 2], knn_list=[10, 20], pk=10, points_label=False, local_folding=False):
        super(MSAP_SKN_decoder, self).__init__()
        self.num_coarse_raw = num_coarse_raw
        self.num_fps = num_fps
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.points_label = points_label
        self.local_folding = local_folding
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse_raw * 3)
        self.dense_feature_size = 256
        self.expand_feature_size = 64
        if points_label:
            self.input_size = 4
        else:
            self.input_size = 3
        self.encoder = SA_SKN_Res_encoder(input_size=self.input_size, k=knn_list, pk=pk, output_size=self.dense_feature_size, layers=layers)
        self.up_scale = int(np.ceil(num_fine / (num_coarse_raw + 2048)))
        if self.up_scale >= 2:
            self.expansion1 = EF_expansion(input_size=self.dense_feature_size, output_size=self.expand_feature_size, step_ratio=self.up_scale, k=4)
            self.conv_cup1 = nn.Conv1d(self.expand_feature_size, self.expand_feature_size, 1)
        else:
            self.expansion1 = None
            self.conv_cup1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        self.conv_cup2 = nn.Conv1d(self.expand_feature_size, 3, 1, bias=True)
        self.conv_s1 = nn.Conv1d(self.expand_feature_size, 16, 1, bias=True)
        self.conv_s2 = nn.Conv1d(16, 8, 1, bias=True)
        self.conv_s3 = nn.Conv1d(8, 1, 1, bias=True)
        if self.local_folding:
            self.expansion2 = Folding(input_size=self.expand_feature_size, output_size=self.dense_feature_size, step_ratio=num_fine // num_coarse)
        else:
            self.expansion2 = EF_expansion(input_size=self.expand_feature_size, output_size=self.dense_feature_size, step_ratio=num_fine // num_coarse, k=4)
        self.conv_f1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        self.conv_f2 = nn.Conv1d(self.expand_feature_size, 3, 1)
        self.af = nn.ReLU(inplace=False)

    def forward(self, global_feat, point_input):
        batch_size = global_feat.size()[0]
        coarse_raw = self.fc3(self.af(self.fc2(self.af(self.fc1(global_feat))))).view(batch_size, 3, self.num_coarse_raw)
        input_points_num = point_input.size()[2]
        org_points_input = point_input
        if self.points_label:
            id0 = torch.zeros(coarse_raw.shape[0], 1, coarse_raw.shape[2]).contiguous()
            coarse_input = torch.cat((coarse_raw, id0), 1)
            id1 = torch.ones(org_points_input.shape[0], 1, org_points_input.shape[2]).contiguous()
            org_points_input = torch.cat((org_points_input, id1), 1)
        else:
            coarse_input = coarse_raw
        points = torch.cat((coarse_input, org_points_input), 2)
        dense_feat = self.encoder(points)
        if self.up_scale >= 2:
            dense_feat = self.expansion1(dense_feat)
        coarse_features = self.af(self.conv_cup1(dense_feat))
        coarse_high = self.conv_cup2(coarse_features)
        if coarse_high.size()[2] > self.num_fps:
            idx_fps = pn2.furthest_point_sample(coarse_high.transpose(1, 2).contiguous(), self.num_fps)
            coarse_fps = pn2.gather_operation(coarse_high, idx_fps)
            coarse_features = pn2.gather_operation(coarse_features, idx_fps)
        else:
            coarse_fps = coarse_high
        if coarse_fps.size()[2] > self.num_coarse:
            scores = F.softplus(self.conv_s3(self.af(self.conv_s2(self.af(self.conv_s1(coarse_features))))))
            idx_scores = scores.topk(k=self.num_coarse, dim=2)[1].view(batch_size, -1).int()
            coarse = pn2.gather_operation(coarse_fps, idx_scores)
            coarse_features = pn2.gather_operation(coarse_features, idx_scores)
        else:
            coarse = coarse_fps
        if coarse.size()[2] < self.num_fine:
            if self.local_folding:
                up_features = self.expansion2(coarse_features, global_feat)
                center = coarse.transpose(2, 1).contiguous().unsqueeze(2).repeat(1, 1, self.num_fine // self.num_coarse, 1).view(batch_size, self.num_fine, 3).transpose(2, 1).contiguous()
                fine = self.conv_f2(self.af(self.conv_f1(up_features))) + center
            else:
                up_features = self.expansion2(coarse_features)
                fine = self.conv_f2(self.af(self.conv_f1(up_features)))
        else:
            assert coarse.size()[2] == self.num_fine
            fine = coarse
        return coarse_raw, coarse_high, coarse, fine


class PCN_encoder(nn.Module):

    def __init__(self, output_size=1024):
        super(PCN_encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)


def fscore(dist1, dist2, threshold=0.0001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


def calc_cd(output, gt, calc_f1=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = dist1.mean(1) + dist2.mean(1)
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t


def calc_emd(output, gt, eps=0.005, iterations=50):
    emd_loss = emd.emdModule()
    dist, _ = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sqrt(dist).mean(1)
    return emd_out


class Model(nn.Module):

    def __init__(self, args, size_z=128, global_feature_size=1024):
        super(Model, self).__init__()
        layers = [int(i) for i in args.layers.split(',')]
        knn_list = [int(i) for i in args.knn_list.split(',')]
        self.size_z = size_z
        self.distribution_loss = args.distribution_loss
        self.train_loss = args.loss
        self.encoder = PCN_encoder(output_size=global_feature_size)
        self.posterior_infer1 = Linear_ResBlock(input_size=global_feature_size, output_size=global_feature_size)
        self.posterior_infer2 = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        self.prior_infer = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        self.generator = Linear_ResBlock(input_size=size_z, output_size=global_feature_size)
        self.decoder = MSAP_SKN_decoder(num_fps=args.num_fps, num_fine=args.num_points, num_coarse=args.num_coarse, num_coarse_raw=args.num_coarse_raw, layers=layers, knn_list=knn_list, pk=args.pk, local_folding=args.local_folding, points_label=args.points_label)

    def compute_kernel(self, x, y):
        x_size = x.size()[0]
        y_size = y.size()[0]
        dim = x.size()[1]
        tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / float(dim))

    def mmd_loss(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):
        num_input = x.size()[2]
        if is_training:
            y = pn2.gather_operation(gt.transpose(1, 2).contiguous(), pn2.furthest_point_sample(gt, num_input))
            gt = torch.cat([gt, gt], dim=0)
            points = torch.cat([x, y], dim=0)
            x = torch.cat([x, x], dim=0)
        else:
            points = x
        feat = self.encoder(points)
        if is_training:
            feat_x, feat_y = feat.chunk(2)
            o_x = self.posterior_infer2(self.posterior_infer1(feat_x))
            q_mu, q_std = torch.split(o_x, self.size_z, dim=1)
            o_y = self.prior_infer(feat_y)
            p_mu, p_std = torch.split(o_y, self.size_z, dim=1)
            q_std = F.softplus(q_std)
            p_std = F.softplus(p_std)
            q_distribution = torch.distributions.Normal(q_mu, q_std)
            p_distribution = torch.distributions.Normal(p_mu, p_std)
            p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_std.detach())
            m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_std))
            z_q = q_distribution.rsample()
            z_p = p_distribution.rsample()
            z = torch.cat([z_q, z_p], dim=0)
            feat = torch.cat([feat_x, feat_x], dim=0)
        else:
            o_x = self.posterior_infer2(self.posterior_infer1(feat))
            q_mu, q_std = torch.split(o_x, self.size_z, dim=1)
            q_std = F.softplus(q_std)
            q_distribution = torch.distributions.Normal(q_mu, q_std)
            p_distribution = q_distribution
            p_distribution_fix = p_distribution
            m_distribution = p_distribution
            z = q_distribution.rsample()
        feat += self.generator(z)
        coarse_raw, coarse_high, coarse, fine = self.decoder(feat, x)
        coarse_raw = coarse_raw.transpose(1, 2).contiguous()
        coarse_high = coarse_high.transpose(1, 2).contiguous()
        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        if is_training:
            if self.distribution_loss == 'MMD':
                z_m = m_distribution.rsample()
                z_q = q_distribution.rsample()
                z_p = p_distribution.rsample()
                z_p_fix = p_distribution_fix.rsample()
                dl_rec = self.mmd_loss(z_m, z_p)
                dl_g = self.mmd_loss2(z_q, z_p_fix)
            elif self.distribution_loss == 'KLD':
                dl_rec = torch.distributions.kl_divergence(m_distribution, p_distribution)
                dl_g = torch.distributions.kl_divergence(p_distribution_fix, q_distribution)
            else:
                raise NotImplementedError('Distribution loss is either MMD or KLD')
            if self.train_loss == 'cd':
                loss1, _ = calc_cd(coarse_raw, gt)
                loss2, _ = calc_cd(coarse_high, gt)
                loss3, _ = calc_cd(coarse, gt)
                loss4, _ = calc_cd(fine, gt)
            else:
                raise NotImplementedError('Only CD is supported')
            total_train_loss = loss1.mean() * 10 + loss2.mean() * 0.5 + loss3.mean() + loss4.mean() * alpha
            total_train_loss += (dl_rec.mean() + dl_g.mean()) * 20
            return fine, loss4, total_train_loss
        else:
            emd = calc_emd(fine, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(fine, gt, calc_f1=True)
            return {'out1': coarse_raw, 'out2': fine, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}


class Stack_conv(nn.Module):

    def __init__(self, input_size, output_size, act=None):
        super(Stack_conv, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv', nn.Conv2d(input_size, output_size, 1))
        if act is not None:
            self.model.add_module('act', act)

    def forward(self, x):
        y = self.model(x)
        y = torch.cat((x, y), 1)
        return y


class Dense_conv(nn.Module):

    def __init__(self, input_size, growth_rate=64, dense_n=3, k=16):
        super(Dense_conv, self).__init__()
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.k = k
        self.comp = growth_rate * 2
        self.input_size = input_size
        self.first_conv = nn.Conv2d(self.input_size * 2, growth_rate, 1)
        self.input_size += self.growth_rate
        self.model = nn.Sequential()
        for i in range(dense_n - 1):
            if i == dense_n - 2:
                self.model.add_module('stack_conv_%d' % (i + 1), Stack_conv(self.input_size, self.growth_rate, None))
            else:
                self.model.add_module('stack_conv_%d' % (i + 1), Stack_conv(self.input_size, self.growth_rate, nn.ReLU()))
                self.input_size += growth_rate

    def forward(self, x):
        y = get_graph_feature(x, k=self.k)
        y = F.relu(self.first_conv(y))
        y = torch.cat((y, x.unsqueeze(3).repeat(1, 1, 1, self.k)), 1)
        y = self.model(y)
        y, _ = torch.max(y, 3)
        return y


class EF_encoder(nn.Module):

    def __init__(self, growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64], input_size=3, output_size=256):
        super(EF_encoder, self).__init__()
        self.growth_rate = growth_rate
        self.comp = growth_rate * 2
        self.dense_n = dense_n
        self.k = k
        self.hierarchy = hierarchy
        self.init_channel = 24
        self.conv1 = nn.Conv1d(input_size, self.init_channel, 1)
        self.dense_conv1 = Dense_conv(self.init_channel, self.growth_rate, self.dense_n, self.k)
        out_channel_size_1 = self.init_channel * 2 + self.growth_rate * self.dense_n
        self.conv2 = nn.Conv1d(out_channel_size_1 * 2, self.comp, 1)
        self.dense_conv2 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)
        out_channel_size_2 = out_channel_size_1 * 2 + self.comp + self.growth_rate * self.dense_n
        self.conv3 = nn.Conv1d(out_channel_size_2 * 2, self.comp, 1)
        self.dense_conv3 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)
        out_channel_size_3 = out_channel_size_2 * 2 + self.comp + self.growth_rate * self.dense_n
        self.conv4 = nn.Conv1d(out_channel_size_3 * 2, self.comp, 1)
        self.dense_conv4 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)
        out_channel_size_4 = out_channel_size_3 * 2 + self.comp + self.growth_rate * self.dense_n
        self.gf_conv = nn.Conv1d(out_channel_size_4, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)
        out_channel_size = out_channel_size_4 + 1024
        self.conv5 = nn.Conv1d(out_channel_size, 1024, 1)
        out_channel_size = out_channel_size_3 + 1024
        self.conv6 = nn.Conv1d(out_channel_size, 768, 1)
        out_channel_size = out_channel_size_2 + 768
        self.conv7 = nn.Conv1d(out_channel_size, 512, 1)
        out_channel_size = out_channel_size_1 + 512
        self.conv8 = nn.Conv1d(out_channel_size, output_size, 1)

    def forward(self, x):
        point_cloud1 = x[:, 0:3, :]
        point_cloud1 = point_cloud1.transpose(1, 2).contiguous()
        x0 = F.relu(self.conv1(x))
        x1 = F.relu(self.dense_conv1(x0))
        x1 = torch.cat((x1, x0), 1)
        x1d, _, _, point_cloud2 = edge_preserve_sampling(x1, point_cloud1, self.hierarchy[0], self.k)
        x2 = F.relu(self.conv2(x1d))
        x2 = F.relu(self.dense_conv2(x2))
        x2 = torch.cat((x2, x1d), 1)
        x2d, _, _, point_cloud3 = edge_preserve_sampling(x2, point_cloud2, self.hierarchy[1], self.k)
        x3 = F.relu(self.conv3(x2d))
        x3 = F.relu(self.dense_conv3(x3))
        x3 = torch.cat((x3, x2d), 1)
        x3d, _, _, point_cloud4 = edge_preserve_sampling(x3, point_cloud3, self.hierarchy[2], self.k)
        x4 = F.relu(self.conv4(x3d))
        x4 = F.relu(self.dense_conv4(x4))
        x4 = torch.cat((x4, x3d), 1)
        global_feat = self.gf_conv(x4)
        global_feat, _ = torch.max(global_feat, -1)
        global_feat = F.relu(self.fc1(global_feat))
        global_feat = F.relu(self.fc2(global_feat)).unsqueeze(2).repeat(1, 1, self.hierarchy[2])
        x4 = torch.cat((global_feat, x4), 1)
        x4 = F.relu(self.conv5(x4))
        idx, weight = three_nn_upsampling(point_cloud3, point_cloud4)
        x4 = pn2.three_interpolate(x4, idx, weight)
        x3 = torch.cat((x3, x4), 1)
        x3 = F.relu(self.conv6(x3))
        idx, weight = three_nn_upsampling(point_cloud2, point_cloud3)
        x3 = pn2.three_interpolate(x3, idx, weight)
        x2 = torch.cat((x2, x3), 1)
        x2 = F.relu(self.conv7(x2))
        idx, weight = three_nn_upsampling(point_cloud1, point_cloud2)
        x2 = pn2.three_interpolate(x2, idx, weight)
        x1 = torch.cat((x1, x2), 1)
        x1 = self.conv8(x1)
        return x1


class ECG_decoder(nn.Module):

    def __init__(self, num_coarse, num_fine, num_input):
        super(ECG_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.scale = int(np.ceil(num_fine / (num_coarse + num_input)))
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)
        self.dense_feature_size = 256
        self.expand_feature_size = 64
        self.input_size = 3
        self.encoder = EF_encoder(growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64], input_size=self.input_size, output_size=self.dense_feature_size)
        if self.scale >= 2:
            self.expansion = EF_expansion(input_size=self.dense_feature_size, output_size=self.expand_feature_size, step_ratio=self.scale, k=4)
            self.conv1 = nn.Conv1d(self.expand_feature_size, self.expand_feature_size, 1)
        else:
            self.expansion = None
            self.conv1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        self.conv2 = nn.Conv1d(self.expand_feature_size, 3, 1)

    def forward(self, global_feat, point_input):
        batch_size = global_feat.size()[0]
        coarse = F.relu(self.fc1(global_feat))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(batch_size, 3, self.num_coarse)
        org_points_input = point_input
        points = torch.cat((coarse, org_points_input), 2)
        dense_feat = self.encoder(points)
        if self.scale >= 2:
            dense_feat = self.expansion(dense_feat)
        point_feat = F.relu(self.conv1(dense_feat))
        fine = self.conv2(point_feat)
        num_out = fine.size()[2]
        if num_out > self.num_fine:
            fine = pn2.gather_operation(fine, pn2.furthest_point_sample(fine.transpose(1, 2).contiguous(), self.num_fine))
        return coarse, fine


class RandomPointSampling(torch.nn.Module):

    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)
        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points,))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])
        return torch.cat(ptclouds, dim=0).contiguous()


class STN3d(nn.Module):

    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):

    def __init__(self, num_points=8192, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class PointGenCon(nn.Module):

    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)
        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class PointNetRes(nn.Module):

    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x


class PCN_decoder(nn.Module):

    def __init__(self, num_coarse, num_fine, scale, cat_feature_num):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)
        self.scale = scale
        self.grid = gen_grid_up(2 ** int(math.log2(scale)), 0.05).contiguous()
        self.conv1 = nn.Conv1d(cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)
        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous()
        point_feat = coarse.transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine, 3).transpose(1, 2).contiguous()
        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)
        feat = torch.cat((grid_feat, point_feat, global_feat), 1)
        center = coarse.transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine, 3).transpose(1, 2).contiguous()
        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine


class CreateLevel(nn.Module):

    def __init__(self, level, input_channels, output_channels, bn, tarch):
        super().__init__()
        self.output_channels = output_channels
        self.mlp_conv = MLPConv([input_channels, input_channels, int(input_channels / 2), int(input_channels / 4), int(input_channels / 8), output_channels * int(tarch[level])], bn=bn)

    def forward(self, inputs):
        features = self.mlp_conv(inputs)
        features = features.view(features.shape[0], self.output_channels, -1)
        return features


class PCNEncoder(nn.Module):

    def __init__(self, embed_size=1024):
        super().__init__()
        self.conv1 = MLPConv([3, 128, 256])
        self.conv2 = MLPConv([512, 512, embed_size])

    def forward(self, inputs):
        """
        :param inputs: B * C * N
        :return: B * C
        """
        features = self.conv1(inputs)
        features_global, _ = torch.max(features, 2, keepdim=True)
        features_global_tiled = features_global.repeat(1, 1, inputs.shape[2])
        features = torch.cat([features, features_global_tiled], dim=1)
        features = self.conv2(features)
        features, _ = torch.max(features, 2)
        return features


tree_arch = {}


def get_arch(nlevels, npts):
    logmult = int(math.log2(npts / 2048))
    assert 2048 * 2 ** logmult == npts, 'Number of points is %d, expected 2048x(2^n)' % npts
    arch = deepcopy(tree_arch[nlevels])
    while logmult > 0:
        last_min_pos = np.where(arch == np.min(arch))[0][-1]
        arch[last_min_pos] *= 2
        logmult -= 1
    return arch


class TopnetDecoder(nn.Module):

    def __init__(self, npts):
        super().__init__()
        self.tarch = get_arch(6, npts)
        self.N = int(np.prod([int(k) for k in self.tarch]))
        assert self.N == npts, 'Number of tree outputs is %d, expected %d' % (self.N, npts)
        self.NFEAT = 8
        self.CODE_NFTS = 1024
        self.Nin = self.NFEAT + self.CODE_NFTS
        self.Nout = self.NFEAT
        self.N0 = int(self.tarch[0])
        self.nlevels = len(self.tarch)
        self.mlp = MLP([1024, 256, 64, self.NFEAT * self.N0], bn=True)
        self.mlp_conv_list = nn.ModuleList()
        bn = True
        for i in range(1, self.nlevels):
            if i == self.nlevels - 1:
                self.Nout = 3
                bn = False
            self.mlp_conv_list.append(CreateLevel(i, self.Nin, self.Nout, bn, self.tarch))

    def forward(self, code):
        level0 = self.mlp(code)
        level0 = torch.tanh(level0)
        level0 = level0.view(-1, self.NFEAT, self.N0)
        outs = [level0]
        for i in range(self.nlevels - 1):
            inp = outs[-1]
            y = torch.unsqueeze(code, dim=2)
            y = y.repeat(1, 1, inp.shape[2])
            y = torch.cat([inp, y], dim=1)
            conv_outs = self.mlp_conv_list[i](y)
            outs.append(torch.tanh(conv_outs))
        return outs[-1]


class chamfer_2DFunction(Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        dist1 = dist1
        dist2 = dist2
        idx1 = idx1
        idx2 = idx2
        torch.cuda.set_device(device)
        chamfer_2D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        gradxyz1 = gradxyz1
        gradxyz2 = gradxyz2
        chamfer_2D.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class chamfer_2DDist(nn.Module):

    def __init__(self):
        super(chamfer_2DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_2DFunction.apply(input1, input2)


class chamfer_3DFunction(Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        dist1 = dist1
        dist2 = dist2
        idx1 = idx1
        idx2 = idx2
        torch.cuda.set_device(device)
        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        gradxyz1 = gradxyz1
        gradxyz2 = gradxyz2
        chamfer_3D.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):

    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)


class chamfer_5DFunction(Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        dist1 = dist1
        dist2 = dist2
        idx1 = idx1
        idx2 = idx2
        torch.cuda.set_device(device)
        chamfer_5D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        gradxyz1 = gradxyz1
        gradxyz2 = gradxyz2
        chamfer_5D.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class chamfer_5DDist(nn.Module):

    def __init__(self):
        super(chamfer_5DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_5DFunction.apply(input1, input2)


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int=None, radius: float=None, nsample: int=None, bn: bool=True, use_xyz: bool=True, pool_method='max_pool', instance_norm=False):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, pool_method=pool_method, instance_norm=instance_norm)


class PointnetFPModule(nn.Module):
    """Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool=True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) ->torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) ->torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.IntTensor(B, npoint, nsample).zero_()
        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) ->torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.FloatTensor(B, C, nfeatures, nsample)
        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)
        ctx.for_backwards = idx, N
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards
        B, C, npoint, nsample = grad_out.size()
        grad_features = Variable(torch.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):

    def __init__(self, radius: float, nsample: int, use_xyz: bool=True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None) ->Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz
        return new_features


class GroupAll(nn.Module):

    def __init__(self, use_xyz: bool=True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        return new_features


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=''):
        super().__init__()
        self.add_module(name + 'bn', batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=None, batch_norm=None, bias=True, preact=False, name='', instance_norm=False, instance_norm_func=None):
        super().__init__()
        bias = bias and not bn
        conv_unit = conv(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)
        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)
        self.add_module(name + 'conv', conv_unit)
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class Conv2d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: Tuple[int, int]=(1, 1), stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv2d, batch_norm=BatchNorm2d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm2d)


class SharedMLP(nn.Sequential):

    def __init__(self, args: List[int], *, bn: bool=False, activation=nn.ReLU(inplace=True), preact: bool=False, first: bool=False, name: str='', instance_norm: bool=False):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(name + 'layer{}'.format(i), Conv2d(args[i], args[i + 1], bn=(not first or not preact or i != 0) and bn, activation=activation if not first or not preact or i != 0 else None, preact=preact, instance_norm=instance_norm))


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class Conv1d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: int=1, stride: int=1, padding: int=0, activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv1d, batch_norm=BatchNorm1d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm1d)


class FC(nn.Sequential):

    def __init__(self, in_size: int, out_size: int, *, activation=nn.ReLU(inplace=True), bn: bool=False, init=None, preact: bool=False, name: str=''):
        super().__init__()
        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)
        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)
        self.add_module(name + 'fc', fc)
        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)


CLS_FC = [128]


FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]


MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]], [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]


NPOINTS = [4096, 1024, 256, 64]


NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]


RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]


class Pointnet2MSG(nn.Module):

    def __init__(self, input_channels=6):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(PointnetSAModuleMSG(npoint=NPOINTS[k], radii=RADIUS[k], nsamples=NSAMPLE[k], mlps=mlps, use_xyz=True, bn=True))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k]))
        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=True))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud: torch.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()
        return pred_cls


class DiceLoss(nn.Module):

    def __init__(self, ignore_target=-1):
        super().__init__()
        self.ignore_target = ignore_target

    def forward(self, input, target):
        """
        :param input: (N), logit
        :param target: (N), {0, 1}
        :return:
        """
        input = torch.sigmoid(input.view(-1))
        target = target.float().view(-1)
        mask = (target != self.ignore_target).float()
        return 1.0 - (torch.min(input, target) * mask).sum() / torch.clamp((torch.max(input, target) * mask).sum(), min=1.0)


class CubicFeatureSamplingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ptcloud, cubic_features, neighborhood_size=1):
        scale = cubic_features.size(2)
        point_features, grid_pt_indexes = cubic_feature_sampling.forward(scale, neighborhood_size, ptcloud, cubic_features)
        ctx.save_for_backward(torch.Tensor([scale]), torch.Tensor([neighborhood_size]), grid_pt_indexes)
        return point_features

    @staticmethod
    def backward(ctx, grad_point_features):
        scale, neighborhood_size, grid_pt_indexes = ctx.saved_tensors
        scale = int(scale.item())
        neighborhood_size = int(neighborhood_size.item())
        grad_point_features = grad_point_features.contiguous()
        grad_ptcloud, grad_cubic_features = cubic_feature_sampling.backward(scale, neighborhood_size, grad_point_features, grid_pt_indexes)
        return grad_ptcloud, grad_cubic_features, None


class CubicFeatureSampling(torch.nn.Module):

    def __init__(self):
        super(CubicFeatureSampling, self).__init__()

    def forward(self, ptcloud, cubic_features, neighborhood_size=1):
        h_scale = cubic_features.size(2) / 2
        ptcloud = ptcloud * h_scale + h_scale
        return CubicFeatureSamplingFunction.apply(ptcloud, cubic_features, neighborhood_size)


class emdFunction(Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        assert n == m
        assert xyz1.size()[0] == xyz2.size()[0]
        assert batchsize <= 512
        xyz1 = xyz1.contiguous().float()
        xyz2 = xyz2.contiguous().float()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)
        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()
        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()
        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None


class emdModule(nn.Module):

    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)


class expansionPenaltyFunction(Function):

    @staticmethod
    def forward(ctx, xyz, primitive_size, alpha):
        assert primitive_size <= 512
        batchsize, n, _ = xyz.size()
        assert n % primitive_size == 0
        xyz = xyz.contiguous().float()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        neighbor = torch.zeros(batchsize, n * 512, device='cuda', dtype=torch.int32).contiguous()
        cost = torch.zeros(batchsize, n * 512, device='cuda').contiguous()
        mean_mst_length = torch.zeros(batchsize, device='cuda').contiguous()
        expansion_penalty.forward(xyz, primitive_size, assignment, dist, alpha, neighbor, cost, mean_mst_length)
        ctx.save_for_backward(xyz, assignment)
        return dist, assignment, mean_mst_length / (n / primitive_size)

    @staticmethod
    def backward(ctx, grad_dist, grad_idx, grad_mml):
        xyz, assignment = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()
        grad_xyz = torch.zeros(xyz.size(), device='cuda').contiguous()
        expansion_penalty.backward(xyz, grad_xyz, grad_dist, assignment)
        return grad_xyz, None, None


class expansionPenaltyModule(nn.Module):

    def __init__(self):
        super(expansionPenaltyModule, self).__init__()

    def forward(self, input, primitive_size, alpha):
        return expansionPenaltyFunction.apply(input, primitive_size, alpha)


class GriddingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scale, ptcloud):
        grid, grid_pt_weights, grid_pt_indexes = gridding.forward(-scale, scale - 1, -scale, scale - 1, -scale, scale - 1, ptcloud)
        ctx.save_for_backward(grid_pt_weights, grid_pt_indexes)
        return grid

    @staticmethod
    def backward(ctx, grad_grid):
        grid_pt_weights, grid_pt_indexes = ctx.saved_tensors
        grad_ptcloud = gridding.backward(grid_pt_weights, grid_pt_indexes, grad_grid)
        return None, grad_ptcloud


class Gridding(torch.nn.Module):

    def __init__(self, scale=1):
        super(Gridding, self).__init__()
        self.scale = scale // 2

    def forward(self, ptcloud):
        ptcloud = ptcloud * self.scale
        _ptcloud = torch.split(ptcloud, 1, dim=0)
        grids = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            grids.append(GriddingFunction.apply(self.scale, p))
        return torch.cat(grids, dim=0).contiguous()


class GriddingReverseFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scale, grid):
        ptcloud = gridding.rev_forward(scale, grid)
        ctx.save_for_backward(torch.Tensor([scale]), grid, ptcloud)
        return ptcloud

    @staticmethod
    def backward(ctx, grad_ptcloud):
        scale, grid, ptcloud = ctx.saved_tensors
        scale = int(scale.item())
        grad_grid = gridding.rev_backward(ptcloud, grid, grad_ptcloud)
        grad_grid = grad_grid.view(-1, scale, scale, scale)
        return None, grad_grid


class GriddingReverse(torch.nn.Module):

    def __init__(self, scale=1):
        super(GriddingReverse, self).__init__()
        self.scale = scale

    def forward(self, grid):
        ptcloud = GriddingReverseFunction.apply(self.scale, grid)
        return ptcloud / self.scale * 2


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm1d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BatchNorm2d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContractExpandOperation,
     lambda: ([], {'num_input_channels': 4, 'up_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     False),
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Folding,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'step_ratio': 4}),
     lambda: ([torch.rand([4, 1024, 64]), torch.rand([4, 4])], {}),
     False),
    (GroupAll,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLPConv,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PCNEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     False),
    (PCN_encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     True),
    (PointNetRes,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (PointNetfeat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     False),
    (RandomPointSampling,
     lambda: ([], {'n_points': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (STN3d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     False),
    (Stack_conv,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_paul007pl_VRCNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

