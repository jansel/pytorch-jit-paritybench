import sys
_module = sys.modules[__name__]
del sys
config = _module
ModelNet40 = _module
data = _module
preprocess = _module
MeshNet = _module
models = _module
layers = _module
test = _module
train = _module
utils = _module
retrival = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import torch


import torch.utils.data as data


import torch.nn as nn


from torch.nn.parameter import Parameter


from torch.autograd import Variable


import copy


import torch.optim as optim


class MeshConvolution(nn.Module):

    def __init__(self, cfg, spatial_in_channel, structural_in_channel, spatial_out_channel, structural_out_channel):
        super(MeshConvolution, self).__init__()
        self.spatial_in_channel = spatial_in_channel
        self.structural_in_channel = structural_in_channel
        self.spatial_out_channel = spatial_out_channel
        self.structural_out_channel = structural_out_channel
        assert cfg['aggregation_method'] in ['Concat', 'Max', 'Average']
        self.aggregation_method = cfg['aggregation_method']
        self.combination_mlp = nn.Sequential(nn.Conv1d(self.spatial_in_channel + self.structural_in_channel, self.spatial_out_channel, 1), nn.BatchNorm1d(self.spatial_out_channel), nn.ReLU())
        if self.aggregation_method == 'Concat':
            self.concat_mlp = nn.Sequential(nn.Conv2d(self.structural_in_channel * 2, self.structural_in_channel, 1), nn.BatchNorm2d(self.structural_in_channel), nn.ReLU())
        self.aggregation_mlp = nn.Sequential(nn.Conv1d(self.structural_in_channel, self.structural_out_channel, 1), nn.BatchNorm1d(self.structural_out_channel), nn.ReLU())

    def forward(self, spatial_fea, structural_fea, neighbor_index):
        b, _, n = spatial_fea.size()
        spatial_fea = self.combination_mlp(torch.cat([spatial_fea, structural_fea], 1))
        if self.aggregation_method == 'Concat':
            structural_fea = torch.cat([structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2, neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel, -1, -1))], 1)
            structural_fea = self.concat_mlp(structural_fea)
            structural_fea = torch.max(structural_fea, 3)[0]
        elif self.aggregation_method == 'Max':
            structural_fea = torch.cat([structural_fea.unsqueeze(3), torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2, neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel, -1, -1))], 3)
            structural_fea = torch.max(structural_fea, 3)[0]
        elif self.aggregation_method == 'Average':
            structural_fea = torch.cat([structural_fea.unsqueeze(3), torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2, neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel, -1, -1))], 3)
            structural_fea = torch.sum(structural_fea, dim=3) / 4
        structural_fea = self.aggregation_mlp(structural_fea)
        return spatial_fea, structural_fea


class SpatialDescriptor(nn.Module):

    def __init__(self):
        super(SpatialDescriptor, self).__init__()
        self.spatial_mlp = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())

    def forward(self, centers):
        return self.spatial_mlp(centers)


class FaceKernelCorrelation(nn.Module):

    def __init__(self, num_kernel=64, sigma=0.2):
        super(FaceKernelCorrelation, self).__init__()
        self.num_kernel = num_kernel
        self.sigma = sigma
        self.weight_alpha = Parameter(torch.rand(1, num_kernel, 4) * np.pi)
        self.weight_beta = Parameter(torch.rand(1, num_kernel, 4) * 2 * np.pi)
        self.bn = nn.BatchNorm1d(num_kernel)
        self.relu = nn.ReLU()

    def forward(self, normals, neighbor_index):
        b, _, n = normals.size()
        center = normals.unsqueeze(2).expand(-1, -1, self.num_kernel, -1).unsqueeze(4)
        neighbor = torch.gather(normals.unsqueeze(3).expand(-1, -1, -1, 3), 2, neighbor_index.unsqueeze(1).expand(-1, 3, -1, -1))
        neighbor = neighbor.unsqueeze(2).expand(-1, -1, self.num_kernel, -1, -1)
        fea = torch.cat([center, neighbor], 4)
        fea = fea.unsqueeze(5).expand(-1, -1, -1, -1, -1, 4)
        weight = torch.cat([torch.sin(self.weight_alpha) * torch.cos(self.weight_beta), torch.sin(self.weight_alpha) * torch.sin(self.weight_beta), torch.cos(self.weight_alpha)], 0)
        weight = weight.unsqueeze(0).expand(b, -1, -1, -1)
        weight = weight.unsqueeze(3).expand(-1, -1, -1, n, -1)
        weight = weight.unsqueeze(4).expand(-1, -1, -1, -1, 4, -1)
        dist = torch.sum((fea - weight) ** 2, 1)
        fea = torch.sum(torch.sum(np.e ** (dist / (-2 * self.sigma ** 2)), 4), 3) / 16
        return self.relu(self.bn(fea))


class FaceRotateConvolution(nn.Module):

    def __init__(self):
        super(FaceRotateConvolution, self).__init__()
        self.rotate_mlp = nn.Sequential(nn.Conv1d(6, 32, 1), nn.BatchNorm1d(32), nn.ReLU(), nn.Conv1d(32, 32, 1), nn.BatchNorm1d(32), nn.ReLU())
        self.fusion_mlp = nn.Sequential(nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())

    def forward(self, corners):
        fea = (self.rotate_mlp(corners[:, :6]) + self.rotate_mlp(corners[:, 3:9]) + self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))) / 3
        return self.fusion_mlp(fea)


class StructuralDescriptor(nn.Module):

    def __init__(self, cfg):
        super(StructuralDescriptor, self).__init__()
        self.FRC = FaceRotateConvolution()
        self.FKC = FaceKernelCorrelation(cfg['num_kernel'], cfg['sigma'])
        self.structural_mlp = nn.Sequential(nn.Conv1d(64 + 3 + cfg['num_kernel'], 131, 1), nn.BatchNorm1d(131), nn.ReLU(), nn.Conv1d(131, 131, 1), nn.BatchNorm1d(131), nn.ReLU())

    def forward(self, corners, normals, neighbor_index):
        structural_fea1 = self.FRC(corners)
        structural_fea2 = self.FKC(normals, neighbor_index)
        return self.structural_mlp(torch.cat([structural_fea1, structural_fea2, normals], 1))


class MeshNet(nn.Module):

    def __init__(self, cfg, require_fea=False):
        super(MeshNet, self).__init__()
        self.require_fea = require_fea
        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(cfg['structural_descriptor'])
        self.mesh_conv1 = MeshConvolution(cfg['mesh_convolution'], 64, 131, 256, 256)
        self.mesh_conv2 = MeshConvolution(cfg['mesh_convolution'], 256, 256, 512, 512)
        self.fusion_mlp = nn.Sequential(nn.Conv1d(1024, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.concat_mlp = nn.Sequential(nn.Conv1d(1792, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 40))

    def forward(self, centers, corners, normals, neighbor_index):
        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(corners, normals, neighbor_index)
        spatial_fea1, structural_fea1 = self.mesh_conv1(spatial_fea0, structural_fea0, neighbor_index)
        spatial_fea2, structural_fea2 = self.mesh_conv2(spatial_fea1, structural_fea1, neighbor_index)
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))
        fea = self.concat_mlp(torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1))
        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)
        fea = self.classifier[:-1](fea)
        cls = self.classifier[-1:](fea)
        if self.require_fea:
            return cls, fea / torch.norm(fea)
        else:
            return cls


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FaceRotateConvolution,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 9, 64])], {}),
     True),
    (SpatialDescriptor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     True),
]

class Test_iMoonLab_MeshNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

