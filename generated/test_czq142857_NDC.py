import sys
_module = sys.modules[__name__]
del sys
augment_obj = _module
get_gt_LOD = _module
setup = _module
utils = _module
simplify_obj = _module
dataset = _module
datasetpc = _module
eval_100000 = _module
eval_tri_angle = _module
eval_v_t_count = _module
gather_quantitative = _module
main = _module
model = _module
modelpc = _module

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


import numpy as np


import time


import torch


from sklearn.neighbors import KDTree


import torch.nn as nn


import torch.nn.functional as F


class resnet_block(nn.Module):

    def __init__(self, ef_dim):
        super(resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.pc_conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.pc_conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.pc_conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.pc_conv_2(output)
        output = output + input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class CNN_3d_rec7_resnet(nn.Module):

    def __init__(self, out_bool, out_float, is_undc=False):
        super(CNN_3d_rec7_resnet, self).__init__()
        self.ef_dim = 64
        self.out_bool = out_bool
        self.out_float = out_float
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_1 = resnet_block(self.ef_dim)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_2 = resnet_block(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_3 = resnet_block(self.ef_dim)
        self.res_4 = resnet_block(self.ef_dim)
        self.res_5 = resnet_block(self.ef_dim)
        self.res_6 = resnet_block(self.ef_dim)
        self.res_7 = resnet_block(self.ef_dim)
        self.res_8 = resnet_block(self.ef_dim)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        if self.out_bool:
            if is_undc:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
            else:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 1, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_1(out)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_2(out)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float


class CNN_3d_rec15_resnet(nn.Module):

    def __init__(self, out_bool, out_float, is_undc=False):
        super(CNN_3d_rec15_resnet, self).__init__()
        self.ef_dim = 64
        self.out_bool = out_bool
        self.out_float = out_float
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.res_1 = resnet_block(self.ef_dim)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.res_2 = resnet_block(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.res_3 = resnet_block(self.ef_dim)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.res_4 = resnet_block(self.ef_dim)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_5 = resnet_block(self.ef_dim)
        self.conv_5 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_6 = resnet_block(self.ef_dim)
        self.conv_6 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_7 = resnet_block(self.ef_dim)
        self.res_8 = resnet_block(self.ef_dim)
        self.res_9 = resnet_block(self.ef_dim)
        self.res_10 = resnet_block(self.ef_dim)
        self.conv_7 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_8 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        if self.out_bool:
            if is_undc:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
            else:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 1, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_1(out)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_2(out)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_3(out)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_4(out)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_5(out)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_6(out)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_7(out)
        out = self.res_8(out)
        out = self.res_9(out)
        out = self.res_10(out)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float


class CNN_3d_rec7(nn.Module):

    def __init__(self, out_bool, out_float, is_undc=False):
        super(CNN_3d_rec7, self).__init__()
        self.ef_dim = 64
        self.out_bool = out_bool
        self.out_float = out_float
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        if self.out_bool:
            if is_undc:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
            else:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 1, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float


class CNN_3d_rec15(nn.Module):

    def __init__(self, out_bool, out_float, is_undc=False):
        super(CNN_3d_rec15, self).__init__()
        self.ef_dim = 32
        self.out_bool = out_bool
        self.out_float = out_float
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_6 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_7 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_8 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        if self.out_bool:
            if is_undc:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
            else:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 1, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float


KNN_num = 8


class pc_conv_first(nn.Module):

    def __init__(self, ef_dim):
        super(pc_conv_first, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(3, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, KNN_xyz):
        output = KNN_xyz
        output = self.linear_1(output)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.linear_2(output)
        output = output.view(-1, KNN_num, self.ef_dim)
        output = torch.max(output, 1)[0]
        return output


class pc_conv(nn.Module):

    def __init__(self, ef_dim):
        super(pc_conv, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(self.ef_dim + 3, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input, KNN_idx, KNN_xyz):
        output = input
        output = output[KNN_idx]
        output = torch.cat([output, KNN_xyz], 1)
        output = self.linear_1(output)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.linear_2(output)
        output = output.view(-1, KNN_num, self.ef_dim)
        output = torch.max(output, 1)[0]
        return output


class pc_resnet_block(nn.Module):

    def __init__(self, ef_dim):
        super(pc_resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(self.ef_dim, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input):
        output = self.linear_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.linear_2(output)
        output = output + input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class resnet_block_rec3(nn.Module):

    def __init__(self, ef_dim):
        super(resnet_block_rec3, self).__init__()
        self.ef_dim = ef_dim
        self.pc_conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.pc_conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.pc_conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.pc_conv_2(output)
        output = output + input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class local_pointnet(nn.Module):

    def __init__(self, out_bool, out_float):
        super(local_pointnet, self).__init__()
        self.ef_dim = 128
        self.out_bool = out_bool
        self.out_float = out_float
        self.pc_conv_0 = pc_conv_first(self.ef_dim)
        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)
        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)
        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)
        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)
        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)
        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)
        self.pc_res_7 = pc_resnet_block(self.ef_dim)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_4 = nn.Linear(self.ef_dim, self.ef_dim)
        self.conv_5 = nn.Linear(self.ef_dim, self.ef_dim)
        if self.out_bool:
            self.pc_conv_out_bool = nn.Linear(self.ef_dim, 3)
        if self.out_float:
            self.pc_conv_out_float = nn.Linear(self.ef_dim, 3)

    def forward(self, pc_KNN_idx, pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz):
        out = pc_KNN_xyz
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_1(out)
        out = self.pc_conv_1(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_2(out)
        out = self.pc_conv_2(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_3(out)
        out = self.pc_conv_3(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_4(out)
        out = self.pc_conv_4(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_5(out)
        out = self.pc_conv_5(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_6(out)
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_7(out)
        voxel_xyz_int_max = torch.max(voxel_xyz_int, 0)[0]
        voxel_xyz_int_min = torch.min(voxel_xyz_int, 0)[0]
        voxel_xyz_int_size = voxel_xyz_int_max - voxel_xyz_int_min + 1
        voxel_xyz_int = voxel_xyz_int - voxel_xyz_int_min.view(1, -1)
        tmp_grid = torch.zeros(voxel_xyz_int_size[0], voxel_xyz_int_size[1], voxel_xyz_int_size[2], self.ef_dim, device=out.device)
        tmp_grid[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]] = out
        tmp_grid = tmp_grid.permute(3, 0, 1, 2).unsqueeze(0)
        out = tmp_grid
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_3(out)
        out = out.squeeze(0).permute(1, 2, 3, 0)
        out = out[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]]
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        if self.out_bool and self.out_float:
            out_bool = self.pc_conv_out_bool(out)
            out_float = self.pc_conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.pc_conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.pc_conv_out_float(out)
            return out_float


class local_pointnet_larger(nn.Module):

    def __init__(self, out_bool, out_float):
        super(local_pointnet_larger, self).__init__()
        self.ef_dim = 128
        self.out_bool = out_bool
        self.out_float = out_float
        self.pc_conv_0 = pc_conv_first(self.ef_dim)
        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)
        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)
        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)
        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)
        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)
        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)
        self.pc_res_7 = pc_resnet_block(self.ef_dim)
        self.res_1 = resnet_block_rec3(self.ef_dim)
        self.res_2 = resnet_block_rec3(self.ef_dim)
        self.res_3 = resnet_block_rec3(self.ef_dim)
        self.res_4 = resnet_block_rec3(self.ef_dim)
        self.res_5 = resnet_block_rec3(self.ef_dim)
        self.res_6 = resnet_block_rec3(self.ef_dim)
        self.res_7 = resnet_block_rec3(self.ef_dim)
        self.res_8 = resnet_block_rec3(self.ef_dim)
        self.linear_1 = nn.Linear(self.ef_dim, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)
        if self.out_bool:
            self.linear_bool = nn.Linear(self.ef_dim, 3)
        if self.out_float:
            self.linear_float = nn.Linear(self.ef_dim, 3)

    def forward(self, pc_KNN_idx, pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz):
        out = pc_KNN_xyz
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_1(out)
        out = self.pc_conv_1(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_2(out)
        out = self.pc_conv_2(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_3(out)
        out = self.pc_conv_3(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_4(out)
        out = self.pc_conv_4(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_5(out)
        out = self.pc_conv_5(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_6(out)
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.pc_res_7(out)
        voxel_xyz_int_max = torch.max(voxel_xyz_int, 0)[0]
        voxel_xyz_int_min = torch.min(voxel_xyz_int, 0)[0]
        voxel_xyz_int_size = voxel_xyz_int_max - voxel_xyz_int_min + 1
        voxel_xyz_int = voxel_xyz_int - voxel_xyz_int_min.view(1, -1)
        tmp_grid = torch.zeros(voxel_xyz_int_size[0], voxel_xyz_int_size[1], voxel_xyz_int_size[2], self.ef_dim, device=out.device)
        tmp_grid[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]] = out
        tmp_grid = tmp_grid.permute(3, 0, 1, 2).unsqueeze(0)
        out = tmp_grid
        out = self.res_1(out)
        out = self.res_2(out)
        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)
        out = out.squeeze(0).permute(1, 2, 3, 0)
        out = out[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]]
        out = self.linear_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.linear_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        if self.out_bool and self.out_float:
            out_bool = self.linear_bool(out)
            out_float = self.linear_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.linear_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.linear_float(out)
            return out_float


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CNN_3d_rec15,
     lambda: ([], {'out_bool': 4, 'out_float': 4}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (CNN_3d_rec15_resnet,
     lambda: ([], {'out_bool': 4, 'out_float': 4}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (CNN_3d_rec7,
     lambda: ([], {'out_bool': 4, 'out_float': 4}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (CNN_3d_rec7_resnet,
     lambda: ([], {'out_bool': 4, 'out_float': 4}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (pc_resnet_block,
     lambda: ([], {'ef_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (resnet_block,
     lambda: ([], {'ef_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (resnet_block_rec3,
     lambda: ([], {'ef_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_czq142857_NDC(_paritybench_base):
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

