import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
benchmark_aflw = _module
benchmark_aflw2000 = _module
convert_to_onnx = _module
mobilenet_v1 = _module
convert_imgs_to_video = _module
rendering = _module
rendering_demo = _module
main = _module
mobilenet_v1 = _module
speed_cpu = _module
train = _module
utils = _module
cv_plot = _module
cython = _module
setup = _module
ddfa = _module
estimate_pose = _module
inference = _module
io = _module
lighting = _module
paf = _module
params = _module
render = _module
vdc_loss = _module
video_demo = _module
visualize = _module
wpdc_loss = _module

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


import torch.utils.data as data


import torchvision.transforms as transforms


import torch.backends.cudnn as cudnn


import time


import numpy as np


import math


import scipy.io as sio


import logging


from torch.utils.data import DataLoader


from math import sqrt


class DepthWiseBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes, bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)
        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)
        return out


class MobileNet(nn.Module):

    def __init__(self, widen_factor=1.0, num_classes=1000, prelu=False, input_channel=3):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNet, self).__init__()
        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)
        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)
        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)
        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)
        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _parse_param_batch(param):
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(N, 3, 1)
    alpha_shp = param[:, 12:52].view(N, -1, 1)
    alpha_exp = param[:, 52:].view(N, -1, 1)
    return p, offset, alpha_shp, alpha_exp


def _tensor_to_cuda(x):
    if x.is_cuda:
        return x
    else:
        return x


_numpy_to_cuda = lambda x: _tensor_to_cuda(torch.from_numpy(x))


_to_tensor = _numpy_to_cuda


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


d = 'test.configs'


class VDCLoss(nn.Module):

    def __init__(self, opt_style='all'):
        super(VDCLoss, self).__init__()
        self.u = _to_tensor(u)
        self.param_mean = _to_tensor(param_mean)
        self.param_std = _to_tensor(param_std)
        self.w_shp = _to_tensor(w_shp)
        self.w_exp = _to_tensor(w_exp)
        self.keypoints = _to_tensor(keypoints)
        self.u_base = self.u[self.keypoints]
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
        self.w_shp_length = self.w_shp.shape[0] // 3
        self.opt_style = opt_style

    def reconstruct_and_parse(self, input, target):
        param = input * self.param_std + self.param_mean
        param_gt = target * self.param_std + self.param_mean
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)
        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def forward_all(self, input, target):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) = self.reconstruct_and_parse(input, target)
        N = input.shape[0]
        offset[:, -1] = offsetg[:, -1]
        gt_vertex = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg).view(N, -1, 3).permute(0, 2, 1) + offsetg
        vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).view(N, -1, 3).permute(0, 2, 1) + offset
        diff = (gt_vertex - vertex) ** 2
        loss = torch.mean(diff)
        return loss

    def forward_resample(self, input, target, resample_num=132):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) = self.reconstruct_and_parse(input, target)
        index = torch.randperm(self.w_shp_length)[:resample_num].reshape(-1, 1)
        keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1)
        keypoints_mix = torch.cat((self.keypoints, keypoints_resample))
        w_shp_base = self.w_shp[keypoints_mix]
        u_base = self.u[keypoints_mix]
        w_exp_base = self.w_exp[keypoints_mix]
        offset[:, -1] = offsetg[:, -1]
        N = input.shape[0]
        gt_vertex = pg @ (u_base + w_shp_base @ alpha_shpg + w_exp_base @ alpha_expg).view(N, -1, 3).permute(0, 2, 1) + offsetg
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).view(N, -1, 3).permute(0, 2, 1) + offset
        diff = (gt_vertex - vertex) ** 2
        loss = torch.mean(diff)
        return loss

    def forward(self, input, target):
        if self.opt_style == 'all':
            return self.forward_all(input, target)
        elif self.opt_style == 'resample':
            return self.forward_resample(input, target)
        else:
            raise Exception(f'Unknown opt style: f{opt_style}')


class WPDCLoss(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132):
        super(WPDCLoss, self).__init__()
        self.opt_style = opt_style
        self.param_mean = _to_tensor(param_mean)
        self.param_std = _to_tensor(param_std)
        self.u = _to_tensor(u)
        self.w_shp = _to_tensor(w_shp)
        self.w_exp = _to_tensor(w_exp)
        self.w_norm = _to_tensor(w_norm)
        self.w_shp_length = self.w_shp.shape[0] // 3
        self.keypoints = _to_tensor(keypoints)
        self.resample_num = resample_num

    def reconstruct_and_parse(self, input, target):
        param = input * self.param_std + self.param_mean
        param_gt = target * self.param_std + self.param_mean
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)
        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def _calc_weights_resample(self, input_, target_):
        if self.resample_num <= 0:
            keypoints_mix = self.keypoints
        else:
            index = torch.randperm(self.w_shp_length)[:self.resample_num].reshape(-1, 1)
            keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1)
            keypoints_mix = torch.cat((self.keypoints, keypoints_resample))
        w_shp_base = self.w_shp[keypoints_mix]
        u_base = self.u[keypoints_mix]
        w_exp_base = self.w_exp[keypoints_mix]
        input = torch.tensor(input_.data.clone(), requires_grad=False)
        target = torch.tensor(target_.data.clone(), requires_grad=False)
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) = self.reconstruct_and_parse(input, target)
        input = self.param_std * input + self.param_mean
        target = self.param_std * target + self.param_mean
        N = input.shape[0]
        offset[:, -1] = offsetg[:, -1]
        weights = torch.zeros_like(input, dtype=torch.float)
        tmpv = (u_base + w_shp_base @ alpha_shpg + w_exp_base @ alpha_expg).view(N, -1, 3).permute(0, 2, 1)
        tmpv_norm = torch.norm(tmpv, dim=2)
        offset_norm = sqrt(w_shp_base.shape[0] // 3)
        param_diff_pose = torch.abs(input[:, :11] - target[:, :11])
        for ind in range(11):
            if ind in [0, 4, 8]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 0]
            elif ind in [1, 5, 9]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 1]
            elif ind in [2, 6, 10]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 2]
            else:
                weights[:, ind] = param_diff_pose[:, ind] * offset_norm
        magic_number = 0.00057339936
        param_diff_shape_exp = torch.abs(input[:, 12:] - target[:, 12:])
        w = torch.cat((w_shp_base, w_exp_base), dim=1)
        w_norm = torch.norm(w, dim=0)
        weights[:, 12:] = magic_number * param_diff_shape_exp * w_norm
        eps = 1e-06
        weights[:, :11] += eps
        weights[:, 12:] += eps
        maxes, _ = weights.max(dim=1)
        maxes = maxes.view(-1, 1)
        weights /= maxes
        weights[:, 11] = 0
        return weights

    def forward(self, input, target, weights_scale=10):
        if self.opt_style == 'resample':
            weights = self._calc_weights_resample(input, target)
            loss = weights * (input - target) ** 2
            return loss.mean()
        else:
            raise Exception(f'Unknown opt style: {self.opt_style}')


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DepthWiseBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_cleardusk_3DDFA(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

