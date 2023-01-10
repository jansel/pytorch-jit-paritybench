import sys
_module = sys.modules[__name__]
del sys
CPNet_test = _module
DAVIS_dataset = _module
CPNet_model = _module
model_module = _module

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


from torch.autograd import Variable


from torch.utils import data


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import torch.utils.model_zoo as model_zoo


from torchvision import models


import matplotlib.pyplot as plt


import numpy as np


import math


import time


import copy


import random


import matplotlib


import collections


import torchvision


from torchvision import transforms


class Conv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, D=1, activation=nn.ReLU()):
        super(Conv2d, self).__init__()
        if activation:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=D), activation)
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=D))

    def forward(self, x):
        x = self.conv(x)
        return x


def init_He(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class A_Encoder(nn.Module):

    def __init__(self):
        super(A_Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU())
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv23 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv34 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())
        self.conv4a = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv4b = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = F.upsample(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        x = self.conv34(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        return x


class A_Regressor(nn.Module):

    def __init__(self):
        super(A_Regressor, self).__init__()
        self.conv45 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())
        self.conv5a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv56 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())
        self.conv6a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv6b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        init_He(self)
        self.fc = nn.Linear(512, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.conv45(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv56(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(-1, x.shape[1])
        theta = self.fc(x)
        theta = theta.view(-1, 2, 3)
        return theta


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU())
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv23 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.value3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=None)
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        v = self.value3(x)
        return v


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_1 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_2 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.convA4_1 = Conv2d(257, 257, kernel_size=3, stride=1, padding=2, D=2, activation=nn.ReLU())
        self.convA4_2 = Conv2d(257, 257, kernel_size=3, stride=1, padding=4, D=4, activation=nn.ReLU())
        self.convA4_3 = Conv2d(257, 257, kernel_size=3, stride=1, padding=8, D=8, activation=nn.ReLU())
        self.convA4_4 = Conv2d(257, 257, kernel_size=3, stride=1, padding=16, D=16, activation=nn.ReLU())
        self.conv3c = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv3b = Conv2d(257, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv3a = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv32 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv21 = Conv2d(64, 3, kernel_size=5, stride=1, padding=2, activation=None)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = self.conv4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.convA4_1(x)
        x = self.convA4_2(x)
        x = self.convA4_3(x)
        x = self.convA4_4(x)
        x = self.conv3c(x)
        x = self.conv3b(x)
        x = self.conv3a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv32(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv21(x)
        p = x * self.std + self.mean
        return p


class CM_Module(nn.Module):

    def __init__(self):
        super(CM_Module, self).__init__()

    def masked_softmax(self, vec, mask, dim):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec - max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros = masked_sums < 0.0001
        masked_sums += zeros.float()
        return masked_exps / masked_sums

    def forward(self, values, tvmap, rvmaps):
        B, C, T, H, W = values.size()
        t_feat = values[:, :, 0]
        r_feats = values[:, :, 1:]
        B, Cv, T, H, W = r_feats.size()
        gs_, vmap_ = [], []
        tvmap_t = (F.upsample(tvmap, size=(H, W), mode='bilinear', align_corners=False) > 0.5).float()
        for r in range(T):
            rvmap_t = (F.upsample(rvmaps[:, :, r], size=(H, W), mode='bilinear', align_corners=False) > 0.5).float()
            vmap = tvmap_t * rvmap_t
            gs = (vmap * t_feat * r_feats[:, :, r]).sum(-1).sum(-1).sum(-1)
            v_sum = vmap[:, 0].sum(-1).sum(-1)
            zeros = v_sum < 0.0001
            gs[zeros] = 0
            v_sum += zeros.float()
            gs = gs / v_sum / C
            gs = torch.ones(t_feat.shape).float() * gs.view(B, 1, 1, 1)
            gs_.append(gs)
            vmap_.append(rvmap_t)
        gss = torch.stack(gs_, dim=2)
        vmaps = torch.stack(vmap_, dim=2)
        c_match = self.masked_softmax(gss, vmaps, dim=2)
        c_out = torch.sum(r_feats * c_match, dim=2)
        c_mask = c_match * vmaps
        c_mask = torch.sum(c_mask, 2)
        c_mask = 1.0 - torch.mean(c_mask, 1, keepdim=True)
        return torch.cat([t_feat, c_out, c_mask], dim=1), c_mask


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = int(lw), int(uw), int(lh), int(uh)
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array


class CPNet(nn.Module):

    def __init__(self, mode='Train'):
        super(CPNet, self).__init__()
        self.A_Encoder = A_Encoder()
        self.A_Regressor = A_Regressor()
        self.Encoder = Encoder()
        self.CM_Module = CM_Module()
        self.Decoder = Decoder()
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('mean3d', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))

    def encoding(self, frames, holes):
        batch_size, _, num_frames, height, width = frames.size()
        (frames, holes), pad = pad_divide_by([frames, holes], 8, (frames.size()[3], frames.size()[4]))
        feat_ = []
        for t in range(num_frames):
            feat = self.A_Encoder(frames[:, :, t], holes[:, :, t])
            feat_.append(feat)
        feats = torch.stack(feat_, dim=2)
        return feats

    def inpainting(self, rfeats, rframes, rholes, frame, hole, gt):
        batch_size, _, height, width = frame.size()
        num_r = rfeats.size()[2]
        (rframes, rholes, frame, hole, gt), pad = pad_divide_by([rframes, rholes, frame, hole, gt], 8, (height, width))
        tfeat = self.A_Encoder(frame, hole)
        c_feat_ = [self.Encoder(frame, hole)]
        L_align = torch.zeros_like(frame)
        aligned_r_ = []
        rvmap_ = []
        for r in range(num_r):
            theta_rt = self.A_Regressor(tfeat, rfeats[:, :, r])
            grid_rt = F.affine_grid(theta_rt, frame.size())
            aligned_r = F.grid_sample(rframes[:, :, r], grid_rt)
            aligned_v = F.grid_sample(1 - rholes[:, :, r], grid_rt)
            aligned_v = (aligned_v > 0.5).float()
            aligned_r_.append(aligned_r)
            trvmap = (1 - hole) * aligned_v
            c_feat_.append(self.Encoder(aligned_r, aligned_v))
            rvmap_.append(aligned_v)
        aligned_rs = torch.stack(aligned_r_, 2)
        c_feats = torch.stack(c_feat_, dim=2)
        rvmaps = torch.stack(rvmap_, dim=2)
        p_in, c_mask = self.CM_Module(c_feats, 1 - hole, rvmaps)
        pred = self.Decoder(p_in)
        _, _, _, H, W = aligned_rs.shape
        c_mask = F.upsample(c_mask, size=(H, W), mode='bilinear', align_corners=False).detach()
        comp = pred * hole + gt * (1.0 - hole)
        if pad[2] + pad[3] > 0:
            comp = comp[:, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            comp = comp[:, :, :, pad[0]:-pad[1]]
        comp = torch.clamp(comp, 0, 1)
        return comp

    def forward(self, *args, **kwargs):
        if len(args) == 2:
            return self.encoding(*args)
        else:
            return self.inpainting(*args, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (A_Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 257, 64, 64])], {}),
     False),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {}),
     True),
]

class Test_shleecs_Copy_and_Paste_Networks_for_Deep_Video_Inpainting(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

