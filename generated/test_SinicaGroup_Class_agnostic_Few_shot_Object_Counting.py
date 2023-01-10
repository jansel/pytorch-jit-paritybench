import sys
_module = sys.modules[__name__]
del sys
coco = _module
crop = _module
main = _module
CFOCNet = _module
JDimPool = _module
layers = _module
loss = _module
resblocks = _module
utils = _module
runner = _module

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


from torch.utils.data import Dataset


import numpy as np


from torchvision import transforms


import scipy.ndimage


import logging


import copy


import torch.nn as nn


from torchvision.models.resnet import resnet50


from torch.utils.data import DataLoader


from torch import optim


import matplotlib.pyplot as plt


from torchvision.utils import save_image


from torchvision.utils import make_grid


class JDimPool(nn.MaxPool1d):

    def forward(self, input):
        b, n, c, w, h = input.size()
        input = input.view(b, n, c * w * h).permute(0, 2, 1)
        pooled = nn.functional.max_pool1d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(b, c, w, h)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention


def make_resblocks():
    net = resnet50(pretrained=False)
    layer0_name = ['conv1', 'bn1', 'relu']
    layer1_name = ['maxpool', 'layer1']
    layer2_name = ['layer2']
    layer3_name = ['layer3']
    layer0 = nn.Sequential()
    layer1 = nn.Sequential()
    layer2 = nn.Sequential()
    layer3 = nn.Sequential()
    for n, c in net.named_children():
        if n in layer0_name:
            layer0.add_module(n, c)
        elif n in layer1_name:
            layer1.add_module(n, c)
        elif n in layer2_name:
            layer2.add_module(n, c)
        elif n in layer3_name:
            layer3 = c
        else:
            break
    return layer0, layer1, layer2, layer3


class CFOCNet(nn.Module):

    def __init__(self):
        super(CFOCNet, self).__init__()
        self.relu = nn.ReLU()
        self.res_q0, self.res_q1, self.res_q2, self.res_q3 = make_resblocks()
        self.res_r0, self.res_r1, self.res_r2, self.res_r3 = make_resblocks()
        self.sa_q1 = Self_Attn(256, nn.ReLU())
        self.sa_q2 = Self_Attn(512, nn.ReLU())
        self.sa_q3 = Self_Attn(1024, nn.ReLU())
        self.j_maxpool = JDimPool(5, 1)
        self.maxpool_r1 = nn.MaxPool2d(4, stride=4, padding=0)
        self.maxpool_r2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.match_query_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.match_query_conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.match_query_conv3 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.match_reference_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.match_reference_conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.match_reference_conv3 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.sum_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.sum_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.sum_conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.transpose_convolution = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, queury, references):
        q0 = self.res_q0(queury)
        q1 = self.res_q1(q0)
        q2 = self.res_q2(q1)
        q3 = self.res_q3(q2)
        org_ref_size = references.size()
        references = references.view(-1, org_ref_size[-3], org_ref_size[-2], org_ref_size[-1])
        r0 = self.res_r0(references)
        r1 = self.res_r1(r0)
        r2 = self.res_r2(r1)
        r3 = self.res_r3(r2)
        r1_size = r1.size()
        r2_size = r2.size()
        r3_size = r3.size()
        r1 = r1.view(org_ref_size[0], org_ref_size[1], r1_size[-3], r1_size[-2], r1_size[-1])
        r2 = r2.view(org_ref_size[0], org_ref_size[1], r2_size[-3], r2_size[-2], r2_size[-1])
        r3 = r3.view(org_ref_size[0], org_ref_size[1], r3_size[-3], r3_size[-2], r3_size[-1])
        r1 = self.j_maxpool(r1)
        r2 = self.j_maxpool(r2)
        r3 = self.j_maxpool(r3)
        input1, _ = self.sa_q1(q1)
        input2, _ = self.sa_q2(q2)
        input3, _ = self.sa_q3(q3)
        input1 = self.match_query_conv1(input1)
        input2 = self.match_query_conv2(input2)
        input3 = self.match_query_conv3(input3)
        kernel1 = self.maxpool_r1(r1)
        kernel2 = self.maxpool_r2(r2)
        kernel3 = r3
        kernel1 = self.match_reference_conv1(kernel1)
        kernel2 = self.match_reference_conv2(kernel2)
        kernel3 = self.match_reference_conv3(kernel3)
        M1 = []
        M2 = []
        M3 = []
        for i in range(kernel1.size(0)):
            tmp_m1 = nn.functional.conv2d(input1[i:i + 1], kernel1[i:i + 1], padding=1, stride=2)
            tmp_m2 = nn.functional.conv2d(input2[i:i + 1], kernel2[i:i + 1], padding=1, stride=2)
            tmp_m3 = nn.functional.conv2d(input3[i:i + 1], kernel3[i:i + 1], padding=1, stride=2)
            M1.append(tmp_m1)
            M2.append(tmp_m2)
            M3.append(tmp_m3)
        M1 = torch.cat(M1, 0)
        M2 = torch.cat(M2, 0)
        M3 = torch.cat(M3, 0)
        M2 = nn.functional.interpolate(M2, scale_factor=2)
        M3 = nn.functional.interpolate(M3, scale_factor=4)
        S1 = self.sum_conv1(M1)
        S2 = self.sum_conv2(M2)
        S3 = self.sum_conv3(M3)
        S1_W = []
        S2_W = []
        S3_W = []
        for k in range(S1.shape[0]):
            S1_W.append(torch.sum(S1[k]))
            S2_W.append(torch.sum(S2[k]))
            S3_W.append(torch.sum(S3[k]))
        S1_W = torch.stack(S1_W)
        S2_W = torch.stack(S2_W)
        S3_W = torch.stack(S3_W)
        W = []
        for k in range(S1_W.shape[0]):
            W.append(self.softmax(torch.tensor([S1_W[k], S2_W[k], S3_W[k]])))
        FS = []
        for k in range(M1.shape[0]):
            FS.append(W[k][0] * M1[k] + W[k][1] * M2[k] + W[k][2] * M3[k])
        FS = torch.stack(FS)
        FS = self.transpose_convolution(FS)
        FS = nn.functional.interpolate(FS, scale_factor=4, mode='bilinear', align_corners=True)
        return FS


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (JDimPool,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_SinicaGroup_Class_agnostic_Few_shot_Object_Counting(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

