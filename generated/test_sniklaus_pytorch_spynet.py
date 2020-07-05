import sys
_module = sys.modules[__name__]
del sys
comparison = _module
run = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import math


import numpy


arguments_strModel = 'sintel-final'


backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([tenHorizontal, tenVertical], 1).cuda()
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()


        class Preprocess(torch.nn.Module):

            def __init__(self):
                super(Preprocess, self).__init__()

            def forward(self, tenInput):
                tenBlue = (tenInput[:, 0:1, :, :] - 0.406) / 0.225
                tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
                tenRed = (tenInput[:, 2:3, :, :] - 0.485) / 0.229
                return torch.cat([tenRed, tenGreen, tenBlue], 1)


        class Basic(torch.nn.Module):

            def __init__(self, intLevel):
                super(Basic, self).__init__()
                self.netBasic = torch.nn.Sequential(torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

            def forward(self, tenInput):
                return self.netBasic(tenInput)
        self.netPreprocess = Preprocess()
        self.netBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(6)])
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(__file__.replace('run.py', 'network-' + arguments_strModel + '.pytorch')).items()})

    def forward(self, tenFirst, tenSecond):
        tenFlow = []
        tenFirst = [self.netPreprocess(tenFirst)]
        tenSecond = [self.netPreprocess(tenSecond)]
        for intLevel in range(5):
            if tenFirst[0].shape[2] > 32 or tenFirst[0].shape[3] > 32:
                tenFirst.insert(0, torch.nn.functional.avg_pool2d(input=tenFirst[0], kernel_size=2, stride=2, count_include_pad=False))
                tenSecond.insert(0, torch.nn.functional.avg_pool2d(input=tenSecond[0], kernel_size=2, stride=2, count_include_pad=False))
        tenFlow = tenFirst[0].new_zeros([tenFirst[0].shape[0], 2, int(math.floor(tenFirst[0].shape[2] / 2.0)), int(math.floor(tenFirst[0].shape[3] / 2.0))])
        for intLevel in range(len(tenFirst)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            if tenUpsampled.shape[2] != tenFirst[intLevel].shape[2]:
                tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tenUpsampled.shape[3] != tenFirst[intLevel].shape[3]:
                tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[0, 1, 0, 0], mode='replicate')
            tenFlow = self.netBasic[intLevel](torch.cat([tenFirst[intLevel], backwarp(tenInput=tenSecond[intLevel], tenFlow=tenUpsampled), tenUpsampled], 1)) + tenUpsampled
        return tenFlow

