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


arguments_strModel = 'bsds500'


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.netVggOne = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False))
        self.netVggTwo = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False))
        self.netVggThr = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False))
        self.netVggFou = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False))
        self.netVggFiv = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=False))
        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netCombine = torch.nn.Sequential(torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0), torch.nn.Sigmoid())
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(__file__.replace('run.py', 'network-' + arguments_strModel + '.pytorch')).items()})

    def forward(self, tenInput):
        tenBlue = tenInput[:, 0:1, :, :] * 255.0 - 104.00698793
        tenGreen = tenInput[:, 1:2, :, :] * 255.0 - 116.66876762
        tenRed = tenInput[:, 2:3, :, :] * 255.0 - 122.67891434
        tenInput = torch.cat([tenBlue, tenGreen, tenRed], 1)
        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)
        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)
        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        return self.netCombine(torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))

