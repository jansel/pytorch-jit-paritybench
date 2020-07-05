import sys
_module = sys.modules[__name__]
del sys
octconv = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


class OctConv(nn.Module):

    def __init__(self, ch_in, ch_out, kernel_size, stride=1, alphas=(0.5, 0.5)):
        super(OctConv, self).__init__()
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_in <= 1, 'Alphas must be in interval [0, 1]'
        self.ch_in_hf = int((1 - self.alpha_in) * ch_in)
        self.ch_in_lf = ch_in - self.ch_in_hf
        self.ch_out_hf = int((1 - self.alpha_out) * ch_out)
        self.ch_out_lf = ch_out - self.ch_out_hf
        self.wHtoH = nn.Parameter(torch.randn(self.ch_out_hf, self.ch_in_hf, kernel_size, kernel_size))
        self.wHtoL = nn.Parameter(torch.randn(self.ch_out_lf, self.ch_in_hf, kernel_size, kernel_size))
        self.wLtoH = nn.Parameter(torch.randn(self.ch_out_hf, self.ch_in_lf, kernel_size, kernel_size))
        self.wLtoL = nn.Parameter(torch.randn(self.ch_out_lf, self.ch_in_lf, kernel_size, kernel_size))
        self.padding = (kernel_size - stride) // 2

    def forward(self, input):
        if self.alpha_in == 0:
            hf_input = input
            lf_input = torch.Tensor([]).reshape(1, 0)
        else:
            fmap_size = input.shape[-1]
            hf_input = input[:, :self.ch_in_hf * 4, (...)].reshape(-1, self.ch_in_hf, fmap_size * 2, fmap_size * 2)
            lf_input = input[:, self.ch_in_hf * 4:, (...)]
        HtoH = HtoL = LtoL = LtoH = 0.0
        if self.alpha_in < 1:
            if self.ch_out_hf > 0:
                HtoH = F.conv2d(hf_input, self.wHtoH, padding=self.padding)
            if self.ch_out_lf > 0:
                HtoL = F.conv2d(F.avg_pool2d(hf_input, 2), self.wHtoL, padding=self.padding)
        if self.alpha_in > 0:
            if self.ch_out_hf > 0:
                LtoH = F.interpolate(F.conv2d(lf_input, self.wLtoH, padding=self.padding), scale_factor=2, mode='nearest')
            if self.ch_out_lf > 0:
                LtoL = F.conv2d(lf_input, self.wLtoL, padding=self.padding)
        hf_output = HtoH + LtoH
        lf_output = LtoL + HtoL
        if 0 < self.alpha_out < 1:
            fmap_size = hf_output.shape[-1] // 2
            hf_output = hf_output.reshape(-1, 4 * self.ch_out_hf, fmap_size, fmap_size)
            output = torch.cat([hf_output, lf_output], dim=1)
        elif np.isclose(self.alpha_out, 1.0, atol=1e-08):
            output = lf_output
        elif np.isclose(self.alpha_out, 0.0, atol=1e-08):
            output = hf_output
        return output

