import sys
_module = sys.modules[__name__]
del sys
P2PaLA = _module
data = _module
dataset = _module
imgprocess = _module
transforms = _module
evalTools = _module
metrics = _module
page2page_eval = _module
nn_models = _module
models = _module
page_xml = _module
xmlPAGE = _module
setup = _module
utils = _module
art = _module
get_inference_model = _module
img_to_page = _module
misc = _module
optparse = _module
polyapprox = _module
show_mask = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import logging


import time


import numpy as np


import torch


from torch.autograd import Variable


from torch.utils.data import DataLoader


import torch.optim as optim


from torch.utils.data import Dataset


import math


from torchvision import transforms as tv_transforms


from scipy.ndimage.interpolation import map_coordinates


from scipy.ndimage.interpolation import affine_transform


from scipy.ndimage.filters import gaussian_filter


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import init


import re


from collections import OrderedDict


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    Borrowed from: https://github.com/pytorch/pytorch/issues/3223

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError('Sum of split sizes exceeds tensor dim')
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]
    return tuple(tensor.narrow(int(dim), int(start), int(length)) for start, length in zip(splits, split_sizes))


class uSkipBlock(nn.Module):
    """
    """

    def __init__(self, input_nc, inner_nc, output_nc, inner_slave, block_type='inner', out_mode=None, i_id='0', useDO=False):
        super(uSkipBlock, self).__init__()
        self.type = block_type
        self.name = str(input_nc) + str(inner_nc) + str(output_nc) + self.type
        self.id = i_id
        self.out_mode = out_mode
        if self.type == 'R':
            e_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
            d_conv = nn.ConvTranspose2d(2 * inner_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False)
            d_non_lin = nn.ReLU(True)
            model = [e_conv] + [inner_slave] + [d_non_lin, d_conv, nn.Tanh()]
        elif self.type == 'C':
            e_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
            d_conv = nn.ConvTranspose2d(2 * inner_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False)
            d_non_lin = nn.ReLU(True)
            model = [e_conv] + [inner_slave] + [d_non_lin, d_conv]
        elif self.type == 'center':
            e_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
            e_non_lin = nn.LeakyReLU(0.2, True)
            d_conv = nn.ConvTranspose2d(inner_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False)
            d_non_lin = nn.ReLU(True)
            d_norm = nn.BatchNorm2d(output_nc)
            model = [e_non_lin, e_conv, d_non_lin, d_conv, d_norm, nn.Dropout(0.5)]
        elif self.type == 'inner':
            e_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
            e_non_lin = nn.LeakyReLU(0.2, True)
            e_norm = nn.BatchNorm2d(inner_nc)
            d_conv = nn.ConvTranspose2d(2 * inner_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False)
            d_non_lin = nn.ReLU(True)
            d_norm = nn.BatchNorm2d(output_nc)
            model = [e_non_lin, e_conv, e_norm, inner_slave, d_non_lin, d_conv, d_norm]
            if useDO:
                model = model + [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, input_x):
        """
        """
        if self.type == 'R':
            return self.model(input_x)
        elif self.type == 'C':
            if self.out_mode == 'L' or self.out_mode == 'R':
                return F.log_softmax(self.model(input_x), dim=1)
            elif self.out_mode == 'LR':
                x = self.model(input_x)
                l, r = size_splits(x, [2, x.size(1) - 2], dim=1)
                return F.log_softmax(l, dim=1), F.log_softmax(r, dim=1)
            else:
                pass
        else:
            return torch.cat([input_x, self.model(input_x)], 1)


class buildUnet(nn.Module):
    """
    doc goes here :)
    """

    def __init__(self, input_nc, output_nc, ngf=64, net_type='R', out_mode=None):
        super(buildUnet, self).__init__()
        model = uSkipBlock(ngf * 8, ngf * 8, ngf * 8, inner_slave=None, block_type='center', i_id='center')
        model = uSkipBlock(ngf * 8, ngf * 8, ngf * 8, inner_slave=model, i_id='a_1', useDO=True)
        model = uSkipBlock(ngf * 8, ngf * 8, ngf * 8, inner_slave=model, i_id='a_2', useDO=True)
        model = uSkipBlock(ngf * 8, ngf * 8, ngf * 8, inner_slave=model, i_id='a_3')
        model = uSkipBlock(ngf * 4, ngf * 8, ngf * 4, inner_slave=model, i_id='a_5')
        model = uSkipBlock(ngf * 2, ngf * 4, ngf * 2, inner_slave=model, i_id='a_6')
        model = uSkipBlock(ngf, ngf * 2, ngf, inner_slave=model, i_id='a_7')
        model = uSkipBlock(input_nc, ngf, output_nc, inner_slave=model, block_type=net_type, out_mode=out_mode, i_id='out')
        self.model = model
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += param.numel()

    def forward(self, input_x):
        """
        ;)
        """
        return self.model(input_x)


class buildDNet(nn.Module):
    """
    """

    def __init__(self, input_nc, output_nc, ngf=64, n_layers=3):
        """
        """
        super(buildDNet, self).__init__()
        model = [nn.Conv2d(input_nc + output_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False)]
        model = model + [nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_prev = 1
        for n in range(1, n_layers):
            nf_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model = model + [nn.Conv2d(ngf * nf_prev, ngf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ngf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model = model + [nn.Conv2d(ngf * nf_prev, ngf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False), nn.BatchNorm2d(ngf * nf_mult), nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * nf_mult, 1, kernel_size=4, stride=1, padding=1, bias=False), nn.Sigmoid()]
        self.model = nn.Sequential(*model)
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += param.numel()

    def forward(self, input_x):
        """
        """
        return self.model(input_x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (buildDNet,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 8, 64, 64])], {}),
     True),
    (buildUnet,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     False),
]

class Test_lquirosd_P2PaLA(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

