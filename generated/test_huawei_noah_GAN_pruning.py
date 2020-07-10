import sys
_module = sys.modules[__name__]
del sys
GA = _module
datasets = _module
finetune = _module
models = _module
models_prune = _module
search = _module
test = _module
utils = _module

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
xrange = range
wraps = functools.wraps


import torch


import numpy as np


import random


from torch.utils.data import Dataset


import torchvision.transforms as transforms


import itertools


from torch.utils.data import DataLoader


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import torch.multiprocessing as mp


from torch.utils.data import dataloader


from torch.multiprocessing import reductions


from torchvision import transforms


from torchvision.utils import save_image


import time


class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.InstanceNorm2d(in_features), nn.ReLU(inplace=True), nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):

    def __init__(self, input_nc, output_nc, compression_ratio=1, n_residual_blocks=9):
        super(Generator, self).__init__()
        first_conv_out = int(64 * compression_ratio)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, first_conv_out, 7), nn.InstanceNorm2d(first_conv_out), nn.ReLU(inplace=True)]
        in_features = int(64 * compression_ratio)
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(in_features, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256, 512, 4, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])


class Generator_Prune(nn.Module):

    def __init__(self, cfg_mask, n_residual_blocks=9):
        super(Generator_Prune, self).__init__()
        first_conv_out = int(sum(cfg_mask[0]))
        model = [nn.ReflectionPad2d(3), nn.Conv2d(3, first_conv_out, 7), nn.InstanceNorm2d(first_conv_out), nn.ReLU(inplace=True)]
        in_features = int(sum(cfg_mask[0]))
        out_features = int(sum(cfg_mask[1]))
        model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
        in_features = int(sum(cfg_mask[1]))
        out_features = int(sum(cfg_mask[2]))
        model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
        in_features = int(sum(cfg_mask[2]))
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = int(sum(cfg_mask[21]))
        model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
        in_features = out_features
        out_features = int(sum(cfg_mask[22]))
        model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
        in_features = out_features
        model += [nn.ReflectionPad2d(3), nn.Conv2d(in_features, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Generator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_huawei_noah_GAN_pruning(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

