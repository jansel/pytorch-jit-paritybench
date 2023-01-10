import sys
_module = sys.modules[__name__]
del sys
convert = _module
data_loader = _module
main = _module
model = _module
preprocess = _module
solver = _module
utils = _module

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


import numpy as np


from torch.utils import data


from torch.backends import cudnn


import torch.nn as nn


import torch.nn.functional as F


import time


from torch.utils.tensorboard import SummaryWriter


class ConditionalInstanceNormalisation(nn.Module):
    """AdaIN Block."""

    def __init__(self, dim_in, dim_c):
        super(ConditionalInstanceNormalisation, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dim_in = dim_in
        self.gamma_t = nn.Linear(dim_c, dim_in)
        self.beta_t = nn.Linear(dim_c, dim_in)

    def forward(self, x, c_trg):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-08)
        gamma = self.gamma_t(c_trg)
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta_t(c_trg)
        beta = beta.view(-1, self.dim_in, 1)
        h = (x - u) / std
        h = h * gamma + beta
        return h


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, style_num):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.cin = ConditionalInstanceNormalisation(dim_out, style_num)
        self.glu = nn.GLU(dim=1)

    def forward(self, x, c):
        x = self.conv(x)
        x = self.cin(x, c)
        x = self.glu(x)
        return x


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, num_speakers=4):
        super(Generator, self).__init__()
        self.down_sample_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 9), padding=(1, 4), bias=False), nn.GLU(dim=1))
        self.down_sample_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False), nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True), nn.GLU(dim=1))
        self.down_sample_3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False), nn.InstanceNorm2d(num_features=512, affine=True, track_running_stats=True), nn.GLU(dim=1))
        self.down_conversion = nn.Sequential(nn.Conv1d(in_channels=2304, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False), nn.InstanceNorm1d(num_features=256, affine=True))
        self.residual_1 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_2 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_3 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_4 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_5 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_6 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_7 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_8 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_9 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.up_conversion = nn.Conv1d(in_channels=256, out_channels=2304, kernel_size=1, stride=1, padding=0, bias=False)
        self.up_sample_1 = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True), nn.GLU(dim=1))
        self.up_sample_2 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(num_features=128, affine=True, track_running_stats=True), nn.GLU(dim=1))
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c):
        width_size = x.size(3)
        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)
        x = self.residual_1(x, c)
        x = self.residual_2(x, c)
        x = self.residual_3(x, c)
        x = self.residual_4(x, c)
        x = self.residual_5(x, c)
        x = self.residual_6(x, c)
        x = self.residual_7(x, c)
        x = self.residual_8(x, c)
        x = self.residual_9(x, c)
        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)
        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        x = self.out(x)
        return x


class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, num_speakers=10):
        super(Discriminator, self).__init__()
        self.num_speakers = num_speakers
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.GLU(dim=1))
        self.down_sample_1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True), nn.GLU(dim=1))
        self.down_sample_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.InstanceNorm2d(num_features=512, affine=True, track_running_stats=True), nn.GLU(dim=1))
        self.down_sample_3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.InstanceNorm2d(num_features=1024, affine=True, track_running_stats=True), nn.GLU(dim=1))
        self.down_sample_4 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False), nn.InstanceNorm2d(num_features=1024, affine=True, track_running_stats=True), nn.GLU(dim=1))
        self.fully_connected = nn.Linear(in_features=512, out_features=1)
        self.projection = nn.Linear(2 * self.num_speakers, 512)

    def forward(self, x, c, c_):
        c_onehot = torch.cat((c, c_), dim=1)
        x = self.conv_layer_1(x)
        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x_ = self.down_sample_4(x)
        h = torch.sum(x_, dim=(2, 3))
        x = self.fully_connected(x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        p = self.projection(c_onehot)
        in_prod = p * h
        x = x.view(x.size(0), -1)
        x = torch.mean(x, dim=-1) + torch.mean(in_prod, dim=-1)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConditionalInstanceNormalisation,
     lambda: ([], {'dim_in': 4, 'dim_c': 4}),
     lambda: ([torch.rand([4, 64, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'style_num': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_SamuelBroughton_StarGAN_Voice_Conversion_2(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

