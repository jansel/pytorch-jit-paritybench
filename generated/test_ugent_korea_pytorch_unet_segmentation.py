import sys
_module = sys.modules[__name__]
del sys
accuracy = _module
advanced_model = _module
dataset = _module
main = _module
mean_std = _module
modules = _module
post_processing = _module
pre_processing = _module
result_visualization = _module
save_history = _module
simple_model = _module

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


import torch


import torch.nn as nn


from torch.autograd import Variable


from torch.nn.functional import sigmoid


from random import randint


from torch.utils.data.dataset import Dataset


class Double_conv(nn.Module):
    """(conv => ReLU) * 2 => MaxPool2d"""

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=0, stride=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_down(nn.Module):
    """(conv => ReLU) * 2 => MaxPool2d"""

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        pool_x = self.pool(x)
        return pool_x, x


def extract_img(size, in_tensor):
    """
    Args:
        size(int) : size of cut
        in_tensor(tensor) : tensor to be cut
    """
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1 - size) / 2):int((dim1 + size) / 2), int((dim2 - size) / 2):int((dim2 + size) / 2)]
    return in_tensor


class Conv_up(nn.Module):
    """(conv => ReLU) * 2 => MaxPool2d"""

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = Double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1_dim = x1.size()[2]
        x2 = extract_img(x1_dim, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


class CleanU_Net(nn.Module):

    def __init__(self):
        super(CleanU_Net, self).__init__()
        self.conv1_block = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True))
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_block = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True))
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_block = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True))
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_block = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True))
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_block = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True))
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_up_1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True))
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_up_2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True))
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv_up_3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True))
        self.up_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv_up_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0, stride=1), nn.ReLU(inplace=True))
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1_block(x)
        conv1_out = x
        conv1_dim = x.shape[2]
        x = self.max1(x)
        x = self.conv2_block(x)
        conv2_out = x
        conv2_dim = x.shape[2]
        x = self.max2(x)
        x = self.conv3_block(x)
        conv3_out = x
        conv3_dim = x.shape[2]
        x = self.max3(x)
        x = self.conv4_block(x)
        conv4_out = x
        conv4_dim = x.shape[2]
        x = self.max4(x)
        x = self.conv5_block(x)
        x = self.up_1(x)
        lower = int((conv4_dim - x.shape[2]) / 2)
        upper = int(conv4_dim - lower)
        conv4_out_modified = conv4_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv4_out_modified], dim=1)
        x = self.conv_up_1(x)
        x = self.up_2(x)
        lower = int((conv3_dim - x.shape[2]) / 2)
        upper = int(conv3_dim - lower)
        conv3_out_modified = conv3_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv3_out_modified], dim=1)
        x = self.conv_up_2(x)
        x = self.up_3(x)
        lower = int((conv2_dim - x.shape[2]) / 2)
        upper = int(conv2_dim - lower)
        conv2_out_modified = conv2_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv2_out_modified], dim=1)
        x = self.conv_up_3(x)
        x = self.up_4(x)
        lower = int((conv1_dim - x.shape[2]) / 2)
        upper = int(conv1_dim - lower)
        conv1_out_modified = conv1_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv1_out_modified], dim=1)
        x = self.conv_up_4(x)
        x = self.conv_final(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv_down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Double_conv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_ugent_korea_pytorch_unet_segmentation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

