import sys
_module = sys.modules[__name__]
del sys
dataset = _module
demo = _module
eval = _module
lapsrn = _module
lapsrn_wgan = _module
main_lapsrn = _module
main_lapwgan = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import numpy as np


import math


import random


import torch.backends.cudnn as cudnn


import torch.optim as optim


from torch.autograd import Variable


from torch.utils.data import DataLoader


import torch.utils.model_zoo as model_zoo


class _Conv_Block(nn.Module):

    def __init__(self):
        super(_Conv_Block, self).__init__()
        self.cov_block = nn.Sequential(nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=2,
            padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        output = self.cov_block(x)
        return output


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) /
        factor)
    return torch.from_numpy(filter).float()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)
        self.convt_I2 = nn.ConvTranspose2d(in_channels=1, out_channels=1,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=64, out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1
        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2
        return HR_2x, HR_4x


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-06

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class _Conv_Block(nn.Module):

    def __init__(self):
        super(_Conv_Block, self).__init__()
        self.cov_block = nn.Sequential(nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=2,
            padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        output = self.cov_block(x)
        return output


class _netG(nn.Module):

    def __init__(self):
        super(_netG, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)
        self.convt_I2 = nn.ConvTranspose2d(in_channels=1, out_channels=1,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=64, out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1
        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2
        return HR_2x, HR_4x


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-06

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class _netD(nn.Module):

    def __init__(self):
        super(_netD, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels
            =64, kernel_size=3, stride=1, padding=1, bias=False), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=64, kernel_size=4, stride=2, padding=1, bias=False
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64,
            out_channels=128, kernel_size=3, stride=1, padding=1, bias=
            False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=
            128, out_channels=128, kernel_size=4, stride=2, padding=1, bias
            =False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels
            =128, out_channels=256, kernel_size=3, stride=1, padding=1,
            bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=4, stride=2,
            padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride
            =1, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn
            .Conv2d(in_channels=512, out_channels=512, kernel_size=4,
            stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        out = out.mean(0)
        return out.view(1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_twtygqyy_pytorch_LapSRN(_paritybench_base):
    pass
    def test_000(self):
        self._check(_Conv_Block(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_001(self):
        self._check(Net(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    def test_002(self):
        self._check(L1_Charbonnier_loss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(_netG(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

