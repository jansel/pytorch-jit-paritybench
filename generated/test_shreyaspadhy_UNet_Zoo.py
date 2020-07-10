import sys
_module = sys.modules[__name__]
del sys
CLSTM = _module
data = _module
dataParser = _module
losses = _module
main = _module
main_bdclstm = _module
main_small = _module
models = _module
plot_ims = _module

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


import torch.nn as nn


from torch.autograd import Variable


import re


from torch.utils.data.dataset import Dataset


import numpy as np


import scipy.io as sio


import torchvision.transforms as tr


import torch.functional as f


import torch.optim as optim


from torch.utils.data import DataLoader


class CLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(CLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, self.num_features * self.hidden_channels, self.kernel_size, 1, self.padding)

    def forward(self, x, h, c):
        combined = torch.cat((x, h), dim=1)
        A = self.conv(combined)
        Ai, Af, Ao, Ag = torch.split(A, A.size()[1] // self.num_features, dim=1)
        i = torch.sigmoid(Ai)
        f = torch.sigmoid(Af)
        o = torch.sigmoid(Ao)
        g = torch.tanh(Ag)
        c = c * f + i * g
        h = o * torch.tanh(c)
        return h, c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])), Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1]))
        except:
            return Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])), Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1]))


class CLSTM(nn.Module):

    def __init__(self, input_channels=64, hidden_channels=[64], kernel_size=5, bias=True):
        super(CLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.bias = bias
        self.all_layers = []
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = CLSTMCell(self.input_channels[layer], self.hidden_channels[layer], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    def forward(self, x):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        outputs = []
        for step in range(steps):
            input = torch.squeeze(x[:, (step), :, :, :], dim=1)
            for layer in range(self.num_layers):
                if step == 0:
                    h, c = CLSTMCell.init_hidden(bsize, self.hidden_channels[layer], (height, width))
                    internal_state.append((h, c))
                name = 'cell{}'.format(layer)
                h, c = internal_state[layer]
                input, c = getattr(self, name)(input, h, c)
                internal_state[layer] = input, c
            outputs.append(input)
        return outputs


class BDCLSTM(nn.Module):

    def __init__(self, input_channels=64, hidden_channels=[64], kernel_size=5, bias=True, num_classes=2):
        super(BDCLSTM, self).__init__()
        self.forward_net = CLSTM(input_channels, hidden_channels, kernel_size, bias)
        self.reverse_net = CLSTM(input_channels, hidden_channels, kernel_size, bias)
        self.conv = nn.Conv2d(2 * hidden_channels[-1], num_classes, kernel_size=1)
        self.soft = nn.Softmax2d()

    def forward(self, x1, x2, x3):
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        x3 = torch.unsqueeze(x3, dim=1)
        xforward = torch.cat((x1, x2), dim=1)
        xreverse = torch.cat((x3, x2), dim=1)
        yforward = self.forward_net(xforward)
        yreverse = self.reverse_net(xreverse)
        ycat = torch.cat((yforward[-1], yreverse[-1]), dim=1)
        y = self.conv(ycat)
        y = self.soft(y)
        return y


class DICELossMultiClass(nn.Module):

    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, mask):
        num_classes = output.size(1)
        dice_eso = 0
        for i in range(num_classes):
            probs = torch.squeeze(output[:, (i), :, :], 1)
            mask = torch.squeeze(mask[:, (i), :, :], 1)
            num = probs * mask
            num = torch.sum(num, 2)
            num = torch.sum(num, 1)
            den1 = probs * probs
            den1 = torch.sum(den1, 2)
            den1 = torch.sum(den1, 1)
            den2 = mask * mask
            den2 = torch.sum(den2, 2)
            den2 = torch.sum(den2, 1)
            eps = 1e-07
            dice = 2 * ((num + eps) / (den1 + den2 + eps))
            dice_eso += dice
        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):
        probs = torch.squeeze(output, 1)
        mask = torch.squeeze(mask, 1)
        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)
        den1 = probs * probs
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)
        den2 = mask * mask
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)
        eps = 1e-08
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        dice_eso = dice
        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


class Conv3x3(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_feat), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_feat), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


class UNet(nn.Module):

    def __init__(self, num_channels=1, num_classes=2):
        super(UNet, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]
        self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0]))
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[0], num_feat[1]))
        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[1], num_feat[2]))
        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[2], num_feat[3]))
        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[3], num_feat[4]))
        self.up1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3])
        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2])
        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1])
        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0])
        self.final = nn.Sequential(nn.Conv2d(num_feat[0], num_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs, return_features=False):
        down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        down4_feat = self.down4(down3_feat)
        bottom_feat = self.bottom(down4_feat)
        up1_feat = self.up1(bottom_feat, down4_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.up2(up1_feat, down3_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.up3(up2_feat, down2_feat)
        up3_feat = self.upconv3(up3_feat)
        up4_feat = self.up4(up3_feat, down1_feat)
        up4_feat = self.upconv4(up4_feat)
        if return_features:
            outputs = up4_feat
        else:
            outputs = self.final(up4_feat)
        return outputs


class Conv3x3Small(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(Conv3x3Small, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.ELU(), nn.Dropout(p=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.ELU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpSample(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.up(inputs)
        out = torch.cat([outputs, down_outputs], 1)
        return out


class UNetSmall(nn.Module):

    def __init__(self, num_channels=1, num_classes=2):
        super(UNetSmall, self).__init__()
        num_feat = [32, 64, 128, 256]
        self.down1 = nn.Sequential(Conv3x3Small(num_channels, num_feat[0]))
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(num_feat[0]), Conv3x3Small(num_feat[0], num_feat[1]))
        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(num_feat[1]), Conv3x3Small(num_feat[1], num_feat[2]))
        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(num_feat[2]), Conv3x3Small(num_feat[2], num_feat[3]), nn.BatchNorm2d(num_feat[3]))
        self.up1 = UpSample(num_feat[3], num_feat[2])
        self.upconv1 = nn.Sequential(Conv3x3Small(num_feat[3] + num_feat[2], num_feat[2]), nn.BatchNorm2d(num_feat[2]))
        self.up2 = UpSample(num_feat[2], num_feat[1])
        self.upconv2 = nn.Sequential(Conv3x3Small(num_feat[2] + num_feat[1], num_feat[1]), nn.BatchNorm2d(num_feat[1]))
        self.up3 = UpSample(num_feat[1], num_feat[0])
        self.upconv3 = nn.Sequential(Conv3x3Small(num_feat[1] + num_feat[0], num_feat[0]), nn.BatchNorm2d(num_feat[0]))
        self.final = nn.Sequential(nn.Conv2d(num_feat[0], 1, kernel_size=1), nn.Sigmoid())

    def forward(self, inputs, return_features=False):
        down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        bottom_feat = self.bottom(down3_feat)
        up1_feat = self.up1(bottom_feat, down3_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.up2(up1_feat, down2_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.up3(up2_feat, down1_feat)
        up3_feat = self.upconv3(up3_feat)
        if return_features:
            outputs = up3_feat
        else:
            outputs = self.final(up3_feat)
        return outputs


class Conv3x3Drop(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(Conv3x3Drop, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.Dropout(p=0.2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_feat), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CLSTMCell,
     lambda: ([], {'input_channels': 4, 'hidden_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 3, 3])], {}),
     True),
    (Conv3x3,
     lambda: ([], {'in_feat': 4, 'out_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3x3Drop,
     lambda: ([], {'in_feat': 4, 'out_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3x3Small,
     lambda: ([], {'in_feat': 4, 'out_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DICELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (UNetSmall,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (UpConcat,
     lambda: ([], {'in_feat': 4, 'out_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 8, 8])], {}),
     True),
    (UpSample,
     lambda: ([], {'in_feat': 4, 'out_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 8, 8])], {}),
     True),
]

class Test_shreyaspadhy_UNet_Zoo(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

