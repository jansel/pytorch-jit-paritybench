import sys
_module = sys.modules[__name__]
del sys
lednet_imagenet = _module
main = _module
dataset = _module
eval_cityscapes_color = _module
eval_cityscapes_server = _module
eval_forward_time = _module
eval_iou = _module
iouEval = _module
lednet_no_bn = _module
transform = _module
lednet = _module
lednet_1 = _module
main = _module
utils = _module
dataset = _module
iouEval = _module
loss = _module
transform = _module
visualize = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


import torch.nn.init as init


import torch.nn.functional as F


import time


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


from torch.optim import lr_scheduler


import numpy as np


from torch.utils.data import Dataset


from torch.autograd import Variable


from torch.utils.data import DataLoader


from torchvision.transforms import Compose


from torchvision.transforms import CenterCrop


from torchvision.transforms import Normalize


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


from torchvision.transforms import ToPILImage


from torch.nn.functional import interpolate as interpolate


import random


import math


from torch.optim import SGD


from torch.optim import Adam


from torchvision.transforms import Pad


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias), nn.BatchNorm2d(out_ch, eps=0.001), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DownsamplerBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DownsamplerBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel - in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


class SS_nbt_module(nn.Module):

    def __init__(self, chann, dropprob, dilated):
        super(SS_nbt_module, self).__init__()
        oup_inc = chann // 2
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=0.001)
        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=0.001)
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=0.001)
        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, input):
        residual = input
        x1, x2 = split(input)
        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)
        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)
        if self.dropout.p != 0:
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)
        out = self._concat(output1, output2)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out, 2)


class Encoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 32)
        self.layers = nn.ModuleList()
        for x in range(0, 3):
            self.layers.append(SS_nbt_module(32, 0.03, 1))
        self.layers.append(DownsamplerBlock(32, 64))
        for x in range(0, 2):
            self.layers.append(SS_nbt_module(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 1):
            self.layers.append(SS_nbt_module(128, 0.3, 1))
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
        for x in range(0, 1):
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
            self.layers.append(SS_nbt_module(128, 0.3, 17))
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        if predict:
            output = self.output_conv(output)
        return output


class Features(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.extralayer1 = nn.MaxPool2d(2, stride=2)
        self.extralayer2 = nn.AvgPool2d(14, 1, 0)

    def forward(self, input):
        output = self.encoder(input)
        output = self.extralayer1(output)
        output = self.extralayer2(output)
        return output


class Classifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(128, num_classes)

    def forward(self, input):
        output = input.view(input.size(0), 128)
        output = self.linear(output)
        return output


class LEDNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.features = Features()
        self.classifier = Classifier(num_classes)

    def forward(self, input):
        output = self.features(input)
        output = self.classifier(output)
        return output


class Interpolate(nn.Module):

    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
        return x


class APN_Module(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0))
        self.mid = nn.Sequential(Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0))
        self.down1 = Conv2dBnRelu(in_ch, 128, kernel_size=7, stride=2, padding=3)
        self.down2 = Conv2dBnRelu(128, 128, kernel_size=5, stride=2, padding=2)
        self.down3 = nn.Sequential(Conv2dBnRelu(128, 128, kernel_size=3, stride=2, padding=1), Conv2dBnRelu(128, 20, kernel_size=1, stride=1, padding=0))
        self.conv2 = Conv2dBnRelu(128, 20, kernel_size=1, stride=1, padding=0)
        self.conv1 = Conv2dBnRelu(128, 20, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        b1 = self.branch1(x)
        b1 = interpolate(b1, size=(h, w), mode='bilinear', align_corners=True)
        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = interpolate(x3, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        x2 = self.conv2(x2)
        x = x2 + x3
        x = interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x = x + x1
        x = interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.mul(x, mid)
        x = x + b1
        return x


class Decoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.apn = APN_Module(in_ch=128, out_ch=20)

    def forward(self, input):
        output = self.apn(input)
        out = interpolate(output, size=(512, 1024), mode='bilinear', align_corners=True)
        return out


class Net(nn.Module):

    def __init__(self, num_classes, encoder=None):
        super().__init__()
        if encoder == None:
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)
            return self.decoder.forward(output)


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Classifier,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 128])], {}),
     True),
    (Conv2dBnRelu,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Net,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SS_nbt_module,
     lambda: ([], {'chann': 4, 'dropprob': 0.5, 'dilated': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_xiaoyufenfei_LEDNet(_paritybench_base):
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

