import sys
_module = sys.modules[__name__]
del sys
datasets = _module
duc = _module
gcn = _module
loss = _module
resnet = _module
tester = _module
trainer = _module
transform = _module
upsample = _module
visualize = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import torch.utils.model_zoo as model_zoo


from torchvision import models


import math


from torch import nn


from torch.utils import data


import torch.optim as optim


from torch.autograd import Variable


from torchvision.transforms import ToPILImage


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


from torchvision.transforms import CenterCrop


import numpy as np


from torchvision.transforms import Normalize


class DUC(nn.Module):

    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class FCN(nn.Module):

    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.num_classes = num_classes
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.duc1 = DUC(2048, 2048 * 2)
        self.duc2 = DUC(1024, 1024 * 2)
        self.duc3 = DUC(512, 512 * 2)
        self.duc4 = DUC(128, 128 * 2)
        self.duc5 = DUC(64, 64 * 2)
        self.out1 = self._classifier(1024)
        self.out2 = self._classifier(512)
        self.out3 = self._classifier(128)
        self.out4 = self._classifier(64)
        self.out5 = self._classifier(32)
        self.transformer = nn.Conv2d(320, 128, kernel_size=1)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(nn.Conv2d(inplanes, self.num_classes, 1), nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, padding=1))
        return nn.Sequential(nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False), nn.BatchNorm2d(inplanes / 2, momentum=0.95), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(inplanes / 2, self.num_classes, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x
        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
        dfm1 = fm3 + self.duc1(fm4)
        out16 = self.out1(dfm1)
        dfm2 = fm2 + self.duc2(dfm1)
        out8 = self.out2(dfm2)
        dfm3 = fm1 + self.duc3(dfm2)
        dfm3_t = self.transformer(torch.cat((dfm3, pool_x), 1))
        out4 = self.out3(dfm3_t)
        dfm4 = conv_x + self.duc4(dfm3_t)
        out2 = self.out4(dfm4)
        dfm5 = self.duc5(dfm4)
        out = self.out5(dfm5)
        return out, out2, out4, out8, out16


class GCN(nn.Module):

    def __init__(self, inplanes, planes, ks=7):
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, planes, kernel_size=(ks, 1), padding=(ks / 2, 0))
        self.conv_l2 = nn.Conv2d(planes, planes, kernel_size=(1, ks), padding=(0, ks / 2))
        self.conv_r1 = nn.Conv2d(inplanes, planes, kernel_size=(1, ks), padding=(0, ks / 2))
        self.conv_r2 = nn.Conv2d(planes, planes, kernel_size=(ks, 1), padding=(ks / 2, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class Refine(nn.Module):

    def __init__(self, planes):
        super(Refine, self).__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = residual + x
        return out


class FCN(nn.Module):

    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.num_classes = num_classes
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.gcn1 = GCN(2048, self.num_classes)
        self.gcn2 = GCN(1024, self.num_classes)
        self.gcn3 = GCN(512, self.num_classes)
        self.gcn4 = GCN(64, self.num_classes)
        self.gcn5 = GCN(64, self.num_classes)
        self.refine1 = Refine(self.num_classes)
        self.refine2 = Refine(self.num_classes)
        self.refine3 = Refine(self.num_classes)
        self.refine4 = Refine(self.num_classes)
        self.refine5 = Refine(self.num_classes)
        self.refine6 = Refine(self.num_classes)
        self.refine7 = Refine(self.num_classes)
        self.refine8 = Refine(self.num_classes)
        self.refine9 = Refine(self.num_classes)
        self.refine10 = Refine(self.num_classes)
        self.out0 = self._classifier(2048)
        self.out1 = self._classifier(1024)
        self.out2 = self._classifier(512)
        self.out_e = self._classifier(256)
        self.out3 = self._classifier(64)
        self.out4 = self._classifier(64)
        self.out5 = self._classifier(32)
        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def _classifier(self, inplanes):
        return nn.Sequential(nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False), nn.BatchNorm2d(inplanes / 2), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(inplanes / 2, self.num_classes, 1))

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x
        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
        gcfm1 = self.refine1(self.gcn1(fm4))
        gcfm2 = self.refine2(self.gcn2(fm3))
        gcfm3 = self.refine3(self.gcn3(fm2))
        gcfm4 = self.refine4(self.gcn4(pool_x))
        gcfm5 = self.refine5(self.gcn5(conv_x))
        fs1 = self.refine6(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)
        fs2 = self.refine7(F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3)
        fs3 = self.refine8(F.upsample_bilinear(fs2, pool_x.size()[2:]) + gcfm4)
        fs4 = self.refine9(F.upsample_bilinear(fs3, conv_x.size()[2:]) + gcfm5)
        out = self.refine10(F.upsample_bilinear(fs4, input.size()[2:]))
        return out, fs4, fs3, fs2, fs1, gcfm1


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu(out)
        return out


class DeconvBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.upsample is not None:
            shortcut = self.upsample(x)
        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, downblock, upblock, num_layers, n_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dlayer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.dlayer2 = self._make_downlayer(downblock, 128, num_layers[1], stride=2)
        self.dlayer3 = self._make_downlayer(downblock, 256, num_layers[2], stride=2)
        self.dlayer4 = self._make_downlayer(downblock, 512, num_layers[3], stride=2)
        self.uplayer1 = self._make_up_block(upblock, 512, 1, stride=2)
        self.uplayer2 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(upblock, 64, 2, stride=2)
        upsample = nn.Sequential(nn.ConvTranspose2d(self.in_channels, 64, kernel_size=1, stride=2, bias=False, output_padding=1), nn.BatchNorm2d(64))
        self.uplayer_top = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample)
        self.conv1_1 = nn.ConvTranspose2d(64, n_classes, kernel_size=1, stride=1, bias=False)

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, init_channels * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(init_channels * block.expansion))
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))
        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(nn.ConvTranspose2d(self.in_channels, init_channels * 2, kernel_size=1, stride=stride, bias=False, output_padding=1), nn.BatchNorm2d(init_channels * 2))
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        img = x
        x_size = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dlayer1(x)
        x = self.dlayer2(x)
        x = self.dlayer3(x)
        x = self.dlayer4(x)
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_top(x)
        x = self.conv1_1(x, output_size=img.size())
        return x


class Upsample(nn.Module):

    def __init__(self, inplanes, planes):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, size):
        x = F.upsample_bilinear(x, size=size)
        x = self.conv1(x)
        x = self.bn(x)
        return x


class Fusion(nn.Module):

    def __init__(self, inplanes):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        out = self.bn(self.conv(x1)) + x2
        out = self.relu(out)
        return out


class FCN(nn.Module):

    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.num_classes = num_classes
        resnet = models.resnet101(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)
        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)
        self.out0 = self._classifier(2048)
        self.out1 = self._classifier(1024)
        self.out2 = self._classifier(512)
        self.out_e = self._classifier(256)
        self.out3 = self._classifier(64)
        self.out4 = self._classifier(64)
        self.out5 = self._classifier(32)
        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(nn.Conv2d(inplanes, self.num_classes, 1), nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, padding=1))
        return nn.Sequential(nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False), nn.BatchNorm2d(inplanes / 2), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(inplanes / 2, self.num_classes, 1))

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x
        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
        out32 = self.out0(fm4)
        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        out16 = self.out1(fsfm1)
        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        out8 = self.out2(fsfm2)
        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        out4 = self.out3(fsfm3)
        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        out2 = self.out4(fsfm4)
        fsfm5 = self.upsample5(fsfm4, input.size()[2:])
        out = self.out5(fsfm5)
        return out, out2, out4, out8, out16, out32


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DUC,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fusion,
     lambda: ([], {'inplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Refine,
     lambda: ([], {'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ycszen_pytorch_segmentation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

