import sys
_module = sys.modules[__name__]
del sys
RGB_Loader = _module
dataloader = _module
filename_loader = _module
grayscale_Loader = _module
preprocess = _module
testing_loader = _module
main = _module
LCV_ours_sub3 = _module
models = _module
sub_ASPP = _module
testing = _module
utils = _module
disp2de = _module

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


import torch.utils.data as data


import torchvision.transforms as transforms


import random


import numpy as np


import torch.nn as nn


import torch.nn.parallel


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


import torch.nn.functional as F


import time


import math


from torchvision import transforms


class forfilter(nn.Module):

    def __init__(self, inplanes):
        super(forfilter, self).__init__()
        self.forfilter1 = nn.Conv2d(1, 1, (7, 1), 1, (0, 0), bias=False)
        self.inplanes = inplanes

    def forward(self, x):
        out = self.forfilter1(F.pad(torch.unsqueeze(x[:, 0, :, :], 1), pad=(0, 0, 3, 3), mode='replicate'))
        for i in range(1, self.inplanes):
            out = torch.cat((out, self.forfilter1(F.pad(torch.unsqueeze(x[:, i, :, :], 1), pad=(0, 0, 3, 3), mode='replicate'))), 1)
        return out


class disparityregression_sub3(nn.Module):

    def __init__(self, maxdisp):
        super(disparityregression_sub3, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp * 3)), [1, maxdisp * 3, 1, 1]) / 3), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False), nn.BatchNorm3d(out_planes))


class hourglass(nn.Module):

    def __init__(self, inplanes):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)
        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=(1, 1, 1), stride=2, bias=False), nn.BatchNorm3d(inplanes * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=2, bias=False), nn.BatchNorm3d(inplanes))

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)
        out = self.conv3(pre)
        out = self.conv4(out)
        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)
        out = self.conv6(post)
        return out, pre, post


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class feature_extraction(nn.Module):

    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.aspp1 = nn.Sequential(convbn(160, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.aspp2 = nn.Sequential(convbn(160, 32, 3, 1, 1, 6), nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(convbn(160, 32, 3, 1, 1, 12), nn.ReLU(inplace=True))
        self.aspp4 = nn.Sequential(convbn(160, 32, 3, 1, 1, 18), nn.ReLU(inplace=True))
        self.aspp5 = nn.Sequential(convbn(160, 32, 3, 1, 1, 24), nn.ReLU(inplace=True))
        self.newlastconv = nn.Sequential(convbn(224, 128, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))
        self.firstcoord = nn.Sequential(convbn(1, 32, 3, 2, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 2, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x[:, :3, :, :])
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)
        out_coord = self.firstcoord(x[:, 3:, :, :])
        output_skip_c = torch.cat((output_skip, out_coord), 1)
        ASPP1 = self.aspp1(output_skip_c)
        ASPP2 = self.aspp2(output_skip_c)
        ASPP3 = self.aspp3(output_skip_c)
        ASPP4 = self.aspp4(output_skip_c)
        ASPP5 = self.aspp5(output_skip_c)
        output_feature = torch.cat((output_raw, ASPP1, ASPP2, ASPP3, ASPP4, ASPP5), 1)
        output_feature = self.newlastconv(output_feature)
        return output_feature


class LCV(nn.Module):

    def __init__(self, maxdisp):
        super(LCV, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)
        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.forF = forfilter(32)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, up, down):
        refimg_fea = self.feature_extraction(up)
        targetimg_fea = self.feature_extraction(down)
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp / 4 * 3, refimg_fea.size()[2], refimg_fea.size()[3]).zero_())
        for i in range(self.maxdisp / 4 * 3):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea[:, :, :, :]
                cost[:, refimg_fea.size()[1]:, i, :, :] = shift_down[:, :, :, :]
                shift_down = self.forF(shift_down)
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
                shift_down = self.forF(targetimg_fea)
        cost = cost.contiguous()
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0
        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0
        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp * 3, up.size()[2], up.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp * 3, up.size()[2], up.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression_sub3(self.maxdisp)(pred1)
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression_sub3(self.maxdisp)(pred2)
        cost3 = F.upsample(cost3, [self.maxdisp * 3, up.size()[2], up.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparityregression_sub3(self.maxdisp)(pred3)
        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (disparityregression_sub3,
     lambda: ([], {'maxdisp': 4}),
     lambda: ([torch.rand([4, 12, 4, 4])], {}),
     True),
    (feature_extraction,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (forfilter,
     lambda: ([], {'inplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_albert100121_360SD_Net(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

