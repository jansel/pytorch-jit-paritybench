import sys
_module = sys.modules[__name__]
del sys
Test_img = _module
KITTILoader = _module
KITTI_submission_loader = _module
KITTI_submission_loader2012 = _module
KITTIloader2012 = _module
KITTIloader2015 = _module
SecenFlowLoader = _module
dataloader = _module
listflowfile = _module
preprocess = _module
readpfm = _module
main = _module
models = _module
basic = _module
submodule = _module
submission = _module
utils = _module

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


import random


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import math


import torch.optim as optim


from torch.autograd import Variable


import torch.utils.data


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=
        kernel_size, padding=pad, stride=stride, bias=False), nn.
        BatchNorm3d(out_planes))


class PSMNet(nn.Module):

    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1), nn.ReLU(
            inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(
            inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(
            inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(
            inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(
            inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(
            inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1,
            stride=1, bias=False))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2
                    ] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.
            size()[1] * 2, self.maxdisp / 4, refimg_fea.size()[2],
            refimg_fea.size()[3]).zero_(), volatile=not self.training)
        for i in range(self.maxdisp / 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], (i), :, i:] = refimg_fea[:,
                    :, :, i:]
                cost[:, refimg_fea.size()[1]:, (i), :, i:] = targetimg_fea[:,
                    :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], (i), :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, (i), :, :] = targetimg_fea
        cost = cost.contiguous()
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0
        cost = self.classify(cost0)
        cost = F.upsample(cost, [self.maxdisp, left.size()[2], left.size()[
            3]], mode='trilinear')
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost)
        pred = disparityregression(self.maxdisp)(pred)
        return pred


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
        kernel_size, stride=stride, padding=dilation if dilation > 1 else
        pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad,
            dilation), nn.ReLU(inplace=True))
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


class matchshifted(nn.Module):

    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = F.pad(torch.index_select(left, 3, Variable(torch.
            LongTensor([i for i in range(shift, width)]))), (shift, 0, 0, 0))
        shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.
            LongTensor([i for i in range(width - shift)]))), (shift, 0, 0, 0))
        out = torch.cat((shifted_left, shifted_right), 1).view(batch, 
            filters * 2, 1, height, width)
        return out


class disparityregression(nn.Module):

    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp
            )), [1, maxdisp, 1, 1])), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class feature_extraction(nn.Module):

    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1), nn.ReLU(
            inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True
            ), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)
            ), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)
            ), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)
            ), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1), nn.ReLU
            (inplace=True), nn.Conv2d(128, 32, kernel_size=1, padding=0,
            stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad,
            dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)
        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],
            output_skip.size()[3]), mode='bilinear')
        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],
            output_skip.size()[3]), mode='bilinear')
        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],
            output_skip.size()[3]), mode='bilinear')
        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],
            output_skip.size()[3]), mode='bilinear')
        output_feature = torch.cat((output_raw, output_skip, output_branch4,
            output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        return output_feature


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_JiaRenChang_PSMNet(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(matchshifted(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0], {})

    def test_001(self):
        self._check(disparityregression(*[], **{'maxdisp': 4}), [torch.rand([4, 4, 4, 4])], {})

