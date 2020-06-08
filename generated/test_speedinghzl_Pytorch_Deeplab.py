import sys
_module = sys.modules[__name__]
del sys
deeplab = _module
loss = _module
metric = _module
model = _module
evaluate = _module
evaluate_msc = _module
train = _module
train_msc = _module

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


import torch.nn.functional as F


import torch.nn as nn


from torch.autograd import Variable


import math


import torch.utils.model_zoo as model_zoo


import numpy as np


import scipy


from scipy import ndimage


from torch.utils import data


from collections import OrderedDict


import torch.optim as optim


import scipy.misc


import torch.backends.cudnn as cudnn


import random


import scipy.ndimage as nd


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)
            ].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average
            =self.size_average)
        return loss


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


affine_par = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes,
                kernel_size=3, stride=1, padding=padding, dilation=dilation,
                bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class Residual_Covolution(nn.Module):

    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding
            =12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1,
            padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1,
            padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding
            =0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg


class Residual_Refinement_Module(nn.Module):

    def __init__(self, num_classes):
        super(Residual_Refinement_Module, self).__init__()
        self.RC1 = Residual_Covolution(2048, 512, num_classes)
        self.RC2 = Residual_Covolution(2048, 512, num_classes)

    def forward(self, x):
        x, seg1 = self.RC1(x)
        _, seg2 = self.RC2(x)
        return [seg1, seg1 + seg2]


class ResNet_Refine(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_Refine, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        self.layer5 = Residual_Refinement_Module(num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion or 
            dilation == 2 or dilation == 4):
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 
            24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion or 
            dilation == 2 or dilation == 4):
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series,
        num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class MS_Deeplab(nn.Module):

    def __init__(self, block, num_classes):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], num_classes)

    def forward(self, x):
        output = self.Scale(x)
        output_size = output.size()[2]
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size=(int(input_size * 0.75) + 1, int(
            input_size * 0.75) + 1), mode='bilinear')
        self.interp2 = nn.Upsample(size=(int(input_size * 0.5) + 1, int(
            input_size * 0.5) + 1), mode='bilinear')
        self.interp3 = nn.Upsample(size=(output_size, output_size), mode=
            'bilinear')
        x75 = self.interp1(x)
        output75 = self.interp3(self.Scale(x75))
        x5 = self.interp2(x)
        output5 = self.interp3(self.Scale(x5))
        out_max = torch.max(torch.max(output, output75), output5)
        return [output, output75, output5, out_max]


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_speedinghzl_Pytorch_Deeplab(_paritybench_base):
    pass

    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(Classifier_Module(*[], **{'dilation_series': [4, 4], 'padding_series': [4, 4], 'num_classes': 4}), [torch.rand([4, 2048, 64, 64])], {})

    def test_002(self):
        self._check(Residual_Covolution(*[], **{'icol': 4, 'ocol': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Residual_Refinement_Module(*[], **{'num_classes': 4}), [torch.rand([4, 2048, 64, 64])], {})
