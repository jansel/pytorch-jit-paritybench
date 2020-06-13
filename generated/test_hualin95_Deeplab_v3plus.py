import sys
_module = sys.modules[__name__]
del sys
global_config = _module
Voc_Dataset = _module
datasets = _module
cityscapes_Dataset = _module
graphs = _module
AlignedXceptionWithoutDeformable = _module
ResNet101 = _module
Xception = _module
models = _module
decoder = _module
encoder = _module
sync_batchnorm = _module
batchnorm = _module
comm = _module
replicate = _module
unittest = _module
imgaes = _module
tools = _module
test_cityscapes = _module
train_cityscapes = _module
train_voc = _module
utils = _module
data_utils = _module
eval = _module
generate_list = _module

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


import math


import logging


import torch


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


from torch.nn import init


import collections


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


import functools


from torch.nn.parallel.data_parallel import DataParallel


import numpy as np


from math import ceil


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1
            )
        pad_total = kernel_size_effective - 1
        padding = pad_total // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding=padding, dilation=dilation, groups=in_channels,
            bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
            bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps=3, strides=1,
        start_with_relu=True, grow_first=True, dilation=1):
        """

        :param in_filters:
        :param out_filters:
        :param reps:
        :param strides:
        :param start_with_relu:
        :param grow_first: whether add channels at first
        """
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=
                strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                bias=False, dilation=dilation))
            rep.append(nn.BatchNorm2d(out_filters))
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_filters, out_filters, 3, stride=
                1, bias=False, dilation=dilation))
            rep.append(nn.BatchNorm2d(out_filters))
        else:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, in_filters, 3, stride=1,
                bias=False, dilation=dilation))
            rep.append(nn.BatchNorm2d(in_filters))
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                bias=False, dilation=dilation))
            rep.append(nn.BatchNorm2d(out_filters))
        rep.append(self.relu)
        rep.append(SeparableConv2d(out_filters, out_filters, 3, stride=
            strides, bias=False, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, output_stride, pretrained=True):
        super(Xception, self).__init__()
        if output_stride == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = 2, 4
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = 1, 2
        else:
            raise Warning('atrous_rates must be 8 or 16!')
        self.conv1 = nn.Conv2d(3, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 3, 2, start_with_relu=False,
            grow_first=True)
        self.block2 = Block(128, 256, 3, 2, start_with_relu=True,
            grow_first=True)
        self.block3 = Block(256, 728, 3, strides=entry_block3_stride,
            start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block12 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block13 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block14 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block15 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block16 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block17 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block18 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block19 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True, dilation=middle_block_rate)
        self.block20 = Block(728, 1024, 3, 1, start_with_relu=True,
            grow_first=False, dilation=exit_block_rates[0])
        self.conv3 = SeparableConv2d(1024, 1536, kernel_size=3, stride=1,
            dilation=exit_block_rates[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, kernel_size=3, stride=1,
            dilation=exit_block_rates[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, kernel_size=3, stride=1,
            dilation=exit_block_rates[1])
        self.bn5 = nn.BatchNorm2d(2048)
        self._init_weights()
        if pretrained is not False:
            self._load_xception_weight()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        low_level_features = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_features

    def _load_xception_weight(self):
        None
        pretrained_dict = model_zoo.load_url(url=
            'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
            , model_dir='/data/linhua/VOCdevkit/')
        model_dict = self.state_dict()
        new_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    new_dict[k] = v
                    new_dict[k.replace('block11', 'block12')] = v
                    new_dict[k.replace('block11', 'block13')] = v
                    new_dict[k.replace('block11', 'block14')] = v
                    new_dict[k.replace('block11', 'block15')] = v
                    new_dict[k.replace('block11', 'block16')] = v
                    new_dict[k.replace('block11', 'block17')] = v
                    new_dict[k.replace('block11', 'block18')] = v
                    new_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    new_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    new_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


model_urls = {'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


class ResNet(nn.Module):

    def __init__(self, block, layers, bn_momentum=0.1, pretrained=False,
        output_stride=16):
        if output_stride == 16:
            dilations = [1, 1, 1, 2]
            strides = [1, 2, 2, 1]
        elif output_stride == 8:
            dilations = [1, 1, 2, 4]
            strides = [1, 2, 1, 1]
        else:
            raise Warning('output_stride must be 8 or 16!')
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = SynchronizedBatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides
            [0], dilation=dilations[0], bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=
            strides[1], dilation=dilations[1], bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=
            strides[2], dilation=dilations[2], bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=
            strides[3], dilation=dilations[3], bn_momentum=bn_momentum)
        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion, momentum=
                bn_momentum))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation,
            downsample, bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                bn_momentum=bn_momentum))
        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1,
        start_with_relu=True, grow_first=True):
        """

        :param in_filters:
        :param out_filters:
        :param reps:
        :param strides:
        :param start_with_relu:
        :param grow_first: whether add channels at first
        """
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=
                strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes):
        super(Xception, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False,
            grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True,
            grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True,
            grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True,
            grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class Decoder(nn.Module):

    def __init__(self, class_num, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)
        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        x_4 = F.interpolate(x, size=low_level_feature.size()[2:4], mode=
            'bilinear', align_corners=True)
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)
        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def resnet101(bn_momentum=0.1, pretrained=False, output_stride=16):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], bn_momentum, pretrained,
        output_stride)
    return model


class DeepLab(nn.Module):

    def __init__(self, output_stride, class_num, pretrained, bn_momentum=
        0.1, freeze_bn=False):
        super(DeepLab, self).__init__()
        self.Resnet101 = resnet101(bn_momentum, pretrained)
        self.encoder = Encoder(bn_momentum, output_stride)
        self.decoder = Decoder(class_num, bn_momentum)
        if freeze_bn:
            self.freeze_bn()
            None

    def forward(self, input):
        x, low_level_features = self.Resnet101(input)
        x = self.encoder(x)
        predict = self.decoder(x, low_level_features)
        output = F.interpolate(predict, size=input.size()[2:4], mode=
            'bilinear', align_corners=True)
        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()


def _AsppConv(in_channels, out_channels, kernel_size, stride=1, padding=0,
    dilation=1, bn_momentum=0.1):
    asppconv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
        kernel_size, stride, padding, dilation, bias=False),
        SynchronizedBatchNorm2d(out_channels, momentum=bn_momentum), nn.ReLU())
    return asppconv


class AsppModule(nn.Module):

    def __init__(self, bn_momentum=0.1, output_stride=16):
        super(AsppModule, self).__init__()
        if output_stride == 16:
            atrous_rates = [0, 6, 12, 18]
        elif output_stride == 8:
            atrous_rates = 2 * [0, 12, 24, 36]
        else:
            raise Warning('output_stride must be 8 or 16!')
        self._atrous_convolution1 = _AsppConv(2048, 256, 1, 1, bn_momentum=
            bn_momentum)
        self._atrous_convolution2 = _AsppConv(2048, 256, 3, 1, padding=
            atrous_rates[1], dilation=atrous_rates[1], bn_momentum=bn_momentum)
        self._atrous_convolution3 = _AsppConv(2048, 256, 3, 1, padding=
            atrous_rates[2], dilation=atrous_rates[2], bn_momentum=bn_momentum)
        self._atrous_convolution4 = _AsppConv(2048, 256, 3, 1, padding=
            atrous_rates[3], dilation=atrous_rates[3], bn_momentum=bn_momentum)
        self._image_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.
            Conv2d(2048, 256, kernel_size=1, bias=False),
            SynchronizedBatchNorm2d(256, momentum=bn_momentum), nn.ReLU())
        self.__init_weight()

    def forward(self, input):
        input1 = self._atrous_convolution1(input)
        input2 = self._atrous_convolution2(input)
        input3 = self._atrous_convolution3(input)
        input4 = self._atrous_convolution4(input)
        input5 = self._image_pool(input)
        input5 = F.interpolate(input=input5, size=input4.size()[2:4], mode=
            'bilinear', align_corners=True)
        return torch.cat((input1, input2, input3, input4, input5), dim=1)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Encoder(nn.Module):

    def __init__(self, bn_momentum=0.1, output_stride=16):
        super(Encoder, self).__init__()
        self.ASPP = AsppModule(bn_momentum=bn_momentum, output_stride=
            output_stride)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout = nn.Dropout(0.5)
        self.__init_weight()

    def forward(self, input):
        input = self.ASPP(input)
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu(input)
        input = self.dropout(input)
        return input

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier',
    'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(
                ), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0
            ] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum',
    'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(
                input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(
                input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std *
                self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.
            get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 +
                2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self.running_mean = (1 - self.momentum
            ) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module,
            device_ids)
        execute_replication_callbacks(modules)
        return modules


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hualin95_Deeplab_v3plus(_paritybench_base):
    pass
    def test_000(self):
        self._check(Block(*[], **{'in_filters': 4, 'out_filters': 4, 'reps': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(SeparableConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Xception(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

