import sys
_module = sys.modules[__name__]
del sys
demo = _module
demo_transform = _module
loaddata = _module
loaddata_demo = _module
densenet = _module
modules = _module
net = _module
resnet = _module
senet = _module
nyu_transform = _module
sobel = _module
test = _module
train = _module
util = _module

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


import torch.nn.parallel


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import copy


import math


from torch.utils import model_zoo


import torch.backends.cudnn as cudnn


import torch.optim


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, kernel_size=7, stride=2, padding=3, bias=
            False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0',
            nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.num_features = num_features
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size
            (0), -1)
        out = self.classifier(out)
        return out, self.num_features


class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
            kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)
        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
            kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear')
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out


class E_resnet(nn.Module):

    def __init__(self, original_model, num_features=2048):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)
        return x_block1, x_block2, x_block3, x_block4


class E_densenet(nn.Module):

    def __init__(self, original_model, num_features=2208):
        super(E_densenet, self).__init__()
        self.features = original_model.features

    def forward(self, x):
        x01 = self.features[0](x)
        x02 = self.features[1](x01)
        x03 = self.features[2](x02)
        x04 = self.features[3](x03)
        x_block1 = self.features[4](x04)
        x_block1 = self.features[5][0](x_block1)
        x_block1 = self.features[5][1](x_block1)
        x_block1 = self.features[5][2](x_block1)
        x_tran1 = self.features[5][3](x_block1)
        x_block2 = self.features[6](x_tran1)
        x_block2 = self.features[7][0](x_block2)
        x_block2 = self.features[7][1](x_block2)
        x_block2 = self.features[7][2](x_block2)
        x_tran2 = self.features[7][3](x_block2)
        x_block3 = self.features[8](x_tran2)
        x_block3 = self.features[9][0](x_block3)
        x_block3 = self.features[9][1](x_block3)
        x_block3 = self.features[9][2](x_block3)
        x_tran3 = self.features[9][3](x_block3)
        x_block4 = self.features[10](x_tran3)
        x_block4 = F.relu(self.features[11](x_block4))
        return x_block1, x_block2, x_block3, x_block4


class E_senet(nn.Module):

    def __init__(self, original_model, num_features=2048):
        super(E_senet, self).__init__()
        self.base = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.base[0](x)
        x_block1 = self.base[1](x)
        x_block2 = self.base[2](x_block1)
        x_block3 = self.base[3](x_block2)
        x_block4 = self.base[4](x_block3)
        return x_block1, x_block2, x_block3, x_block4


class D(nn.Module):

    def __init__(self, num_features=2048):
        super(D, self).__init__()
        self.conv = nn.Conv2d(num_features, num_features // 2, kernel_size=
            1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)
        self.up1 = _UpProjection(num_input_features=num_features,
            num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up2 = _UpProjection(num_input_features=num_features,
            num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up3 = _UpProjection(num_input_features=num_features,
            num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up4 = _UpProjection(num_input_features=num_features,
            num_output_features=num_features // 2)
        num_features = num_features // 2

    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        return x_d4


class MFF(nn.Module):

    def __init__(self, block_channel, num_features=64):
        super(MFF, self).__init__()
        self.up1 = _UpProjection(num_input_features=block_channel[0],
            num_output_features=16)
        self.up2 = _UpProjection(num_input_features=block_channel[1],
            num_output_features=16)
        self.up3 = _UpProjection(num_input_features=block_channel[2],
            num_output_features=16)
        self.up4 = _UpProjection(num_input_features=block_channel[3],
            num_output_features=16)
        self.conv = nn.Conv2d(num_features, num_features, kernel_size=5,
            stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)
        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)
        return x


class R(nn.Module):

    def __init__(self, block_channel):
        super(R, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.conv0 = nn.Conv2d(num_features, num_features, kernel_size=5,
            stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=5,
            stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, 1, kernel_size=5, stride=1,
            padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)
        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1)
        return x2


class model(nn.Module):

    def __init__(self, Encoder, num_features, block_channel):
        super(model, self).__init__()
        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder
            .size(2), x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
        inplanes=128, input_3x3=True, downsample_kernel_size=3,
        downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2,
                padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), (
                'relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64,
                3, stride=1, padding=1, bias=False)), ('bn2', nn.
                BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), (
                'conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3',
                nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=
                7, stride=2, padding=3, bias=False)), ('bn1', nn.
                BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=
            True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0],
            groups=groups, reduction=reduction, downsample_kernel_size=1,
            downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=
        1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=downsample_kernel_size, stride
                =stride, padding=downsample_padding, bias=False), nn.
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction,
            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x, x_):
        x = self.features(x)
        x = self.logits(x)
        return x


class Sobel(nn.Module):

    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1,
            bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))
        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_JunjH_Revisiting_Single_Depth_Estimation(_paritybench_base):
    pass

    def test_000(self):
        self._check(_Transition(*[], **{'num_input_features': 4, 'num_output_features': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(D(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 2048, 64, 64])], {})

    def test_002(self):
        self._check(R(*[], **{'block_channel': [4, 4, 4, 4]}), [torch.rand([4, 64, 64, 64])], {})

    def test_003(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(SEModule(*[], **{'channels': 4, 'reduction': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Sobel(*[], **{}), [torch.rand([4, 1, 64, 64])], {})
