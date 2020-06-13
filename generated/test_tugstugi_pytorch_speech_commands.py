import sys
_module = sys.modules[__name__]
del sys
datasets = _module
split_dataset = _module
speech_commands_dataset = _module
mixup = _module
models = _module
densenet = _module
dpn = _module
resnet = _module
resnext = _module
vgg = _module
wideresnet = _module
test_cifar10 = _module
test_speech_commands = _module
train_cifar10 = _module
train_speech_commands = _module
transforms = _module
transforms_stft = _module
transforms_wav = _module

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


import numpy as np


import torch


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import math


import torch.utils.model_zoo as model_zoo


from torch.nn import init


from torch.utils.data import DataLoader


from torch.utils.data.sampler import WeightedRandomSampler


class Bottleneck(nn.Module):

    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, padding=1,
            bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, padding
            =1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=22, block=Bottleneck, dropRate=0, num_classes=
        10, growthRate=12, compressionRate=2, in_channels=3):
        super(DenseNet, self).__init__()
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6
        self.growthRate = growthRate
        self.dropRate = dropRate
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3,
            padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, growthRate=self.growthRate,
                dropRate=self.dropRate))
            self.inplanes += self.growthRate
        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, last_planes, in_planes, out_planes, dense_depth,
        stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=
            False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=
            stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes + dense_depth,
            kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes + dense_depth)
        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(nn.Conv2d(last_planes, out_planes +
                dense_depth, kernel_size=1, stride=stride, bias=False), nn.
                BatchNorm2d(out_planes + dense_depth))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:, :d, :, :] + out[:, :d, :, :], x[:, d:, :, :],
            out[:, d:, :, :]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):

    def __init__(self, num_classes, in_channels, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0],
            num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1],
            num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2],
            num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3],
            num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3] + (num_blocks[3] + 1) *
            dense_depth[3], num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth,
        stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes,
                out_planes, dense_depth, stride, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
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

    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(1, stride=1)
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


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality,
        base_width, widen_factor):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.0)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=
            1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride,
            padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride
            =1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(
                out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, nlabels, cardinality=8, depth=29, base_width=64,
        widen_factor=4, in_channels=3):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor,
            256 * self.widen_factor]
        self.conv_1_3x3 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.

        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels,
                    out_channels, pool_stride, self.cardinality, self.
                    base_width, self.widen_factor))
            else:
                block.add_module(name_, ResNeXtBottleneck(out_channels,
                    out_channels, 1, self.cardinality, self.base_width,
                    self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 1 * 1, 4096), nn.
            ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride
            =stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes,
            out_planes, kernel_size=1, stride=stride, padding=0, bias=False
            ) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride,
        dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes,
            nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
        dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes,
                out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, depth, num_classes, in_channels=3, widen_factor=1,
        dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 *
            widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(in_channels, nChannels[0], kernel_size=3,
            stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,
            dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
            dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
            dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_tugstugi_pytorch_speech_commands(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'out_planes': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Transition(*[], **{'inplanes': 4, 'outplanes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ResNeXtBottleneck(*[], **{'in_channels': 64, 'out_channels': 64, 'stride': 64, 'cardinality': 4, 'base_width': 4, 'widen_factor': 4}), [torch.rand([4, 64, 64, 64])], {})

    def test_003(self):
        self._check(CifarResNeXt(*[], **{'nlabels': 4}), [torch.rand([4, 3, 64, 64])], {})

