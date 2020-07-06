import sys
_module = sys.modules[__name__]
del sys
PFEC_finetune = _module
PFEC_vggprune = _module
VGG_cifar = _module
main_cifar_vgg = _module
main_cifar_vgg_log = _module
pruning_cifar_vgg = _module
cal_flop = _module
infer_pruned = _module
models = _module
caffe_cifar = _module
densenet = _module
imagenet_resnet = _module
imagenet_resnet_small = _module
preresnet = _module
res_utils = _module
resnet = _module
resnet_feature = _module
resnet_mod = _module
resnet_small = _module
resnet_small_V3 = _module
resnext = _module
vgg = _module
vgg_cifar = _module
vgg_cifar10 = _module
original_train = _module
pruning_cifar10 = _module
pruning_imagenet = _module
utils = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


from torch.autograd import Variable


import time


import random


from scipy.spatial import distance


from collections import OrderedDict


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from torch.nn import init


import math


import torch.utils.model_zoo as model_zoo


import torchvision.models


import torchvision.datasets as dset


class CifarCaffeNet(nn.Module):

    def __init__(self, num_classes):
        super(CifarCaffeNet, self).__init__()
        self.num_classes = num_classes
        self.block_1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm2d(32))
        self.block_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.AvgPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(64))
        self.block_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.AvgPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(128))
        self.classifier = nn.Linear(128 * 9, self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block_1.forward(x)
        x = self.block_2.forward(x)
        x = self.block_3.forward(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes_after_prune, planes_expand, planes_before_prune, index, bn_value, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes_after_prune, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes_after_prune)
        self.conv2 = nn.Conv2d(planes_after_prune, planes_after_prune, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes_after_prune)
        self.conv3 = nn.Conv2d(planes_after_prune, planes_expand, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes_expand)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.index = Variable(index)
        self.bn_value = bn_value

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
        residual += self.bn_value
        residual.index_add_(1, self.index, out)
        residual = self.relu(residual)
        return residual


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()
        if bottleneck:
            nDenseBlocks = int((depth - 4) / 6)
        else:
            nDenseBlocks = int((depth - 4) / 3)
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes_after_prune, planes_expand, planes_before_prune, index, bn_value, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes_after_prune, stride)
        self.bn1 = nn.BatchNorm2d(planes_after_prune)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes_after_prune, planes_after_prune)
        self.bn2 = nn.BatchNorm2d(planes_after_prune)
        self.downsample = downsample
        self.stride = stride
        self.index = Variable(index)
        self.bn_value = bn_value

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        residual += self.bn_value
        residual.index_add_(1, self.index, out)
        residual = self.relu(residual)
        return residual


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
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


class ResNet_small(nn.Module):

    def __init__(self, block, layers, index, bn_value, num_for_construct=[64, 64, 64 * 4, 128, 128 * 4, 256, 256 * 4, 512, 512 * 4], num_classes=1000):
        super(ResNet_small, self).__init__()
        self.inplanes = num_for_construct[0]
        self.conv1 = nn.Conv2d(3, num_for_construct[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_for_construct[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.index_layer1 = {key: index[key] for key in index.keys() if 'layer1' in key}
        self.index_layer2 = {key: index[key] for key in index.keys() if 'layer2' in key}
        self.index_layer3 = {key: index[key] for key in index.keys() if 'layer3' in key}
        self.index_layer4 = {key: index[key] for key in index.keys() if 'layer4' in key}
        self.bn_layer1 = {key: bn_value[key] for key in bn_value.keys() if 'layer1' in key}
        self.bn_layer2 = {key: bn_value[key] for key in bn_value.keys() if 'layer2' in key}
        self.bn_layer3 = {key: bn_value[key] for key in bn_value.keys() if 'layer3' in key}
        self.bn_layer4 = {key: bn_value[key] for key in bn_value.keys() if 'layer4' in key}
        self.layer1 = self._make_layer(block, num_for_construct[1], num_for_construct[2], 64, self.index_layer1, self.bn_layer1, layers[0])
        self.layer2 = self._make_layer(block, num_for_construct[3], num_for_construct[4], 128, self.index_layer2, self.bn_layer2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_for_construct[5], num_for_construct[6], 256, self.index_layer3, self.bn_layer3, layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_for_construct[7], num_for_construct[8], 512, self.index_layer4, self.bn_layer4, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes_after_prune, planes_expand, planes_before_prune, index, bn_layer, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes_before_prune * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes_before_prune * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes_before_prune * block.expansion))
        None
        index_block_0_dict = {key: index[key] for key in index.keys() if '0.conv3' in key}
        index_block_0_value = list(index_block_0_dict.values())[0]
        bn_layer_0_value = list(bn_layer.values())[0]
        layers = []
        layers.append(block(self.inplanes, planes_after_prune, planes_expand, planes_before_prune, index_block_0_value, bn_layer_0_value, stride, downsample))
        self.inplanes = planes_before_prune * block.expansion
        for i in range(1, blocks):
            index_block_i_dict = {key: index[key] for key in index.keys() if str(i) + '.conv3' in key}
            index_block_i_value = list(index_block_i_dict.values())[0]
            bn_layer_i = {key: bn_layer[key] for key in bn_layer.keys() if str(i) + '.bn3' in key}
            bn_layer_i_value = list(bn_layer_i.values())[0]
            layers.append(block(self.inplanes, planes_after_prune, planes_expand, planes_before_prune, index_block_i_value, bn_layer_i_value))
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


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """

    def __init__(self, inplanes, planes, index, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.inplanes = inplanes
        self.index = index

    def forward(self, x):
        residual = x
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = torch.rand(self.inplanes, basicblock.size()[1], basicblock.size()[2])
        return F.relu(out, inplace=True)


class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class CifarPreResNet(nn.Module):
    """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """

    def __init__(self, block, depth, num_classes):
        """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
        super(CifarPreResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        None
        self.num_classes = num_classes
        self.conv_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.lastact = nn.Sequential(nn.BatchNorm2d(64 * block.expansion), nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 'both_preact'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, 'normal'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.lastact(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DownsampleC(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CifarResNet(nn.Module):
    """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """

    def __init__(self, block, depth, num_classes, index, rate=[16, 16, 32, 64, 16, 32, 64]):
        """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
        super(CifarResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        self.stage_num = (depth - 2) // 3
        None
        None
        self.num_classes = num_classes
        self.rate = rate
        self.index = index
        self.conv_1_3x3 = nn.Conv2d(3, rate[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(rate[0])
        None
        self.inplanes = rate[0]
        self.stage_1 = self._make_layer(block, rate[4], rate[1], index[1:self.stage_num + 1], layer_blocks, 1)
        self.stage_2 = self._make_layer(block, rate[5], rate[2], index[self.stage_num + 1:self.stage_num * 2 + 1], layer_blocks, 2)
        self.stage_3 = self._make_layer(block, rate[6], rate[3], index[self.stage_num * 2 + 1:self.stage_num * 3 + 1], layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, planes, index, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        i = 0
        j = 2
        layers.append(block(self.inplanes, planes, index[i:j], stride, downsample))
        i += 2
        j += 2
        self.inplanes = inplanes
        None
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, index[i:j]))
            i += 2
            j += 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ResNeXtBottleneck(nn.Module):
    expansion = 4
    """
  RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
  """

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality
        self.conv_reduce = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D * C)
        self.conv_conv = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D * C)
        self.conv_expand = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """
  ResNext optimized for the Cifar dataset, as specified in
  https://arxiv.org/pdf/1611.05431.pdf
  """

    def __init__(self, block, depth, cardinality, base_width, num_classes):
        super(CifarResNeXt, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9
        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(256 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
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


defaultcfg = {(11): [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512], (13): [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512], (16): [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512], (19): [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]}


class vgg(nn.Module):

    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.cfg = cfg
        self.feature = self.make_layers(cfg, True)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(nn.Linear(cfg[-1], 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, num_classes))
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CifarCaffeNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     True),
    (DownsampleA,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownsampleC,
     lambda: ([], {'nIn': 1, 'nOut': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (DownsampleD,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'stride': 2}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetBasicblock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleLayer,
     lambda: ([], {'nChannels': 4, 'growthRate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'nChannels': 4, 'nOutChannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (vgg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     False),
]

class Test_he_y_filter_pruning_geometric_median(_paritybench_base):
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

