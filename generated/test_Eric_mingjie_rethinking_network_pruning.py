import sys
_module = sys.modules[__name__]
del sys
compute_flops = _module
main = _module
main_B = _module
main_E = _module
main_finetune = _module
models = _module
resnet = _module
vgg = _module
res110prune = _module
res56prune = _module
vggprune = _module
lottery_res110prune = _module
lottery_resprune = _module
lottery_vggprune = _module
main = _module
main_lottery = _module
main_scratch_mask = _module
resnet = _module
vgg = _module
cifar = _module
cifar_prune_iterative = _module
cifar_scratch_no_longer = _module
lottery_ticket = _module
alexnet = _module
densenet = _module
preresnet = _module
resnet = _module
resnext = _module
vgg = _module
wrn = _module
utils = _module
eval = _module
logger = _module
misc = _module
visualize = _module
compute_flops = _module
denseprune = _module
main = _module
main_B = _module
main_E = _module
main_finetune = _module
channel_selection = _module
densenet = _module
preresnet = _module
vgg = _module
resprune = _module
vggprune = _module
compute_flops = _module
pruning_cifar10_pretrain = _module
pruning_cifar10_resnet = _module
pruning_resnet_longer_scratch = _module
pruning_resnet_scratch = _module
cifar = _module
cifar_B = _module
cifar_E = _module
cifar_finetune = _module
cifar_prune = _module
count_flops = _module
alexnet = _module
densenet = _module
preresnet = _module
resnet = _module
vgg = _module
misc = _module
visualize = _module
compute_flops = _module
main_B = _module
main_E = _module
main_finetune = _module
prune = _module
resnet = _module
compute_flops = _module
main = _module
main_B = _module
main_E = _module
main_finetune = _module
prune = _module
vgg = _module
compute_flops = _module
main_B = _module
main_E = _module
channel_selection = _module
resnet = _module
resnet_2x = _module
vgg_5x = _module
compute_flops = _module
main_B = _module
main_E = _module
thinetconv = _module
thinetresnet = _module
thinetvgg = _module
compute_flops = _module
main_B = _module
main_E = _module
main_finetune = _module
prune = _module

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


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import torch.optim as optim


import math


from functools import partial


import random


import torch.backends.cudnn as cudnn


import torch.utils.data as data


from torch.nn import init


import torch.utils.model_zoo as model_zoo


import torch.nn.init as init


import torch.nn.parallel


import torch.distributed as dist


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import warnings


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
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


def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2, 2)(x)
    zero_pads = torch.Tensor(x.size(0), planes - x.size(1), x.size(2), x.
        size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([x.data, zero_pads], dim=1))
    return out


class ResNet(nn.Module):

    def __init__(self, depth, dataset='cifar10', cfg=None):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        block = BasicBlock
        if cfg == None:
            cfg = [[16] * n, [32] * n, [64] * n]
            cfg = [item for sub_list in cfg for item in sub_list]
        self.cfg = cfg
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[n:2 * n], stride=2
            )
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[2 * n:3 * n],
            stride=2)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes *
                block.expansion)
        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


defaultcfg = {(11): [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 
    512], (13): [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 
    512, 512], (16): [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 
    512, 512, 'M', 512, 512, 512], (19): [64, 64, 'M', 128, 128, 'M', 256, 
    256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]}


class vgg(nn.Module):

    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None
        ):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.cfg = cfg
        self.feature = self.make_layers(cfg, True)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(nn.Linear(cfg[-1], 512), nn.
            BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512,
            num_classes))
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1,
                    bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
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


class ResNet(nn.Module):

    def __init__(self, depth, dataset='cifar10', cfg=None):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        block = BasicBlock
        if cfg == None:
            cfg = [[16] * n, [32] * n, [64] * n]
            cfg = [item for sub_list in cfg for item in sub_list]
        self.cfg = cfg
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[n:2 * n], stride=2
            )
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[2 * n:3 * n],
            stride=2)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes *
                block.expansion)
        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class vgg(nn.Module):

    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None
        ):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.cfg = cfg
        self.feature = self.make_layers(cfg, True)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(nn.Linear(cfg[-1], 512), nn.
            BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512,
            num_classes))
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1,
                    bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
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


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11,
            stride=4, padding=5), nn.ReLU(inplace=True), nn.MaxPool2d(
            kernel_size=2, stride=2), nn.Conv2d(64, 192, kernel_size=5,
            padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2,
            stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=
            1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2,
            stride=2))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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

    def __init__(self, depth=22, block=BasicBlock, dropRate=0, num_classes=
        10, growthRate=12, compressionRate=1):
        super(DenseNet, self).__init__()
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6
        n = int(n)
        self.growthRate = growthRate
        self.dropRate = dropRate
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
            bias=False)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class PreResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(PreResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        block = Bottleneck if depth >= 44 else BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
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
                block.expansion, kernel_size=1, stride=stride, bias=False))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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

    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        block = Bottleneck if depth >= 54 else BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality,
        widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = cardinality * out_channels // widen_factor
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

    def __init__(self, cardinality, depth, num_classes, widen_factor=4,
        dropRate=0):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            num_classes: number of classes
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor,
            256 * self.widen_factor]
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(1024, num_classes)
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
                    widen_factor))
            else:
                block.add_module(name_, ResNeXtBottleneck(out_channels,
                    out_channels, 1, self.cardinality, self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, 1024)
        return self.classifier(x)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
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

    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 *
            widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
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


class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """

    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
	    """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
		"""
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().
            numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, (selected_index), :, :]
        return output


class BasicBlock(nn.Module):

    def __init__(self, inplanes, cfg, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg, growthRate, kernel_size=3, padding=1,
            bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, inplanes, outplanes, cfg):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class densenet(nn.Module):

    def __init__(self, depth=40, dropRate=0, dataset='cifar10', growthRate=
        12, compressionRate=1, cfg=None):
        super(densenet, self).__init__()
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3
        block = BasicBlock
        self.growthRate = growthRate
        self.dropRate = dropRate
        if cfg == None:
            cfg = []
            start = growthRate * 2
            for i in range(3):
                cfg.append([(start + 12 * i) for i in range(n + 1)])
                start += growthRate * 12
            cfg = [item for sub_list in cfg for item in sub_list]
        assert len(cfg
            ) == 3 * n + 3, 'length of config variable cfg should be 3n+3'
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
            bias=False)
        self.dense1 = self._make_denseblock(block, n, cfg[0:n])
        self.trans1 = self._make_transition(compressionRate, cfg[n])
        self.dense2 = self._make_denseblock(block, n, cfg[n + 1:2 * n + 1])
        self.trans2 = self._make_transition(compressionRate, cfg[2 * n + 1])
        self.dense3 = self._make_denseblock(block, n, cfg[2 * n + 2:3 * n + 2])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.select = channel_selection(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, cfg):
        layers = []
        assert blocks == len(cfg), 'Length of the cfg parameter is not right.'
        for i in range(blocks):
            layers.append(block(self.inplanes, cfg=cfg[i], growthRate=self.
                growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate
        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, cfg):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class resnet(nn.Module):

    def __init__(self, depth=164, dataset='cifar10', cfg=None):
        super(resnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
        n = (depth - 2) // 9
        block = Bottleneck
        if cfg is None:
            cfg = [[16, 16, 16], [64, 16, 16] * (n - 1), [64, 32, 32], [128,
                32, 32] * (n - 1), [128, 64, 64], [256, 64, 64] * (n - 1),
                [256]]
            cfg = [item for sub_list in cfg for item in sub_list]
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:3 * n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[3 * n:6 * n],
            stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[6 * n:9 * n],
            stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False))
        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride,
            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3 * i:3 * (i + 1)]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class vgg(nn.Module):

    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None
        ):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.feature = self.make_layers(cfg, True)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1,
                    bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
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


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11,
            stride=4, padding=5), nn.ReLU(inplace=True), nn.MaxPool2d(
            kernel_size=2, stride=2), nn.Conv2d(64, 192, kernel_size=5,
            padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2,
            stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=
            1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2,
            stride=2))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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

    def __init__(self, depth=22, block=BasicBlock, dropRate=0, num_classes=
        10, growthRate=12, compressionRate=1):
        super(DenseNet, self).__init__()
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6
        n = int(n)
        self.growthRate = growthRate
        self.dropRate = dropRate
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
            bias=False)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class PreResNet(nn.Module):

    def __init__(self, depth, num_classes=10):
        super(PreResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 9
        block = Bottleneck if depth >= 44 else BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
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
                block.expansion, kernel_size=1, stride=stride, bias=False))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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

    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        block = Bottleneck if depth >= 54 else BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
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

    def __init__(self, block, layers, cfg=None, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if cfg == None:
            cfg = [[64] * layers[0], [128] * layers[1], [256] * layers[2], 
                [512] * layers[3]]
            cfg = [item for sub_list in cfg for item in sub_list]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        count = 0
        self.layer1 = self._make_layer(block, 64, layers[0], cfg[:layers[0]])
        count += layers[0]
        self.layer2 = self._make_layer(block, 128, layers[1], cfg[count:
            count + layers[1]], stride=2)
        count += layers[1]
        self.layer3 = self._make_layer(block, 256, layers[2], cfg[count:
            count + layers[2]], stride=2)
        count += layers[2]
        self.layer4 = self._make_layer(block, 512, layers[3], cfg[count:
            count + layers[3]], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(cfg[-1], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))
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


class VGG(nn.Module):

    def __init__(self, features, cfg, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(cfg[0] * 7 * 7, cfg[1]),
            nn.BatchNorm1d(cfg[1]), nn.ReLU(True), nn.Linear(cfg[1], cfg[2]
            ), nn.BatchNorm1d(cfg[2]), nn.ReLU(True), nn.Linear(cfg[2],
            num_classes))
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
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()


class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """

    def __init__(self, num_channels, mask):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))
        assert len(mask) == num_channels
        mask = torch.from_numpy(mask)
        self.indexes.data.mul_(mask)

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().
            numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, (selected_index), :, :]
        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
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

    def __init__(self, block, layers, cfg, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, cfg[0:9], 64, layers[0])
        self.layer2 = self._make_layer(block, cfg[9:21], 128, layers[1],
            stride=2)
        self.layer3 = self._make_layer(block, cfg[21:39], 256, layers[2],
            stride=2)
        self.layer4 = self._make_layer(block, cfg[39:48], 512, layers[3],
            stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, cfg, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, cfg[:3], stride, downsample)
            )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3 * i:3 * (i + 1)]))
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, mask, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.selection = channel_selection(inplanes, mask)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.selection(x)
        out = self.conv1(out)
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

    def __init__(self, block, layers, cfg, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.count = 0
        self.layer1 = self._make_layer(block, cfg[0:9], 64, layers[0])
        self.layer2 = self._make_layer(block, cfg[9:21], 128, layers[1],
            stride=2)
        self.layer3 = self._make_layer(block, cfg[21:39], 256, layers[2],
            stride=2)
        self.layer4 = self._make_layer(block, cfg[39:48], 512, layers[3],
            stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, cfg, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        mask = filter_index[self.count]
        layers.append(block(self.inplanes, planes, cfg[:3], mask, stride,
            downsample))
        self.count += 1
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            mask = filter_index[self.count]
            layers.append(block(self.inplanes, planes, cfg[3 * i:3 * (i + 1
                )], mask))
            self.count += 1
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


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.
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
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.
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
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
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

    def __init__(self, block, layers, cfg, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, cfg[0:9], 64, layers[0])
        self.layer2 = self._make_layer(block, cfg[9:21], 128, layers[1],
            stride=2)
        self.layer3 = self._make_layer(block, cfg[21:39], 256, layers[2],
            stride=2)
        self.layer4 = self._make_layer(block, cfg[39:48], 512, layers[3],
            stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, cfg, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, cfg[:3], stride, downsample)
            )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3 * i:3 * (i + 1)]))
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


class VGG(nn.Module):

    def __init__(self, features, cfg, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(cfg, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(14, 2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Eric_mingjie_rethinking_network_pruning(_paritybench_base):
    pass

    def test_000(self):
        self._check(Transition(*[], **{'inplanes': 4, 'outplanes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ResNeXtBottleneck(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1, 'cardinality': 4, 'widen_factor': 4}), [torch.rand([4, 4, 4, 4])], {})
