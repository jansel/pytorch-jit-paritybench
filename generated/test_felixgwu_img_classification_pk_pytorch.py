import sys
_module = sys.modules[__name__]
del sys
args = _module
config = _module
dataloader = _module
getbest = _module
main = _module
models = _module
densenet = _module
preact_resnet = _module
resnet = _module
resnet_kuangliu = _module
train = _module
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


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.nn.functional as F


from collections import OrderedDict


import math


from torch import nn


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        if bn_size > 0:
            self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                growth_rate, kernel_size=1, stride=1, bias=False)),
            self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu.2', nn.ReLU(inplace=True)),
            self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate,
                growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        else:
            self.add_module('conv.1', nn.Conv2d(num_input_features,
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
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=12, block_config=(16, 16, 16),
        compression=0.5, num_init_features=24, bn_size=4, drop_rate=0,
        avgpool_size=8, num_classes=10):
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between '
        self.avgpool_size = avgpool_size
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, kernel_size=3, stride=1, padding=1, bias=
            False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0',
            nn.ReLU(inplace=True))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features
            .size(0), -1)
        out = self.classifier(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlockWithDeathRate(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, death_rate=0.0,
        downsample=None):
        super(BasicBlockWithDeathRate, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.death_rate = death_rate

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training or torch.rand(1)[0] >= self.death_rate:
            residual = self.conv1(residual)
            residual = self.bn1(residual)
            residual = self.relu1(residual)
            residual = self.conv2(residual)
            residual = self.bn2(residual)
            if self.training:
                residual /= 1.0 - self.death_rate
            x = x + residual
            x = self.relu2(x)
        return x


class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)


class ResNetCifar(nn.Module):
    """Small ResNet for CIFAR & SVHN
    death_rates: death_rates of each block except for the first and
                 the last block
    """

    def __init__(self, depth, death_rates=None, block=
        BasicBlockWithDeathRate, num_classes=10):
        assert (depth - 2) % 6 == 0, 'depth should be one of 6N+2'
        super(ResNetCifar, self).__init__()
        n = (depth - 2) // 6
        assert death_rates is None or len(death_rates) == 3 * n
        if death_rates is None:
            death_rates = [0.0] * (3 * n)
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, death_rates[:n])
        self.layer2 = self._make_layer(block, 32, death_rates[n:2 * n],
            stride=2)
        self.layer3 = self._make_layer(block, 64, death_rates[2 * n:], stride=2
            )
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, death_rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.
                expansion, stride)
        layers = [block(self.inplanes, planes, stride, downsample=
            downsample, death_rate=death_rates[0])]
        self.inplanes = planes * block.expansion
        for death_rate in death_rates[1:]:
            layers.append(block(self.inplanes, planes, death_rate=death_rate))
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_felixgwu_img_classification_pk_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(PreActBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(PreActBottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(DownsampleB(*[], **{'nIn': 4, 'nOut': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Bottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

