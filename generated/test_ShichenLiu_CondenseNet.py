import sys
_module = sys.modules[__name__]
del sys
master = _module
layers = _module
main = _module
models = _module
condensenet = _module
condensenet_converted = _module
densenet = _module
densenet_LGC = _module
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


from torch.autograd import Variable


import torch.nn.functional as F


import math


import warnings


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


from functools import reduce


class LearnedGroupConv(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, condense_factor=None, dropout_rate=0.0
        ):
        super(LearnedGroupConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor is None:
            self.condense_factor = self.groups
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
        assert self.in_channels % self.groups == 0, 'group number can not be divided by input channels'
        assert self.in_channels % self.condense_factor == 0, 'condensation factor can not be divided by input channels'
        assert self.out_channels % self.groups == 0, 'group number can not be divided by output channels'

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, None, self.conv.stride, self.conv.
            padding, self.conv.dilation, 1)

    def _check_drop(self):
        progress = LearnedGroupConv.global_progress
        delta = 0
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        if not self._reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
        if delta > 0:
            self._dropping(delta)
        return

    def _dropping(self, delta):
        weight = self.conv.weight * self.mask
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.transpose(0, 1).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, (d), :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._reach_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.sum(0).clamp(min=1e-06).sqrt()
        return weight.sum()


class CondensingLinear(nn.Module):

    def __init__(self, model, drop_rate=0.5):
        super(CondensingLinear, self).__init__()
        self.in_features = int(model.in_features * drop_rate)
        self.out_features = model.out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.register_buffer('index', torch.LongTensor(self.in_features))
        _, index = model.weight.data.abs().sum(0).sort()
        index = index[model.in_features - self.in_features:]
        self.linear.bias.data = model.bias.data.clone()
        for i in range(self.in_features):
            self.index[i] = index[i]
            self.linear.weight.data[:, (i)] = model.weight.data[:, (index[i])]

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.linear(x)
        return x


def ShuffleLayer(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class CondensingConv(nn.Module):

    def __init__(self, model):
        super(CondensingConv, self).__init__()
        self.in_channels = (model.conv.in_channels * model.groups // model.
            condense_factor)
        self.out_channels = model.conv.out_channels
        self.groups = model.groups
        self.condense_factor = model.condense_factor
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
            kernel_size=model.conv.kernel_size, padding=model.conv.padding,
            groups=self.groups, bias=False, stride=model.conv.stride)
        self.register_buffer('index', torch.LongTensor(self.in_channels))
        index = 0
        mask = model._mask.mean(-1).mean(-1)
        for i in range(self.groups):
            for j in range(model.conv.in_channels):
                if index < self.in_channels // self.groups * (i + 1) and mask[
                    i, j] == 1:
                    for k in range(self.out_channels // self.groups):
                        idx_i = int(k + i * (self.out_channels // self.groups))
                        idx_j = index % (self.in_channels // self.groups)
                        self.conv.weight.data[(idx_i), (idx_j), :, :
                            ] = model.conv.weight.data[(int(i + k * self.
                            groups)), (j), :, :]
                        self.norm.weight.data[index] = model.norm.weight.data[j
                            ]
                        self.norm.bias.data[index] = model.norm.bias.data[j]
                        self.norm.running_mean[index
                            ] = model.norm.running_mean[j]
                        self.norm.running_var[index] = model.norm.running_var[j
                            ]
                    self.index[index] = j
                    index += 1

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class CondenseLinear(nn.Module):

    def __init__(self, in_features, out_features, drop_rate=0.5):
        super(CondenseLinear, self).__init__()
        self.in_features = int(in_features * drop_rate)
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.register_buffer('index', torch.LongTensor(self.in_features))

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.linear(x)
        return x


class CondenseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1):
        super(CondenseConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, groups
            =self.groups, bias=False)
        self.register_buffer('index', torch.LongTensor(self.in_channels))
        self.index.fill_(0)

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class Conv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=
            False, groups=groups))


class _DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        self.conv_1 = LearnedGroupConv(in_channels, args.bottleneck *
            growth_rate, kernel_size=1, groups=self.group_1x1,
            condense_factor=args.condense_factor, dropout_rate=args.
            dropout_rate)
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
            kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate,
                args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):

    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet(nn.Module):

    def __init__(self, args):
        super(CondenseNet, self).__init__()
        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7
        self.features = nn.Sequential()
        self.num_features = 2 * self.growth[0]
        self.features.add_module('init_conv', nn.Conv2d(3, self.
            num_features, kernel_size=3, stride=self.init_stride, padding=1,
            bias=False))
        for i in range(len(self.stages)):
            self.add_block(i)
        self.classifier = nn.Linear(self.num_features, args.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        return

    def add_block(self, i):
        last = i == len(self.stages) - 1
        block = _DenseBlock(num_layers=self.stages[i], in_channels=self.
            num_features, growth_rate=self.growth[i], args=self.args)
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features, args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last', nn.BatchNorm2d(self.
                num_features))
            self.features.add_module('relu_last', nn.ReLU(inplace=True))
            self.features.add_module('pool_last', nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


class _DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        self.conv_1 = CondenseConv(in_channels, args.bottleneck *
            growth_rate, kernel_size=1, groups=self.group_1x1)
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
            kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate,
                args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):

    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet(nn.Module):

    def __init__(self, args):
        super(CondenseNet, self).__init__()
        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7
        self.features = nn.Sequential()
        self.num_features = 2 * self.growth[0]
        self.features.add_module('init_conv', nn.Conv2d(3, self.
            num_features, kernel_size=3, stride=self.init_stride, padding=1,
            bias=False))
        for i in range(len(self.stages)):
            self.add_block(i)
        self.classifier = CondenseLinear(self.num_features, args.
            num_classes, 0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def add_block(self, i):
        last = i == len(self.stages) - 1
        block = _DenseBlock(num_layers=self.stages[i], in_channels=self.
            num_features, growth_rate=self.growth[i], args=self.args)
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features, args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last', nn.BatchNorm2d(self.
                num_features))
            self.features.add_module('relu_last', nn.ReLU(inplace=True))
            self.features.add_module('pool_last', nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


class _DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        self.conv_1 = Conv(in_channels, args.bottleneck * growth_rate,
            kernel_size=1, groups=self.group_1x1)
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
            kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate,
                args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):

    def __init__(self, in_channels, out_channels, args):
        super(_Transition, self).__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size=1, groups=
            args.group_1x1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


def make_divisible(x, y):
    return int((x // y + 1) * y if x % y else x)


class DenseNet(nn.Module):

    def __init__(self, args):
        super(DenseNet, self).__init__()
        self.stages = args.stages
        self.growth = args.growth
        self.reduction = args.reduction
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7
        self.features = nn.Sequential()
        self.num_features = 2 * self.growth[0]
        self.features.add_module('init_conv', nn.Conv2d(3, self.
            num_features, kernel_size=3, stride=self.init_stride, padding=1,
            bias=False))
        for i in range(len(self.stages)):
            self.add_block(i)
        self.classifier = nn.Linear(self.num_features, args.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def add_block(self, i):
        last = i == len(self.stages) - 1
        block = _DenseBlock(num_layers=self.stages[i], in_channels=self.
            num_features, growth_rate=self.growth[i], args=self.args)
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            out_features = make_divisible(math.ceil(self.num_features *
                self.reduction), self.args.group_1x1)
            trans = _Transition(in_channels=self.num_features, out_channels
                =out_features, args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
            self.num_features = out_features
        else:
            self.features.add_module('norm_last', nn.BatchNorm2d(self.
                num_features))
            self.features.add_module('relu_last', nn.ReLU(inplace=True))
            self.features.add_module('pool_last', nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


class _DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        self.conv_1 = LearnedGroupConv(in_channels, args.bottleneck *
            growth_rate, kernel_size=1, groups=self.group_1x1,
            condense_factor=args.condense_factor, dropout_rate=args.
            dropout_rate)
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
            kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate,
                args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):

    def __init__(self, in_channels, out_channels, args):
        super(_Transition, self).__init__()
        self.conv = LearnedGroupConv(in_channels, out_channels, kernel_size
            =1, groups=args.group_1x1, condense_factor=args.condense_factor)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet_LGC(nn.Module):

    def __init__(self, args):
        super(DenseNet_LGC, self).__init__()
        self.stages = args.stages
        self.growth = args.growth
        self.reduction = args.reduction
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7
        self.features = nn.Sequential()
        self.num_features = 2 * self.growth[0]
        self.features.add_module('init_conv', nn.Conv2d(3, self.
            num_features, kernel_size=3, stride=self.init_stride, padding=1,
            bias=False))
        for i in range(len(self.stages)):
            self.add_block(i)
        self.classifier = nn.Linear(self.num_features, args.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def add_block(self, i):
        last = i == len(self.stages) - 1
        block = _DenseBlock(num_layers=self.stages[i], in_channels=self.
            num_features, growth_rate=self.growth[i], args=self.args)
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            out_features = make_divisible(math.ceil(self.num_features *
                self.reduction), self.args.group_1x1)
            trans = _Transition(in_channels=self.num_features, out_channels
                =out_features, args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
            self.num_features = out_features
        else:
            self.features.add_module('norm_last', nn.BatchNorm2d(self.
                num_features))
            self.features.add_module('relu_last', nn.ReLU(inplace=True))
            self.features.add_module('pool_last', nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ShichenLiu_CondenseNet(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(LearnedGroupConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(CondenseConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Conv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(_DenseLayer(*[], **{'in_channels': 4, 'growth_rate': 4, 'args': _mock_config(group_1x1=4, group_3x3=4, bottleneck=4, condense_factor=4, dropout_rate=0.5)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(_DenseBlock(*[], **{'num_layers': 1, 'in_channels': 4, 'growth_rate': 4, 'args': _mock_config(group_1x1=4, group_3x3=4, bottleneck=4, condense_factor=4, dropout_rate=0.5)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(_Transition(*[], **{'in_channels': 4, 'out_channels': 4, 'args': _mock_config(group_1x1=4, condense_factor=4)}), [torch.rand([4, 4, 4, 4])], {})

