import sys
_module = sys.modules[__name__]
del sys
condensenet_converted = _module
make_cifar_c = _module
make_imagenet_64_c = _module
make_imagenet_c = _module
make_imagenet_c_inception = _module
make_tinyimagenet_c = _module
densenet_cosine_264_k48 = _module
imagenet_c = _module
corruptions = _module
setup = _module
layers = _module
test = _module
make_cifar_p = _module
make_imagenet_64_p = _module
make_imagenet_p = _module
make_imagenet_p_inception = _module
make_tinyimagenet_p = _module
densenet_cosine_264_k48 = _module
resnext_101_32x4d = _module
resnext_101_64x4d = _module
resnext_50_32x4d = _module
test = _module
video_loader = _module
augment = _module
densenet = _module
msdnet = _module
resnet = _module
resnext = _module
shake_shake = _module
wrn = _module
train = _module
train = _module
test = _module
train = _module

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


import math


from functools import reduce


import torch.nn.functional as F


from torch.autograd import Variable as V


import torch.backends.cudnn as cudnn


import torch.utils.model_zoo as model_zoo


import numpy as np


import collections


from scipy.stats import rankdata


from torch.autograd import Function


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


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


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
        if not self._at_stage(stage):
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

    def _at_stage(self, stage):
        return (self._stage == stage).all()

    @property
    def lasso_loss(self):
        if self._at_stage(self.groups - 1):
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


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super(BasicBlock, self).__init__()
        self.drop_rate = drop_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        return torch.cat([x, y], dim=1)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super(BottleneckBlock, self).__init__()
        self.drop_rate = drop_rate
        bottleneck_channels = out_channels * 4
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        return torch.cat([x, y], dim=1)


class TransitionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super(TransitionBlock, self).__init__()
        self.drop_rate = drop_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training,
                inplace=False)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class DenseNet(nn.Module):

    def __init__(self, config):
        super(DenseNet, self).__init__()
        input_shape = config['input_shape']
        n_classes = config['n_classes']
        block_type = config['block_type']
        depth = config['depth']
        self.growth_rate = config['growth_rate']
        self.drop_rate = config['drop_rate']
        self.compression_rate = config['compression_rate']
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 4) // 3
            assert n_blocks_per_stage * 3 + 4 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 4) // 6
            assert n_blocks_per_stage * 6 + 4 == depth
        in_channels = [2 * self.growth_rate]
        for index in range(3):
            denseblock_out_channels = int(in_channels[-1] + 
                n_blocks_per_stage * self.growth_rate)
            if index < 2:
                transitionblock_out_channels = int(denseblock_out_channels *
                    self.compression_rate)
            else:
                transitionblock_out_channels = denseblock_out_channels
            in_channels.append(transitionblock_out_channels)
        self.conv = nn.Conv2d(input_shape[1], in_channels[0], kernel_size=3,
            stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(in_channels[0], n_blocks_per_stage,
            block, True)
        self.stage2 = self._make_stage(in_channels[1], n_blocks_per_stage,
            block, True)
        self.stage3 = self._make_stage(in_channels[2], n_blocks_per_stage,
            block, False)
        self.bn = nn.BatchNorm2d(in_channels[3])
        self.feature_size = self._forward_conv(torch.zeros(*input_shape)).view(
            -1).shape[0]
        self.fc = nn.Linear(self.feature_size, n_classes)
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, n_blocks, block, add_transition_block):
        stage = nn.Sequential()
        for index in range(n_blocks):
            stage.add_module('block{}'.format(index + 1), block(in_channels +
                index * self.growth_rate, self.growth_rate, self.drop_rate))
        if add_transition_block:
            in_channels = int(in_channels + n_blocks * self.growth_rate)
            out_channels = int(in_channels * self.compression_rate)
            stage.add_module('transition', TransitionBlock(in_channels,
                out_channels, self.drop_rate))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class _DynamicInputDenseBlock(nn.Module):

    def __init__(self, conv_modules, debug):
        super(_DynamicInputDenseBlock, self).__init__()
        self.conv_modules = conv_modules
        self.debug = debug

    def forward(self, x):
        """
        Use the first element as raw input, and stream the rest of
        the inputs through the list of modules, then apply concatenation.
        expect x to be [identity, first input, second input, ..]
        and len(x) - len(self.conv_modules) = 1 for identity

        :param x: Input
        :return: Concatenation of the input with 1 or more module outputs
        """
        if self.debug:
            for i, t in enumerate(x):
                None
        out = x[0]
        for calc, m in enumerate(self.conv_modules):
            out = torch.cat([out, m(x[calc + 1])], 1)
            if self.debug:
                None
                None
                None
        return out


def get_conv_params(use_gcn, args):
    """
    Calculates and returns the convulotion parameters

    :param use_gcn: flag to use GCN or not
    :param args: user defined arguments
    :return: convolution type, kernel size and padding
    """
    if use_gcn:
        GCN.share_weights = args.msd_share_weights
        conv_l = GCN
        ks = args.msd_gcn_kernel
    else:
        conv_l = nn.Conv2d
        ks = args.msd_kernel
    pad = int(math.floor(ks / 2))
    return conv_l, ks, pad


class MSDLayer(nn.Module):

    def __init__(self, in_channels, out_channels, in_scales, out_scales,
        orig_scales, args):
        """
        Creates a regular/transition MSDLayer. this layer uses DenseNet like concatenation on each scale,
        and performs spatial reduction between scales. if input and output scales are different, than this
        class creates a transition layer and the first layer (with the largest spatial size) is dropped.

        :param current_channels: number of input channels
        :param in_scales: number of input scales
        :param out_scales: number of output scales
        :param orig_scales: number of scales in the first layer of the MSDNet
        :param args: other arguments
        """
        super(MSDLayer, self).__init__()
        self.current_channels = in_channels
        self.out_channels = out_channels
        self.in_scales = in_scales
        self.out_scales = out_scales
        self.orig_scales = orig_scales
        self.args = args
        self.bottleneck = args.msd_bottleneck
        self.bottleneck_factor = args.msd_bottleneck_factor
        self.growth_factor = self.args.msd_growth_factor
        self.debug = self.args.debug
        self.use_gcn = args.msd_all_gcn
        self.conv_l, self.ks, self.pad = get_conv_params(self.use_gcn, args)
        self.to_drop = in_scales - out_scales
        self.dropped = orig_scales - out_scales
        self.subnets = self.get_subnets()

    def get_subnets(self):
        """
        Builds the different scales of the MSD network layer.

        :return: A list of scale modules
        """
        subnets = nn.ModuleList()
        if self.to_drop:
            in_channels1 = self.current_channels * self.growth_factor[self.
                dropped - 1]
            in_channels2 = self.current_channels * self.growth_factor[self.
                dropped]
            out_channels = self.out_channels * self.growth_factor[self.dropped]
            bn_width1 = self.bottleneck_factor[self.dropped - 1]
            bn_width2 = self.bottleneck_factor[self.dropped]
            subnets.append(self.build_down_densenet(in_channels1,
                in_channels2, out_channels, self.bottleneck, bn_width1,
                bn_width2))
        else:
            in_channels = self.current_channels * self.growth_factor[self.
                dropped]
            out_channels = self.out_channels * self.growth_factor[self.dropped]
            bn_width = self.bottleneck_factor[self.dropped]
            subnets.append(self.build_densenet(in_channels, out_channels,
                self.bottleneck, bn_width))
        for scale in range(1, self.out_scales):
            in_channels1 = self.current_channels * self.growth_factor[self.
                dropped + scale - 1]
            in_channels2 = self.current_channels * self.growth_factor[self.
                dropped + scale]
            out_channels = self.out_channels * self.growth_factor[self.
                dropped + scale]
            bn_width1 = self.bottleneck_factor[self.dropped + scale - 1]
            bn_width2 = self.bottleneck_factor[self.dropped + scale]
            subnets.append(self.build_down_densenet(in_channels1,
                in_channels2, out_channels, self.bottleneck, bn_width1,
                bn_width2))
        return subnets

    def build_down_densenet(self, in_channels1, in_channels2, out_channels,
        bottleneck, bn_width1, bn_width2):
        """
        Builds a scale sub-network for scales 2 and up.

        :param in_channels1: number of same scale input channels
        :param in_channels2: number of upper scale input channels
        :param out_channels: number of output channels
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width1: The first input width of the bottleneck factor
        :param bn_width2: The first input width of the bottleneck factor
        :return: A scale module
        """
        conv_module1 = self.convolve(in_channels1, int(out_channels / 2),
            'down', bottleneck, bn_width1)
        conv_module2 = self.convolve(in_channels2, int(out_channels / 2),
            'normal', bottleneck, bn_width2)
        conv_modules = [conv_module1, conv_module2]
        return _DynamicInputDenseBlock(nn.ModuleList(conv_modules), self.debug)

    def build_densenet(self, in_channels, out_channels, bottleneck, bn_width):
        """
        Builds a scale sub-network for the first layer

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width: The width of the bottleneck factor
        :return: A scale module
        """
        conv_module = self.convolve(in_channels, out_channels, 'normal',
            bottleneck, bn_width)
        return _DynamicInputDenseBlock(nn.ModuleList([conv_module]), self.debug
            )

    def convolve(self, in_channels, out_channels, conv_type, bottleneck,
        bn_width=4):
        """
        Doing the main convolution of a specific scale in the
        MSD network

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param conv_type: convolution type
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width: The width of the bottleneck factor
        :return: A Sequential module of the main convolution
        """
        conv = nn.Sequential()
        tmp_channels = in_channels
        if bottleneck:
            tmp_channels = int(min([in_channels, bn_width * out_channels]))
            conv.add_module('Bottleneck_1x1', nn.Conv2d(in_channels,
                tmp_channels, kernel_size=1, stride=1, padding=0))
            conv.add_module('Bottleneck_BN', nn.BatchNorm2d(tmp_channels))
            conv.add_module('Bottleneck_ReLU', nn.ReLU(inplace=True))
        if conv_type == 'normal':
            conv.add_module('Spatial_forward', self.conv_l(tmp_channels,
                out_channels, kernel_size=self.ks, stride=1, padding=self.pad))
        elif conv_type == 'down':
            conv.add_module('Spatial_down', self.conv_l(tmp_channels,
                out_channels, kernel_size=self.ks, stride=2, padding=self.pad))
        else:
            raise NotImplementedError
        conv.add_module('BN_out', nn.BatchNorm2d(out_channels))
        conv.add_module('ReLU_out', nn.ReLU(inplace=True))
        return conv

    def forward(self, x):
        cur_input = []
        outputs = []
        if self.to_drop:
            for scale in range(0, self.out_scales):
                last_same_scale = x[self.to_drop + scale]
                last_upper_scale = x[self.to_drop + scale - 1]
                cur_input.append([last_same_scale, last_upper_scale,
                    last_same_scale])
        else:
            cur_input.append([x[0], x[0]])
            for scale in range(1, self.out_scales):
                last_same_scale = x[scale]
                last_upper_scale = x[scale - 1]
                cur_input.append([last_same_scale, last_upper_scale,
                    last_same_scale])
        for scale in range(0, self.out_scales):
            outputs.append(self.subnets[scale](cur_input[scale]))
        return outputs


class MSDFirstLayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_scales, args):
        """
        Creates the first layer of the MSD network, which takes
        an input tensor (image) and generates a list of size num_scales
        with deeper features with smaller (spatial) dimensions.

        :param in_channels: number of input channels to the first layer
        :param out_channels: number of output channels in the first scale
        :param num_scales: number of output scales in the first layer
        :param args: other arguments
        """
        super(MSDFirstLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.args = args
        self.use_gcn = args.msd_gcn
        self.conv_l, self.ks, self.pad = get_conv_params(self.use_gcn, args)
        if self.use_gcn:
            None
        else:
            None
        self.subnets = self.create_modules()

    def create_modules(self):
        modules = nn.ModuleList()
        if 'cifar' in self.args.data:
            current_channels = int(self.out_channels * self.args.
                msd_growth_factor[0])
            current_m = nn.Sequential(self.conv_l(self.in_channels,
                current_channels, kernel_size=self.ks, stride=1, padding=
                self.pad), nn.BatchNorm2d(current_channels), nn.ReLU(
                inplace=True))
            modules.append(current_m)
        else:
            raise NotImplementedError
        for scale in range(1, self.num_scales):
            out_channels = int(self.out_channels * self.args.
                msd_growth_factor[scale])
            current_m = nn.Sequential(self.conv_l(current_channels,
                out_channels, kernel_size=self.ks, stride=2, padding=self.
                pad), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            current_channels = out_channels
            modules.append(current_m)
        return modules

    def forward(self, x):
        output = [None] * self.num_scales
        current_input = x
        for scale in range(0, self.num_scales):
            if scale > 0:
                current_input = output[scale - 1]
            output[scale] = self.subnets[scale](current_input)
        return output


class Transition(nn.Sequential):

    def __init__(self, channels_in, channels_out, out_scales, offset,
        growth_factor, args):
        """
        Performs 1x1 convolution to increase channels size after reducing a spatial size reduction
        in transition layer.

        :param channels_in: channels before the transition
        :param channels_out: channels after reduction
        :param out_scales: number of scales after the transition
        :param offset: gap between original number of scales to out_scales
        :param growth_factor: densenet channel growth factor
        :return: A Parallel trainable array with the scales after channel
                 reduction
        """
        super(Transition, self).__init__()
        self.args = args
        self.scales = nn.ModuleList()
        for i in range(0, out_scales):
            cur_in = channels_in * growth_factor[offset + i]
            cur_out = channels_out * growth_factor[offset + i]
            self.scales.append(self.conv1x1(cur_in, cur_out))

    def conv1x1(self, in_channels, out_channels):
        """
        Inner function to define the basic operation

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :return: A Sequential module to perform 1x1 convolution
        """
        scale = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True))
        return scale

    def forward(self, x):
        """
        Propegate output through different scales.

        :param x: input to the transition layer
        :return: list of scales' outputs
        """
        if self.args.debug:
            None
        output = []
        for scale, scale_net in enumerate(self.scales):
            if self.args.debug:
                None
                None
            output.append(scale_net(x[scale]))
        return output


class CifarClassifier(nn.Module):

    def __init__(self, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """
        super(CifarClassifier, self).__init__()
        self.inner_channels = 128
        self.features = nn.Sequential(nn.Conv2d(num_channels, self.
            inner_channels, kernel_size=3, stride=2, padding=1), nn.
            BatchNorm2d(self.inner_channels), nn.ReLU(inplace=True), nn.
            Conv2d(self.inner_channels, self.inner_channels, kernel_size=3,
            stride=2, padding=1), nn.BatchNorm2d(self.inner_channels), nn.
            ReLU(inplace=True), nn.AvgPool2d(2, 2))
        self.classifier = nn.Linear(self.inner_channels, num_classes)

    def forward(self, x):
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """
        x = self.features(x)
        x = x.view(x.size(0), self.inner_channels)
        x = self.classifier(x)
        return x


class GCN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1,
        padding=1):
        """
        Global convolutional network module implementation

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of conv kernel
        :param stride: stride to use in the conv parts
        :param padding: padding to use in the conv parts
        :param share_weights: use shared weights for every side of GCN
        """
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size=(
            kernel_size, 1), padding=(padding, 0), stride=(stride, 1))
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,
            kernel_size), padding=(0, padding), stride=(1, stride))
        self.conv_r1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,
            kernel_size), padding=(0, padding), stride=(1, stride))
        self.conv_r2 = nn.Conv2d(out_channels, out_channels, kernel_size=(
            kernel_size, 1), padding=(padding, 0), stride=(stride, 1))

    def forward(self, x):
        if GCN.share_weights:
            self.conv_l1.shared = 2
            self.conv_l2.shared = 2
            xt = x.transpose(2, 3)
            xl = self.conv_l1(x)
            xl = self.conv_l2(xl)
            xrt = self.conv_l1(xt)
            xrt = self.conv_l2(xrt)
            xr = xrt.transpose(2, 3)
        else:
            xl = self.conv_l1(x)
            xl = self.conv_l2(xl)
            xr = self.conv_r1(x)
            xr = self.conv_r2(xr)
        return xl + xr


class MSDNet(nn.Module):

    def __init__(self, args):
        """
        The main module for Multi Scale Dense Network.
        It holds the different blocks with layers and classifiers of the MSDNet layers

        :param args: Network argument
        """
        super(MSDNet, self).__init__()
        self.args = args
        self.base = self.args.msd_base
        self.step = self.args.msd_step
        self.step_mode = self.args.msd_stepmode
        self.msd_prune = self.args.msd_prune
        self.num_blocks = self.args.msd_blocks
        self.reduction_rate = self.args.reduction
        self.growth = self.args.msd_growth
        self.growth_factor = args.msd_growth_factor
        self.bottleneck = self.args.msd_bottleneck
        self.bottleneck_factor = args.msd_bottleneck_factor
        if args.data in ['cifar10', 'cifar100']:
            self.image_channels = 3
            self.num_channels = 32
            self.num_scales = 3
            self.num_classes = int(args.data.strip('cifar'))
        else:
            raise NotImplementedError
        None
        self.num_layers, self.steps = self.calc_steps()
        None
        self.cur_layer = 1
        self.cur_transition_layer = 1
        self.subnets = nn.ModuleList(self.build_modules(self.num_channels))
        for m in self.subnets:
            self.init_weights(m)
            if hasattr(m, '__iter__'):
                for sub_m in m:
                    self.init_weights(sub_m)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def calc_steps(self):
        """Calculates the number of layers required in each
        Block and the total number of layers, according to
        the step and stepmod.

        :return: number of total layers and list of layers/steps per blocks
        """
        steps = [None] * self.num_blocks
        steps[0] = num_layers = self.base
        for i in range(1, self.num_blocks):
            steps[i] = self.step_mode == 'even' and self.step or self.step * (i
                 - 1) + 1
            num_layers += steps[i]
        return num_layers, steps

    def build_modules(self, num_channels):
        """Builds all blocks and classifiers and add it
        into an array in the order of the format:
        [[block]*num_blocks [classifier]*num_blocks]
        where the i'th block corresponds to the (i+num_block) classifier.

        :param num_channels: number of input channels
        :return: An array with all blocks and classifiers
        """
        modules = [None] * self.num_blocks * 2
        for i in range(0, self.num_blocks):
            None
            modules[i], num_channels = self.create_block(num_channels, i)
            channels_in_last_layer = num_channels * self.growth_factor[self
                .num_scales]
            modules[i + self.num_blocks] = CifarClassifier(
                channels_in_last_layer, self.num_classes)
        return modules

    def create_block(self, num_channels, block_num):
        """
        :param num_channels: number of input channels to the block
        :param block_num: the number of the block (among all blocks)
        :return: A sequential container with steps[block_num] MSD layers
        """
        block = nn.Sequential()
        if block_num == 0:
            block.add_module('MSD_first', MSDFirstLayer(self.image_channels,
                num_channels, self.num_scales, self.args))
        current_channels = num_channels
        for _ in range(0, self.steps[block_num]):
            if self.msd_prune == 'max':
                interval = math.ceil(self.num_layers / self.num_scales)
                in_scales = int(self.num_scales - math.floor(max(0, self.
                    cur_layer - 2) / interval))
                out_scales = int(self.num_scales - math.floor((self.
                    cur_layer - 1) / interval))
            else:
                raise NotImplementedError
            self.print_layer(in_scales, out_scales)
            self.cur_layer += 1
            block.add_module('MSD_layer_{}'.format(self.cur_layer - 1),
                MSDLayer(current_channels, self.growth, in_scales,
                out_scales, self.num_scales, self.args))
            current_channels += self.growth
            if (self.msd_prune == 'max' and in_scales > out_scales and self
                .reduction_rate):
                offset = self.num_scales - out_scales
                new_channels = int(math.floor(current_channels * self.
                    reduction_rate))
                block.add_module('Transition', Transition(current_channels,
                    new_channels, out_scales, offset, self.growth_factor,
                    self.args))
                None
                current_channels = new_channels
                self.cur_transition_layer += 1
            elif self.msd_prune != 'max':
                raise NotImplementedError
        return block, current_channels

    def print_layer(self, in_scales, out_scales):
        None

    def forward(self, x, progress=None):
        """
        Propagate Input image in all blocks of MSD layers and classifiers
        and return a list of classifications

        :param x: Input image / batch
        :return: a list of classification outputs
        """
        outputs = [None] * self.num_blocks
        cur_input = x
        for block_num in range(0, self.num_blocks):
            if self.args.debug:
                None
                None
            block = self.subnets[block_num]
            cur_input = block_output = block(cur_input)
            if self.args.debug:
                None
                for s, b in enumerate(block_output):
                    None
            class_output = self.subnets[block_num + self.num_blocks](
                block_output[-1])
            outputs[block_num] = class_output
        return outputs


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()
        bottleneck_channels = out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class ResNet(nn.Module):

    def __init__(self, config):
        super(ResNet, self).__init__()
        input_shape = config['input_shape']
        n_classes = config['n_classes']
        base_channels = config['base_channels']
        block_type = config['block_type']
        depth = config['depth']
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth
        n_channels = [base_channels, base_channels * 2 * block.expansion, 
            base_channels * 4 * block.expansion]
        self.conv = nn.Conv2d(input_shape[1], n_channels[0], kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(base_channels)
        self.stage1 = self._make_stage(n_channels[0], n_channels[0],
            n_blocks_per_stage, block, stride=1)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, block, stride=2)
        self.feature_size = self._forward_conv(torch.zeros(*input_shape)).view(
            -1).shape[0]
        self.fc = nn.Linear(self.feature_size, n_classes)
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality):
        super(BottleneckBlock, self).__init__()
        bottleneck_channels = cardinality * out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class ResNeXt(nn.Module):

    def __init__(self, config):
        super(ResNeXt, self).__init__()
        input_shape = config['input_shape']
        n_classes = config['n_classes']
        base_channels = config['base_channels']
        depth = config['depth']
        self.cardinality = config['cardinality']
        n_blocks_per_stage = (depth - 2) // 9
        assert n_blocks_per_stage * 9 + 2 == depth
        block = BottleneckBlock
        n_channels = [base_channels, base_channels * block.expansion, 
            base_channels * 2 * block.expansion, base_channels * 4 * block.
            expansion]
        self.conv = nn.Conv2d(input_shape[1], n_channels[0], kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.stage1 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(n_channels[2], n_channels[3],
            n_blocks_per_stage, stride=2)
        with torch.no_grad():
            self.feature_size = self._forward_conv(torch.zeros(*input_shape)
                ).view(-1).shape[0]
        self.fc = nn.Linear(self.feature_size, n_classes)
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(block_name, BottleneckBlock(in_channels,
                    out_channels, stride, self.cardinality))
            else:
                stage.add_module(block_name, BottleneckBlock(out_channels,
                    out_channels, 1, self.cardinality))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualPath(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResidualPath, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.bn2(self.conv2(x))
        return x


class DownsamplingShortcut(nn.Module):

    def __init__(self, in_channels):
        super(DownsamplingShortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(in_channels * 2)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        y1 = F.avg_pool2d(x, kernel_size=1, stride=2, padding=0)
        y1 = self.conv1(y1)
        y2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        y2 = F.avg_pool2d(y2, kernel_size=1, stride=2, padding=0)
        y2 = self.conv2(y2)
        z = torch.cat([y1, y2], dim=1)
        z = self.bn(z)
        return z


class ShakeFunction(Function):

    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        ctx.save_for_backward(x1, x2, alpha, beta)
        y = x1 * alpha.data + x2 * (1 - alpha.data)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, alpha, beta = ctx.saved_variables
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None
        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)
        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_function = ShakeFunction.apply


def get_alpha_beta(batch_size, shake_config, is_cuda):
    forward_shake, backward_shake, shake_image = shake_config
    if forward_shake and not shake_image:
        alpha = torch.rand(1)
    elif forward_shake and shake_image:
        alpha = torch.rand(batch_size).view(batch_size, 1, 1, 1)
    else:
        alpha = torch.tensor(0.5)
    if backward_shake and not shake_image:
        beta = torch.rand(1)
    elif backward_shake and shake_image:
        beta = torch.rand(batch_size).view(batch_size, 1, 1, 1)
    else:
        beta = torch.tensor(0.5)
    if is_cuda:
        alpha, beta = alpha.cuda(), beta.cuda()
    return alpha, beta


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shake_config):
        super(BasicBlock, self).__init__()
        self.shake_config = shake_config
        self.residual_path1 = ResidualPath(in_channels, out_channels, stride)
        self.residual_path2 = ResidualPath(in_channels, out_channels, stride)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('downsample', DownsamplingShortcut(
                in_channels))

    def forward(self, x):
        x1 = self.residual_path1(x)
        x2 = self.residual_path2(x)
        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = False, False, False
        alpha, beta = get_alpha_beta(x.size(0), shake_config, x.is_cuda)
        y = shake_function(x1, x2, alpha, beta)
        return self.shortcut(x) + y


class ResNeXt(nn.Module):

    def __init__(self, config):
        super(ResNeXt, self).__init__()
        input_shape = config['input_shape']
        n_classes = config['n_classes']
        base_channels = config['base_channels']
        depth = config['depth']
        self.shake_config = config['shake_forward'], config['shake_backward'
            ], config['shake_image']
        block = BasicBlock
        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth
        n_channels = [base_channels, base_channels * 2, base_channels * 4]
        self.conv = nn.Conv2d(input_shape[1], n_channels[0], kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(base_channels)
        self.stage1 = self._make_stage(n_channels[0], n_channels[0],
            n_blocks_per_stage, block, stride=1)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, block, stride=2)
        self.feature_size = self._forward_conv(torch.zeros(*input_shape)).view(
            -1).shape[0]
        self.fc = nn.Linear(self.feature_size, n_classes)
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride, shake_config=self.
                    shake_config))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1, shake_config=self.shake_config))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


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
        assert (depth - 4) % 6 == 0
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


class FineTuneModel(nn.Module):
    """
    This freezes the weights of all layers except the last one.

    Arguments:
        original_model: Model to finetune
        arch: Name of model architecture
        num_classes: Number of classes to tune for
    """

    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            self.features = original_model.features
            self.fc = nn.Sequential(*list(original_model.classifier.
                children())[:-1])
            self.classifier = nn.Sequential(nn.Linear(4096, num_classes))
        elif arch.startswith('resnet') or arch.startswith('resnext'):
            self.features = nn.Sequential(*list(original_model.children())[:-1]
                )
            if arch == 'resnet18':
                self.classifier = nn.Sequential(nn.Linear(512, num_classes))
            else:
                self.classifier = nn.Sequential(nn.Linear(2048, num_classes))
        else:
            raise 'Finetuning not supported on this architecture yet. Feel free to add'
        self.unfreeze(False)

    def unfreeze(self, unfreeze):
        for p in self.features.parameters():
            p.requires_grad = unfreeze
        if hasattr(self, 'fc'):
            for p in self.fc.parameters():
                p.requires_grad = unfreeze

    def forward(self, x):
        f = self.features(x)
        if hasattr(self, 'fc'):
            f = f.view(f.size(0), -1)
            f = self.fc(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


class FineTuneModel(nn.Module):
    """
    This freezes the weights of all layers except the last one.

    Arguments:
        original_model: Model to finetune
        arch: Name of model architecture
        num_classes: Number of classes to tune for
    """

    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            self.features = original_model.features
            self.fc = nn.Sequential(*list(original_model.classifier.
                children())[:-1])
            self.classifier = nn.Sequential(nn.Linear(4096, num_classes))
        elif arch.startswith('resnet') or arch.startswith('resnext'):
            self.features = nn.Sequential(*list(original_model.children())[:-1]
                )
            if arch == 'resnet18':
                self.classifier = nn.Sequential(nn.Linear(512, num_classes))
            else:
                self.classifier = nn.Sequential(nn.Linear(2048, num_classes))
        else:
            raise 'Finetuning not supported on this architecture yet. Feel free to add'
        self.unfreeze(False)

    def unfreeze(self, unfreeze):
        for p in self.features.parameters():
            p.requires_grad = unfreeze
        if hasattr(self, 'fc'):
            for p in self.fc.parameters():
                p.requires_grad = unfreeze

    def forward(self, x):
        f = self.features(x)
        if hasattr(self, 'fc'):
            f = f.view(f.size(0), -1)
            f = self.fc(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hendrycks_robustness(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'out_planes': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BottleneckBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1, 'cardinality': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(CondenseConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(CondenseNet(*[], **{'args': _mock_config(stages=[4, 4], growth=[4, 4], data=4, group_1x1=4, group_3x3=4, bottleneck=4, num_classes=4)}), [torch.rand([4, 3, 64, 64])], {})

    def test_004(self):
        self._check(Conv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(DownsamplingShortcut(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(LambdaBase(*[], **{'fn': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(LearnedGroupConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(ResidualPath(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(TransitionBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'drop_rate': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(_DenseBlock(*[], **{'num_layers': 1, 'in_channels': 4, 'growth_rate': 4, 'args': _mock_config(group_1x1=4, group_3x3=4, bottleneck=4)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(_DenseLayer(*[], **{'in_channels': 4, 'growth_rate': 4, 'args': _mock_config(group_1x1=4, group_3x3=4, bottleneck=4)}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(_Transition(*[], **{'in_channels': 4, 'args': _mock_config()}), [torch.rand([4, 4, 4, 4])], {})

