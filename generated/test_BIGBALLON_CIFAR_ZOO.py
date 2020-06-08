import sys
_module = sys.modules[__name__]
del sys
models = _module
alexnet = _module
cbam_resnext = _module
densenet = _module
genet = _module
lenet = _module
preresnet = _module
resnet = _module
resnext = _module
senet = _module
shake_shake = _module
sknet = _module
vgg = _module
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


import torch.nn as nn


import torch


import torch.nn.functional as F


import math


from torch.autograd import Variable


import logging


import torch.backends.cudnn as cudnn


from torch.utils.tensorboard import SummaryWriter


class AlexNet(nn.Module):

    def __init__(self, num_classes):
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
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01,
            affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg',
        'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, 
            gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(
            gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(
                    x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(
                    x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=
                    (x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3
            ).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1)
            .unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg',
        'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio,
            pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, cardinality,
        base_width, expansion):
        super(Bottleneck, self).__init__()
        width_ratio = out_channels / (expansion * 64.0)
        D = cardinality * int(base_width * width_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.cbam_module = CBAM(out_channels)
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
        out = self.conv_reduce.forward(x)
        out = self.relu(self.bn_reduce.forward(out))
        out = self.conv_conv.forward(out)
        out = self.relu(self.bn.forward(out))
        out = self.conv_expand.forward(out)
        out = self.bn_expand.forward(out)
        residual = self.shortcut.forward(x)
        out = self.cbam_module(out) + residual
        out = self.relu(out)
        return out


class SeResNeXt(nn.Module):

    def __init__(self, cardinality, depth, num_classes, base_width, expansion=4
        ):
        super(SeResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.expansion, 128 * self.expansion, 256 *
            self.expansion]
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.fc = nn.Linear(self.stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, name, in_channels, out_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, Bottleneck(in_channels,
                    out_channels, pool_stride, self.cardinality, self.
                    base_width, self.expansion))
            else:
                block.add_module(name_, Bottleneck(out_channels,
                    out_channels, 1, self.cardinality, self.base_width,
                    self.expansion))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.fc(x)


class Bottleneck(nn.Module):

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn_1 = nn.BatchNorm2d(in_planes)
        self.conv_1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1,
            bias=False)
        self.bn_2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv_2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv_1(F.relu(self.bn_1(x)))
        out = self.conv_2(F.relu(self.bn_2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, block, depth, growth_rate=12, reduction=0.5,
        num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        nblocks = (depth - 4) // 6
        num_planes = 2 * growth_rate
        self.conv_1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1,
            bias=False)
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans_1 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans_2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense_3 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate
        self.bn = nn.BatchNorm2d(num_planes)
        self.fc = nn.Linear(num_planes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.trans_1(self.dense1(out))
        out = self.trans_2(self.dense2(out))
        out = self.dense_3(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Downblock(nn.Module):

    def __init__(self, channels, kernel_size):
        super(Downblock, self).__init__()
        self.dwconv = nn.Conv2d(channels, channels, groups=channels, stride
            =1, kernel_size=kernel_size, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        return x


class GEModule(nn.Module):

    def __init__(self, in_planes, out_planes, spatial):
        super(GEModule, self).__init__()
        self.downop = Downblock(out_planes, kernel_size=spatial)
        self.mlp = nn.Sequential(nn.Conv2d(out_planes, out_planes // 16,
            kernel_size=1, padding=0, bias=False), nn.ReLU(), nn.Conv2d(
            out_planes // 16, out_planes, kernel_size=1, padding=0, bias=False)
            )

    def forward(self, x):
        out = self.downop(x)
        out = self.mlp(out)
        shape_in = out.shape[-1]
        out = F.interpolate(out, shape_in)
        out = torch.sigmoid(out)
        out = x * out
        return out


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, spatial,
        cardinality, base_width, expansion):
        super(Bottleneck, self).__init__()
        width_ratio = out_channels / (expansion * 64.0)
        D = cardinality * int(base_width * width_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.ge_module = GEModule(in_channels, out_channels, spatial)
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
        out = self.conv_reduce.forward(x)
        out = self.relu(self.bn_reduce.forward(out))
        out = self.conv_conv.forward(out)
        out = self.relu(self.bn.forward(out))
        out = self.conv_expand.forward(out)
        out = self.bn_expand.forward(out)
        residual = self.shortcut.forward(x)
        out = self.ge_module(out) + residual
        out = self.relu(out)
        return out


class GeResNeXt(nn.Module):

    def __init__(self, cardinality, depth, num_classes, base_width, expansion=4
        ):
        super(GeResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.expansion, 128 * self.expansion, 256 *
            self.expansion]
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1],
            32, 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2],
            16, 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3],
            8, 2)
        self.fc = nn.Linear(self.stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, name, in_channels, out_channels, spatial, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, Bottleneck(in_channels,
                    out_channels, pool_stride, spatial, self.cardinality,
                    self.base_width, self.expansion))
            else:
                block.add_module(name_, Bottleneck(out_channels,
                    out_channels, 1, spatial, self.cardinality, self.
                    base_width, self.expansion))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.fc(x)


class LeNet(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn_1(x)
        out = self.relu(out)
        out = self.conv_1(out)
        out = self.bn_2(out)
        out = self.relu(out)
        out = self.conv_2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(planes)
        self.conv_3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn_1(x)
        out = self.relu(out)
        out = self.conv_1(out)
        out = self.bn_2(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_3(out)
        out = self.relu(out)
        out = self.conv_3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class PreResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(PreResNet, self).__init__()
        if block_name.lower() == 'basicblock':
            assert (depth - 2
                ) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2
                ) % 9 == 0, 'When use bottleneck, depth should be 9n+2 e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')
        self.inplanes = 16
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
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
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
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
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(planes, planes)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.conv_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)
        out = self.conv_3(out)
        out = self.bn_3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        if block_name == 'BasicBlock':
            assert (depth - 2
                ) % 6 == 0, 'depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name == 'Bottleneck':
            assert (depth - 2
                ) % 9 == 0, 'depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')
        self.inplanes = 16
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.stage_1 = self._make_layer(block, 16, n)
        self.stage_2 = self._make_layer(block, 32, n, stride=2)
        self.stage_3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
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
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, cardinality,
        base_width, expansion):
        super(Bottleneck, self).__init__()
        width_ratio = out_channels / (expansion * 64.0)
        D = cardinality * int(base_width * width_ratio)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.conv_reduce.forward(x)
        out = self.relu(self.bn_reduce.forward(out))
        out = self.conv_conv.forward(out)
        out = self.relu(self.bn.forward(out))
        out = self.conv_expand.forward(out)
        out = self.bn_expand.forward(out)
        residual = self.shortcut.forward(x)
        return self.relu(residual + out)


class ResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, num_classes, base_width, expansion=4
        ):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            num_classes: number of classes
            base_width: base number of channels in each group.
            expansion: factor to adjust the channel dimensionality
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.expansion, 128 * self.expansion, 256 *
            self.expansion]
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.fc = nn.Linear(self.stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, name, in_channels, out_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, Bottleneck(in_channels,
                    out_channels, pool_stride, self.cardinality, self.
                    base_width, self.expansion))
            else:
                block.add_module(name_, Bottleneck(out_channels,
                    out_channels, 1, self.cardinality, self.base_width,
                    self.expansion))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.fc(x)


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Conv2d(channels, channels // reduction, kernel_size=
            1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Conv2d(channels // reduction, channels, kernel_size=
            1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original = x
        x = self.avg_pool(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return original * x


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, cardinality,
        base_width, expansion):
        super(Bottleneck, self).__init__()
        width_ratio = out_channels / (expansion * 64.0)
        D = cardinality * int(base_width * width_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(out_channels)
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
        out = self.conv_reduce.forward(x)
        out = self.relu(self.bn_reduce.forward(out))
        out = self.conv_conv.forward(out)
        out = self.relu(self.bn.forward(out))
        out = self.conv_expand.forward(out)
        out = self.bn_expand.forward(out)
        residual = self.shortcut.forward(x)
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SeResNeXt(nn.Module):

    def __init__(self, cardinality, depth, num_classes, base_width, expansion=4
        ):
        super(SeResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.expansion, 128 * self.expansion, 256 *
            self.expansion]
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.fc = nn.Linear(self.stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, name, in_channels, out_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, Bottleneck(in_channels,
                    out_channels, pool_stride, self.cardinality, self.
                    base_width, self.expansion))
            else:
                block.add_module(name_, Bottleneck(out_channels,
                    out_channels, 1, self.cardinality, self.base_width,
                    self.expansion))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, x.size(3), 1)
        x = x.view(-1, self.stages[3])
        return self.fc(x)


class Shortcut(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0,
            bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = F.relu(x)
        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)
        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, self.stride)
        h2 = self.conv2(h2)
        h = torch.cat((h1, h2), 1)
        return self.bn(h)


class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.cuda.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        beta = Variable(beta)
        return beta * grad_output, (1 - beta) * grad_output, None


class ShakeBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super(ShakeBlock, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(in_ch, out_ch,
            stride=stride)
        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(in_ch,
            out_ch, 3, padding=1, stride=stride, bias=False), nn.
            BatchNorm2d(out_ch), nn.ReLU(inplace=False), nn.Conv2d(out_ch,
            out_ch, 3, padding=1, stride=1, bias=False), nn.BatchNorm2d(out_ch)
            )


class ShakeResNet(nn.Module):

    def __init__(self, depth, base_width, num_classes):
        super(ShakeResNet, self).__init__()
        n_units = (depth - 2) / 6
        in_chs = [16, base_width, base_width * 2, base_width * 4]
        self.in_chs = in_chs
        self.conv_1 = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(in_chs[0])
        self.stage_1 = self._make_layer(n_units, in_chs[0], in_chs[1])
        self.stage_2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2)
        self.stage_3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2)
        self.fc_out = nn.Linear(in_chs[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.stage_1(out)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_chs[3])
        out = self.fc_out(out)
        return out

    def _make_layer(self, n_units, in_ch, out_ch, stride=1):
        layers = []
        for _ in range(int(n_units)):
            layers.append(ShakeBlock(in_ch, out_ch, stride=stride))
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)


class SKConv(nn.Module):

    def __init__(self, features, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(nn.Conv2d(features, features,
                kernel_size=1 + i * 2, stride=stride, padding=i, groups=G),
                nn.BatchNorm2d(features), nn.ReLU(inplace=False)))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze_()
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                    dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, cardinality,
        base_width, expansion, M, r, L):
        super(Bottleneck, self).__init__()
        width_ratio = out_channels / (expansion * 64.0)
        D = cardinality * int(base_width * width_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=
            1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_sk = SKConv(D, M, cardinality, r, stride=stride, L=L)
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
        out = self.conv_reduce.forward(x)
        out = self.relu(self.bn_reduce.forward(out))
        out = self.conv_sk.forward(out)
        out = self.relu(self.bn.forward(out))
        out = self.conv_expand.forward(out)
        out = self.bn_expand.forward(out)
        residual = self.shortcut.forward(x)
        return self.relu(residual + out)


class SkResNeXt(nn.Module):

    def __init__(self, cardinality, depth, num_classes, base_width,
        expansion=4, M=2, r=32, L=32):
        super(SkResNeXt, self).__init__()
        self.M = M
        self.r = r
        self.L = L
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.expansion, 128 * self.expansion, 256 *
            self.expansion]
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.fc = nn.Linear(self.stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, name, in_channels, out_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, Bottleneck(in_channels,
                    out_channels, pool_stride, self.cardinality, self.
                    base_width, self.expansion, self.M, self.r, self.L))
            else:
                block.add_module(name_, Bottleneck(out_channels,
                    out_channels, 1, self.cardinality, self.base_width,
                    self.expansion, self.M, self.r, self.L))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.fc(x)


class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_BIGBALLON_CIFAR_ZOO(_paritybench_base):
    pass

    def test_000(self):
        self._check(BasicConv(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ChannelPool(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(SpatialGate(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_004(self):
        self._check(Bottleneck(*[], **{'in_channels': 64, 'out_channels': 64, 'stride': 64, 'cardinality': 4, 'base_width': 4, 'expansion': 4, 'M': 4, 'r': 4, 'L': 4}), [torch.rand([4, 64, 64, 64])], {})

    def test_005(self):
        self._check(Transition(*[], **{'in_planes': 4, 'out_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Downblock(*[], **{'channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(SEModule(*[], **{'channels': 64}), [torch.rand([4, 64, 4, 4])], {})

    def test_009(self):
        self._check(Shortcut(*[], **{'in_ch': 4, 'out_ch': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_010(self):
        self._check(ShakeBlock(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(ShakeResNet(*[], **{'depth': 1, 'base_width': 4, 'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})
    @_fails_compile()

    def test_012(self):
        self._check(SKConv(*[], **{'features': 4, 'M': 4, 'G': 4, 'r': 4}), [torch.rand([4, 4, 4, 4])], {})
