import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
resnet = _module
c2_model_loading = _module
resnet = _module
train_val = _module

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


from collections import namedtuple


import math


import torch


import torch.nn.functional as F


from torch import nn


import torch.nn as nn


from collections import OrderedDict


import time


import numpy as np


import torch.backends.cudnn as cudnn


import torch.nn.parallel


class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """

    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups
        super(SpatialCGNL, self).__init__()
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
            groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        if self.use_scale:
            cprint("=> WARN: SpatialCGNL block uses 'SCALE'", 'yellow')
        if self.groups:
            cprint("=> WARN: SpatialCGNL block uses '{}' groups".format(
                self.groups), 'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)
        att = torch.bmm(p, g)
        if self.use_scale:
            att = att.vid((c * h * w) ** 0.5)
        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)
        return x

    def forward(self, x):
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()
        if self.groups and self.groups > 1:
            _c = int(c / self.groups)
            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)
            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h, w)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h, w)
        x = self.z(x)
        x = self.gn(x) + residual
        return x


class SpatialCGNLx(nn.Module):
    """Spatial CGNL block with Gaussian RBF kernel for image classification.
    """

    def __init__(self, inplanes, planes, use_scale=False, groups=None, order=2
        ):
        self.use_scale = use_scale
        self.groups = groups
        self.order = order
        super(SpatialCGNLx, self).__init__()
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
            groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        if self.use_scale:
            cprint("=> WARN: SpatialCGNLx block uses 'SCALE'", 'yellow')
        if self.groups:
            cprint("=> WARN: SpatialCGNLx block uses '{}' groups".format(
                self.groups), 'yellow')
        cprint(
            '=> WARN: The Taylor expansion order in SpatialCGNLx block is {}'
            .format(self.order), 'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The non-linear kernel (Gaussian RBF).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)
        gamma = torch.Tensor(1).fill_(0.0001)
        beta = torch.exp(-2 * gamma)
        t_taylor = []
        p_taylor = []
        for order in range(self.order + 1):
            alpha = torch.mul(torch.div(torch.pow(2 * gamma, order), math.
                factorial(order)), beta)
            alpha = torch.sqrt(alpha)
            _t = t.pow(order).mul(alpha)
            _p = p.pow(order).mul(alpha)
            t_taylor.append(_t)
            p_taylor.append(_p)
        t_taylor = torch.cat(t_taylor, dim=1)
        p_taylor = torch.cat(p_taylor, dim=1)
        att = torch.bmm(p_taylor, g)
        if self.use_scale:
            att = att.div((c * h * w) ** 0.5)
        att = att.view(b, 1, int(self.order + 1))
        x = torch.bmm(att, t_taylor)
        x = x.view(b, c, h, w)
        return x

    def forward(self, x):
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()
        if self.groups and self.groups > 1:
            _c = int(c / self.groups)
            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)
            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h, w)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h, w)
        x = self.z(x)
        x = self.gn(x) + residual
        return x


class SpatialNL(nn.Module):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
    """

    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale
        super(SpatialNL, self).__init__()
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=
            False)
        self.bn = nn.BatchNorm2d(inplanes)
        if self.use_scale:
            cprint("=> WARN: SpatialNL block uses 'SCALE' before softmax",
                'yellow')

    def forward(self, x):
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()
        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)
        att = torch.bmm(t, p)
        if self.use_scale:
            att = att.div(c ** 0.5)
        att = self.softmax(att)
        x = torch.bmm(att, g)
        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, h, w)
        x = self.z(x)
        x = self.bn(x) + residual
        return x


StageSpec = namedtuple('StageSpec', ['index', 'block_count', 'return_features']
    )


ResNet101FPNStagesTo5 = (StageSpec(index=i, block_count=c, return_features=
    r) for i, c, r in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
    )


ResNet50FPNStagesTo5 = (StageSpec(index=i, block_count=c, return_features=r
    ) for i, c, r in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True)))


ResNet50StagesTo4 = (StageSpec(index=i, block_count=c, return_features=r) for
    i, c, r in ((1, 3, False), (2, 4, False), (3, 6, True)))


ResNet50StagesTo5 = (StageSpec(index=i, block_count=c, return_features=r) for
    i, c, r in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True)))


_STAGE_SPECS = {'R-50-C4': ResNet50StagesTo4, 'R-50-C5': ResNet50StagesTo5,
    'R-50-FPN': ResNet50FPNStagesTo5, 'R-101-FPN': ResNet101FPNStagesTo5}


class BottleneckWithFixedBatchNorm(nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels,
        num_groups=1, stride_in_1x1=True, stride=1):
        super(BottleneckWithFixedBatchNorm, self).__init__()
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, bias=False),
                FrozenBatchNorm2d(out_channels))
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1,
            stride=stride_1x1, bias=False)
        self.bn1 = FrozenBatchNorm2d(bottleneck_channels)
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride_3x3, padding=1, bias=False, groups
            =num_groups)
        self.bn2 = FrozenBatchNorm2d(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=
            1, bias=False)
        self.bn3 = FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)
        out0 = self.conv3(out)
        out = self.bn3(out0)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu_(out)
        return out


class StemWithFixedBatchNorm(nn.Module):

    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__()
        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding.
    """
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
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """

    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups
        super(SpatialCGNL, self).__init__()
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
            groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        if self.use_scale:
            cprint("=> WARN: SpatialCGNL block uses 'SCALE'", 'yellow')
        if self.groups:
            cprint("=> WARN: SpatialCGNL block uses '{}' groups".format(
                self.groups), 'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).

        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)
        att = torch.bmm(p, g)
        if self.use_scale:
            att = att.div((c * h * w) ** 0.5)
        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)
        return x

    def forward(self, x):
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()
        if self.groups and self.groups > 1:
            _c = int(c / self.groups)
            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)
            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h, w)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h, w)
        x = self.z(x)
        x = self.gn(x) + residual
        return x


class SpatialCGNLx(nn.Module):
    """Spatial CGNL block with Gaussian RBF kernel for image classification.
    """

    def __init__(self, inplanes, planes, use_scale=False, groups=None, order=2
        ):
        self.use_scale = use_scale
        self.groups = groups
        self.order = order
        super(SpatialCGNLx, self).__init__()
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
            groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        if self.use_scale:
            cprint("=> WARN: SpatialCGNLx block uses 'SCALE'", 'yellow')
        if self.groups:
            cprint("=> WARN: SpatialCGNLx block uses '{}' groups".format(
                self.groups), 'yellow')
        cprint(
            '=> WARN: The Taylor expansion order in SpatialCGNLx block is {}'
            .format(self.order), 'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The non-linear kernel (Gaussian RBF).

        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)
        gamma = torch.Tensor(1).fill_(0.0001)
        beta = torch.exp(-2 * gamma)
        t_taylor = []
        p_taylor = []
        for order in range(self.order + 1):
            alpha = torch.mul(torch.div(torch.pow(2 * gamma, order), math.
                factorial(order)), beta)
            alpha = torch.sqrt(alpha)
            _t = t.pow(order).mul(alpha)
            _p = p.pow(order).mul(alpha)
            t_taylor.append(_t)
            p_taylor.append(_p)
        t_taylor = torch.cat(t_taylor, dim=1)
        p_taylor = torch.cat(p_taylor, dim=1)
        att = torch.bmm(p_taylor, g)
        if self.use_scale:
            att = att.div((c * h * w) ** 0.5)
        att = att.view(b, 1, int(self.order + 1))
        x = torch.bmm(att, t_taylor)
        x = x.view(b, c, h, w)
        return x

    def forward(self, x):
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()
        if self.groups and self.groups > 1:
            _c = int(c / self.groups)
            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)
            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h, w)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h, w)
        x = self.z(x)
        x = self.gn(x) + residual
        return x


class SpatialNL(nn.Module):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
    """

    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale
        super(SpatialNL, self).__init__()
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=
            False)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=
            False)
        self.bn = nn.BatchNorm2d(inplanes)
        if self.use_scale:
            cprint("=> WARN: SpatialNL block uses 'SCALE' before softmax",
                'yellow')

    def forward(self, x):
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()
        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)
        att = torch.bmm(t, p)
        if self.use_scale:
            att = att.div(c ** 0.5)
        att = self.softmax(att)
        x = torch.bmm(att, g)
        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, h, w)
        x = self.z(x)
        x = self.bn(x) + residual
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nl_type=None,
        nl_nums=None, pool_size=7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if not nl_nums:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                nl_type=nl_type, nl_nums=nl_nums)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(pool_size, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if nl_nums == 1:
            for name, m in self._modules['layer3'][-2].named_modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, nl_type=None,
        nl_nums=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if (i == 5 and blocks == 6 or i == 22 and blocks == 23 or i == 
                35 and blocks == 36):
                if nl_type == 'nl':
                    layers.append(SpatialNL(self.inplanes, int(self.
                        inplanes / 2), use_scale=True))
                elif nl_type == 'cgnl':
                    layers.append(SpatialCGNL(self.inplanes, int(self.
                        inplanes / 2), use_scale=False, groups=8))
                elif nl_type == 'cgnlx':
                    layers.append(SpatialCGNLx(self.inplanes, int(self.
                        inplanes / 2), use_scale=False, groups=8, order=3))
                else:
                    pass
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
        x = self.dropout(x)
        x = self.fc(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_KaiyuYue_cgnl_network_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(SpatialNL(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

