import sys
_module = sys.modules[__name__]
del sys
RandAugment = _module
dataset = _module
util = _module
convert_pretrained = _module
base_trainer = _module
contrast_trainer = _module
linear_trainer = _module
main_contrast = _module
main_linear = _module
alias_multinomial = _module
build_memory = _module
mem_bank = _module
mem_moco = _module
build_backbone = _module
build_linear = _module
resnest = _module
resnet = _module
resnet_cmc = _module
util = _module
base_options = _module
test_options = _module
train_options = _module

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


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from collections import OrderedDict


import torch.utils.data.distributed


import torch.multiprocessing as mp


import torch.nn.functional as F


import math


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import Linear


from torch.nn import BatchNorm2d


from torch.nn import ReLU


from torch.nn.modules.utils import _pair


import torch.utils.model_zoo as model_zoo


class BaseMem(nn.Module):
    """Base Memory Class"""

    def __init__(self, K=65536, T=0.07, m=0.5):
        super(BaseMem, self).__init__()
        self.K = K
        self.T = T
        self.m = m

    def _update_memory(self, memory, x, y):
        """
        Args:
          memory: memory buffer
          x: features
          y: index of updating position
        """
        with torch.no_grad():
            x = x.detach()
            w_pos = torch.index_select(memory, 0, y.view(-1))
            w_pos.mul_(self.m)
            w_pos.add_(torch.mul(x, 1 - self.m))
            updated_weight = F.normalize(w_pos)
            memory.index_copy_(0, y, updated_weight)

    def _compute_logit(self, x, w):
        """
        Args:
          x: feat, shape [bsz, n_dim]
          w: softmax weight, shape [bsz, self.K + 1, n_dim]
        """
        x = x.unsqueeze(2)
        out = torch.bmm(w, x)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()
        return out


class BaseMoCo(nn.Module):
    """base class for MoCo-style memory cache"""

    def __init__(self, K=65536, T=0.07):
        super(BaseMoCo, self).__init__()
        self.K = K
        self.T = T
        self.index = 0

    def _update_pointer(self, bsz):
        self.index = (self.index + bsz) % self.K

    def _update_memory(self, k, queue):
        """
        Args:
          k: key feature
          queue: memory buffer
        """
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg)
            out_ids = torch.fmod(out_ids + self.index, self.K).long()
            queue.index_copy_(0, out_ids, k)

    def _compute_logit(self, q, k, queue):
        """
        Args:
          q: query/anchor feature
          k: key feature
          queue: memory buffer
        """
        bsz = q.shape[0]
        pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1))
        pos = pos.view(bsz, 1)
        neg = torch.mm(queue, q.transpose(1, 0))
        neg = neg.transpose(0, 1)
        out = torch.cat((pos, neg), dim=1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()
        return out


model_urls = {'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=
            progress)
        model.load_state_dict(state_dict)
    return model


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
        pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained,
        progress, **kwargs)


def resnext101_32x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-101 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext101_32x4d', Bottleneck, [3, 4, 23, 3],
        pretrained, progress, **kwargs)


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnext152_32x8d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-152 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext152_32x8d', Bottleneck, [3, 8, 36, 3],
        pretrained, progress, **kwargs)


def resnext152_64x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-152 64x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return _resnet('resnext152_64x4d', Bottleneck, [3, 8, 36, 3],
        pretrained, progress, **kwargs)


def resnext152_32x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-152 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext152_32x4d', Bottleneck, [3, 8, 36, 3],
        pretrained, progress, **kwargs)


def resnest101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], radix=2, groups=1,
        bottleneck_width=64, deep_stem=True, stem_width=64, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    if pretrained:
        raise NotImplementedError('pretrained model not available')
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext101_64x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-101 64x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return _resnet('resnext101_64x4d', Bottleneck, [3, 4, 23, 3],
        pretrained, progress, **kwargs)


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnest50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], radix=2, groups=1,
        bottleneck_width=64, deep_stem=True, stem_width=32, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    if pretrained:
        raise NotImplementedError('pretrained model not available')
    return model


model_dict = {'resnet50': resnet50, 'resnet101': resnet101, 'resnet152':
    resnet152, 'resnext50': resnext50_32x4d, 'resnext101v1':
    resnext101_32x4d, 'resnext101v2': resnext101_32x8d, 'resnext101v3':
    resnext101_64x4d, 'resnext152v1': resnext152_32x4d, 'resnext152v2':
    resnext152_32x8d, 'resnext152v3': resnext152_64x4d, 'resnest50':
    resnest50, 'resnest101': resnest101}


class RGBSingleHead(nn.Module):
    """RGB model with a single linear/mlp projection head"""

    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBSingleHead, self).__init__()
        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width
        self.encoder = model_dict[name](width=width)
        if head == 'linear':
            self.head = nn.Sequential(nn.Linear(dim_in, feat_dim), Normalize(2)
                )
        elif head == 'mlp':
            self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(
                inplace=True), nn.Linear(dim_in, feat_dim), Normalize(2))
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 4
        elif name.endswith('x2'):
            return name[:-2], 2
        else:
            return name, 1

    def forward(self, x, mode=0):
        feat = self.encoder(x)
        if mode == 0 or mode == 1:
            feat = self.head(feat)
        return feat


class CMCSingleHead(nn.Module):
    """CMC model with a single linear/mlp projection head"""

    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(CMCSingleHead, self).__init__()
        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width
        self.encoder1 = model_dict[name](width=width, in_channel=1)
        self.encoder2 = model_dict[name](width=width, in_channel=2)
        if head == 'linear':
            self.head1 = nn.Sequential(nn.Linear(dim_in, feat_dim),
                Normalize(2))
            self.head2 = nn.Sequential(nn.Linear(dim_in, feat_dim),
                Normalize(2))
        elif head == 'mlp':
            self.head1 = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(
                inplace=True), nn.Linear(dim_in, feat_dim), Normalize(2))
            self.head2 = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(
                inplace=True), nn.Linear(dim_in, feat_dim), Normalize(2))
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 2
        elif name.endswith('x2'):
            return name[:-2], 1
        else:
            return name, 0.5

    def forward(self, x, mode=0):
        x1, x2 = torch.split(x, [1, 2], dim=1)
        feat1 = self.encoder1(x1)
        feat2 = self.encoder2(x2)
        if mode == 0 or mode == 1:
            feat1 = self.head1(feat1)
            feat2 = self.head2(feat2)
        return torch.cat((feat1, feat2), dim=1)


class DropBlock2D(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.
            size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, groups=2, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, width=1, in_channel=3,
        zero_init_residual=False, groups=1, width_per_group=64,
        replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = max(int(64 * width), 64)
        self.base = int(64 * width)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7,
            stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1],
            stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.base * 4, layers[2],
            stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, self.base * 8, layers[3],
            stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))
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
        x = torch.flatten(x, 1)
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, groups=2,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, groups=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, groups=2,
            bias=False)
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

    def __init__(self, block, layers, width=1):
        super(ResNet, self).__init__()
        self.inplanes = 64 * 2
        self.conv1_v1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.conv1_v2 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.base = int(64 * width)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1],
            stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2],
            stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3],
            stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
                block.expansion, kernel_size=1, stride=stride, groups=2,
                bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1, x2 = torch.split(x, [1, 2], dim=1)
        x1 = self.conv1_v1(x1)
        x2 = self.conv1_v2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat_dim = x.shape[1]
        x1, x2 = torch.split(x, [feat_dim // 2, feat_dim // 2], dim=1)
        return x1, x2


class Normalize(nn.Module):

    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)


class JigsawHead(nn.Module):
    """Jigswa + linear + l2norm"""

    def __init__(self, dim_in, dim_out, k=9, head='linear'):
        super(JigsawHead, self).__init__()
        if head == 'linear':
            self.fc1 = nn.Linear(dim_in, dim_out)
        elif head == 'mlp':
            self.fc1 = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(
                inplace=True), nn.Linear(dim_in, dim_out))
        else:
            raise NotImplementedError('JigSaw head not supported: {}'.
                format(head))
        self.fc2 = nn.Linear(dim_out * k, dim_out)
        self.l2norm = Normalize(2)
        self.k = k

    def forward(self, x):
        bsz = x.shape[0]
        x = self.fc1(x)
        shuffle_ids = self.get_shuffle_ids(bsz)
        x = x[shuffle_ids]
        n_img = int(bsz / self.k)
        x = x.view(n_img, -1)
        x = self.fc2(x)
        x = self.l2norm(x)
        return x

    def get_shuffle_ids(self, bsz):
        n_img = int(bsz / self.k)
        rnd_ids = [torch.randperm(self.k) for i in range(n_img)]
        rnd_ids = torch.cat(rnd_ids, dim=0)
        base_ids = torch.arange(bsz)
        base_ids = torch.div(base_ids, self.k).long()
        base_ids = base_ids * self.k
        shuffle_ids = rnd_ids + base_ids
        return shuffle_ids


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_HobbitLong_PyContrast(_paritybench_base):
    pass

    def test_000(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(Normalize(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
