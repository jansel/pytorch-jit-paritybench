import sys
_module = sys.modules[__name__]
del sys
main_pretrain = _module
main_prune = _module
main_prune_separable = _module
models = _module
presnet = _module
resnet = _module
vgg = _module
pruner = _module
fisher_diag_pruner = _module
kfac_OBD_F2 = _module
kfac_OBS_F2 = _module
kfac_eigen_pruner = _module
kfac_eigen_svd_pruner = _module
kfac_full_pruner = _module
common_utils = _module
compute_flops = _module
compute_wallclock_time = _module
data_utils = _module
kfac_utils = _module
network_utils = _module
prune_utils = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


import torch.optim as optim


import math


import numpy as np


import torch.nn.functional as F


import torch.nn.init as init


from collections import OrderedDict


import time


import logging


import torchvision


from torch.autograd import Variable


import torchvision.transforms as transforms


from torch.nn.modules.utils import _pair


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
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, (selected_index), :, :]
        return output


def try_cuda(x):
    if torch.cuda.is_available():
        x = x
    return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        is_pruned = hasattr(self.conv1, 'in_indices')
        if is_pruned:
            indices = []
            indices.append(self.conv3.out_indices)
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
            if is_pruned:
                indices.append(self.downsample[0].out_indices)
        elif is_pruned:
            indices.append(self.conv1.in_indices)
        if is_pruned:
            n_c = len(set(indices[0] + indices[1]))
            all_indices = list(set(indices[0] + indices[1]))
            r_indices = []
            o_indices = []
            for i in range(n_c):
                idx = all_indices[i]
                if idx in indices[0] and idx in indices[1]:
                    r_indices.append(i)
                    o_indices.append(i)
                elif idx in indices[0]:
                    o_indices.append(i)
                elif idx in indices[1]:
                    r_indices.append(i)
            res = try_cuda(torch.zeros(x.size(0), n_c, residual.size(2), residual.size(3)))
            res[:, (r_indices), :, :] = residual
            res[:, (o_indices), :, :] += out
            out = res
        else:
            out += residual
        return out


class presnet(nn.Module):

    def __init__(self, depth=164, dataset='cifar10', cfg=None):
        super(presnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
        n = (depth - 2) // 9
        block = Bottleneck
        if cfg is None:
            cfg = [[64, 64, 64], [64, 64, 64] * (n - 1), [64, 64, 64], [128, 128, 128] * (n - 1), [128, 128, 128], [256, 256, 256] * (n - 1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, n, cfg=cfg[0:3 * n])
        self.layer2 = self._make_layer(block, 128, n, cfg=cfg[3 * n:6 * n], stride=2)
        self.layer3 = self._make_layer(block, 256, n, cfg=cfg[6 * n:9 * n], stride=2)
        self.bn = nn.BatchNorm2d(256 * block.expansion)
        self.select = channel_selection(256 * block.expansion)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
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
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvLayerRotation(nn.Module):

    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(ConvLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3)
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)
        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1, x.size(2), x.size(3)).fill_(self.bias)], 1)
        return F.conv2d(x, self.rotation_matrix, None, _pair(1), _pair(0), _pair(1), 1)

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return 'in_channels=%s, out_channels=%s, trainable=%s' % (self.rotation_matrix.size(1), self.rotation_matrix.size(0), self.trainable)


class LinearLayerRotation(nn.Module):

    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(LinearLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)
        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1).fill_(self.bias)], 1)
        return x @ self.rotation_matrix

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return 'in_features=%s, out_features=%s, trainable=%s' % (self.rotation_matrix.size(1), self.rotation_matrix.size(0), self.trainable)


def register_bottleneck_layer(m, Q_g, Q_a, W_star, use_patch, trainable=False):
    assert use_patch
    if isinstance(m, nn.Linear):
        scale = nn.Linear(W_star.size(1), W_star.size(0), bias=False)
        scale.weight.data.copy_(W_star)
        bias = 1.0 if m.bias is not None else 0
        return nn.Sequential(LinearLayerRotation(Q_a, bias, trainable), scale, LinearLayerRotation(Q_g.t(), trainable=trainable))
    elif isinstance(m, nn.Conv2d):
        W_star = W_star.view(W_star.size(0), m.kernel_size[0], m.kernel_size[1], -1)
        W_star = W_star.transpose(2, 3).transpose(1, 2).contiguous()
        scale = nn.Conv2d(W_star.size(1), W_star.size(0), m.kernel_size, m.stride, m.padding, m.dilation, m.groups, False)
        scale.weight.data.copy_(W_star)
        patch_size = m.kernel_size[0] * m.kernel_size[1]
        bias = 1.0 / patch_size if m.bias is not None else 0
        return nn.Sequential(ConvLayerRotation(Q_a.t(), bias, trainable), scale, ConvLayerRotation(Q_g, trainable=trainable))
    else:
        raise NotImplementedError


def update_QQ_dict(Q_g, Q_a, m, n):
    if n is not m:
        Q_g[n] = Q_g[m]
        Q_a[n] = Q_a[m]
        Q_a.pop(m)
        Q_g.pop(m)


class BottleneckPResNet(nn.Module):

    def __init__(self, net_prev, fix_rotation=True):
        super(BottleneckPResNet, self).__init__()
        self.conv1 = net_prev.conv1
        self.layer1 = net_prev.layer1
        self.layer2 = net_prev.layer2
        self.layer3 = net_prev.layer3
        self.bn = net_prev.bn
        self.fc = net_prev.fc
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fix_rotation = fix_rotation
        self._is_registered = False

    def _update_bottleneck(self, bneck, modules, Q_g, Q_a, W_star, use_patch, fix_rotation):
        m = bneck.conv1
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv1 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv1[1])
        m = bneck.conv2
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv2 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv2[1])
        m = bneck.conv3
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv3 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv3[1])
        m = bneck.downsample
        if m is not None:
            if len(m) == 1 and m[0] in modules:
                m = m[0]
                bneck.downsample = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, bneck.downsample[1])
            elif len(m) == 3 and m[1] in modules:
                m = m[1]
                bneck.downsample = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, bneck.downsample[1])
            else:
                assert len(m) == 1 or len(m) == 3, 'Upexpected layer %s' % m

    def register(self, modules, Q_g, Q_a, W_star, use_patch, fix_rotation, re_init):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                self._update_bottleneck(m, modules, Q_g, Q_a, W_star, use_patch, fix_rotation)
        m = self.conv1
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.conv1 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.conv1[1])
        m = self.fc
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.fc = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.fc[1])
        self._is_registered = True
        if re_init:
            raise NotImplementedError

    def forward(self, x):
        assert self._is_registered
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


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


_AFFINE = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=_AFFINE)

    def forward(self, x):
        is_pruned = hasattr(self.conv1, 'in_indices')
        if is_pruned:
            indices = []
            indices.append(self.conv2.out_indices)
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.bn3(self.downsample(x))
            if is_pruned:
                indices.append(self.downsample[0].out_indices)
        elif is_pruned:
            indices.append(self.conv1.in_indices)
        if is_pruned:
            n_c = len(set(indices[0] + indices[1]))
            all_indices = list(set(indices[0] + indices[1]))
            res = []
            r_indices = []
            o_indices = []
            for i in range(n_c):
                idx = all_indices[i]
                if idx in indices[0] and idx in indices[1]:
                    r_indices.append(i)
                    o_indices.append(i)
                elif idx in indices[0]:
                    o_indices.append(i)
                elif idx in indices[1]:
                    r_indices.append(i)
            res = try_cuda(torch.zeros(x.size(0), n_c, residual.size(2), residual.size(3)))
            res[:, (r_indices), :, :] = residual
            res[:, (o_indices), :, :] += out
            out = res
        else:
            out += residual
        out = F.relu(out)
        return out


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    elif isinstance(m, LinearLayerRotation):
        if m.trainable:
            None
            init.kaiming_normal(m.rotation_matrix)
    elif isinstance(m, ConvLayerRotation):
        if m.trainable:
            None
            init.kaiming_normal(m.rotation_matrix)


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        _outputs = [64, 128, 256]
        self.in_planes = _outputs[0]
        self.conv1 = nn.Conv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(_outputs[0], affine=_AFFINE)
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(_outputs[2], num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BottleneckResNet(nn.Module):

    def __init__(self, net_prev, fix_rotation=True):
        super(BottleneckResNet, self).__init__()
        self.conv1 = net_prev.conv1
        self.bn = net_prev.bn
        self.layer1 = net_prev.layer1
        self.layer2 = net_prev.layer2
        self.layer3 = net_prev.layer3
        self.linear = net_prev.linear
        self.fix_rotation = fix_rotation
        self._is_registered = False

    def _update_bottleneck(self, bneck, modules, Q_g, Q_a, W_star, use_patch, fix_rotation):
        m = bneck.conv1
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv1 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv1[1])
        m = bneck.conv2
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv2 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv2[1])
        m = bneck.downsample
        if m is not None:
            if len(m) == 1 and m[0] in modules:
                m = m[0]
                bneck.downsample = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, bneck.downsample[1])
            elif len(m) == 3 and m[1] in modules:
                m = m[1]
                bneck.downsample = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, bneck.downsample[1])
            else:
                assert len(m) == 1 or len(m) == 3, 'Upexpected layer %s' % m

    def register(self, modules, Q_g, Q_a, W_star, use_patch, fix_rotation, re_init):
        for m in self.modules():
            if isinstance(m, BasicBlock):
                self._update_bottleneck(m, modules, Q_g, Q_a, W_star, use_patch, fix_rotation)
        m = self.conv1
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.conv1 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.conv1[1])
        m = self.linear
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.linear = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.linear[1])
        self._is_registered = True
        if re_init:
            self.apply(_weights_init)

    def forward(self, x):
        assert self._is_registered
        out = F.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


defaultcfg = {(11): [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512], (13): [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512], (16): [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512], (19): [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]}


class VGG(nn.Module):

    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.feature = self.make_layers(cfg, True)
        self.dataset = dataset
        if dataset == 'cifar10' or dataset == 'cinic-10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self.apply(_weights_init)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=_AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class BottleneckVGG(nn.Module):

    def __init__(self, vgg_prev, fix_rotation=True):
        super(BottleneckVGG, self).__init__()
        self.dataset = vgg_prev.dataset
        self.feature = vgg_prev.feature
        self.classifier = vgg_prev.classifier
        self.fix_rotation = fix_rotation
        self._is_registered = False

    def register(self, modules, Q_g, Q_a, W_star, use_patch, fix_rotation, re_init):
        n_seqs = len(self.feature)
        for idx in range(n_seqs):
            m = self.feature[idx]
            if isinstance(m, nn.Sequential):
                m = m[1]
            if m in modules:
                self.feature[idx] = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, self.feature[idx][1])
        m = self.classifier
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.classifier = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.classifier)
        self._is_registered = True
        if re_init:
            self.apply(_weights_init)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, LinearLayerRotation):
                if m.trainable:
                    None
                    m.rotation_matrix.data.normal_(0, 0.01)
            elif isinstance(m, ConvLayerRotation):
                if m.trainable:
                    None
                    n = 1 * m.rotation_matrix.size(1)
                    m.rotation_matrix.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, x):
        assert self._is_registered
        nseq = len(self.feature)
        for idx in range(nseq):
            x = self.feature[idx](x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LambdaLayer,
     lambda: ([], {'lambd': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGG,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     False),
    (channel_selection,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (presnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     False),
]

class Test_alecwangcq_EigenDamage_Pytorch(_paritybench_base):
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

