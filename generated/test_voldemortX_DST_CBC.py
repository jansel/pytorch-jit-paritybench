import sys
_module = sys.modules[__name__]
del sys
download_cifar = _module
generate_splits = _module
main_dmt = _module
main_fs = _module
wideresnet = _module
autoaugment = _module
common = _module
cutout = _module
datasets = _module
losses = _module
mixup = _module
randomrandaugment = _module
cityscapes_data_list = _module
convert_coco_resnet101 = _module
dms_sample = _module
main = _module
main_flip = _module
main_naive = _module
main_online = _module
models = _module
_utils = _module
resnet = _module
segmentation = _module
_utils = _module
deeplab = _module
fcn = _module
utils = _module
pascal_sbd_split = _module
common = _module
datasets = _module
functional = _module
losses = _module
transforms = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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
xrange = range
wraps = functools.wraps


import time


import torch


import random


import numpy as np


from torch.utils.tensorboard import SummaryWriter


from torchvision.transforms import Compose


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import Normalize


from torchvision.transforms import ToTensor


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


import torch.nn as nn


import torch.nn.functional as F


import matplotlib.pyplot as plt


import collections


import math


import copy


from collections import OrderedDict


from torch import nn


from torch.nn import functional as F


import torchvision


import numbers


import warnings


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
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


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, num_classes, depth, widen_factor, drop_rate):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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


class _Loss(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, *input):
        raise NotImplementedError


class _WeightedLoss(_Loss):

    def __init__(self, weight=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(reduction)
        self.register_buffer('weight', weight)

    def forward(self, *input):
        raise NotImplementedError


class DynamicMutualLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma1=0, gamma2=0, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, split_index=None):
        if targets.dim() == 4:
            real_targets = targets[:, 0, :, :].long()
        else:
            real_targets = targets
        total_loss = self.criterion_ce(inputs, real_targets)
        stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, 'loss': total_loss.sum().item() / (real_targets != self.ignore_index).sum().item()}
        if split_index is None or self.gamma1 == 0 and self.gamma2 == 0:
            total_loss = total_loss.sum() / (real_targets != self.ignore_index).sum()
        else:
            outputs = inputs.softmax(dim=1).clone().detach()
            decision_current = outputs.argmax(1)
            decision_pseudo = real_targets.clone().detach()
            confidence_current = outputs.max(1).values
            temp = real_targets.unsqueeze(1).clone().detach()
            temp[temp == self.ignore_index] = 0
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)
            confidence_pseudo = targets[:, 1, :, :].clone().detach()
            dynamic_weights = torch.ones_like(decision_current).float()
            disagreement = decision_current != decision_pseudo
            current_win = confidence_current > confidence_pseudo
            stats['disagree'] = (disagreement * (real_targets != self.ignore_index))[:split_index].sum().int().item()
            stats['current_win'] = (disagreement * current_win * (real_targets != self.ignore_index))[:split_index].sum().int().item()
            indices = ~disagreement
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma1
            indices = disagreement * current_win
            dynamic_weights[indices] = 0
            indices = disagreement * ~current_win
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma2
            dynamic_weights[split_index:] = 1
            stats['avg_weights'] = dynamic_weights[real_targets != self.ignore_index].mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()
        return total_loss, stats


class MixupDynamicLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, y_a, y_b, lam, w_a=None, w_b=None, dynamic_weights=None):
        if dynamic_weights is None:
            loss_a = self.criterion_ce(pred, y_a)
            loss_b = self.criterion_ce(pred, y_b)
            true_loss = (lam * loss_a + (1 - lam) * loss_b).mean().item()
            total_loss = lam * w_a * loss_a + (1 - lam) * w_b * loss_b
            total_loss = total_loss.sum() / ((y_a != self.ignore_index) * (y_b != self.ignore_index)).sum()
        else:
            total_loss = lam * self.criterion_ce(pred, y_a) + (1 - lam) * self.criterion_ce(pred, y_b)
            true_loss = total_loss.mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / ((y_a != self.ignore_index) * (y_b != self.ignore_index)).sum()
        return total_loss, true_loss

    @staticmethod
    def dynamic_weights_calc(net, inputs, targets, split_index, gamma=0, labeled_weight=1):
        with torch.no_grad():
            outputs = net(inputs).softmax(dim=1).clone().detach()
            indices = targets.unsqueeze(1).clone().detach()
            probabilities = outputs.gather(dim=1, index=indices).squeeze(1)
            probabilities[split_index:] = labeled_weight
            probabilities[:split_index] = probabilities[:split_index] ** gamma
        return probabilities


class SigmoidAscendingMixupDMTLoss(MixupDynamicLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma1, gamma2, T_max, weight=None, ignore_index=-100, reduction='mean'):
        super(MixupDynamicLoss, self).__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.gamma1_ori = gamma1
        self.gamma2_ori = gamma2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.T_max = T_max
        self.last_iter = 1

    def step(self):
        self.last_iter += 1
        ratio = math.e ** (-5 * (1 - self.last_iter / self.T_max) ** 2)
        self.gamma1 = self.gamma1_ori * ratio
        self.gamma2 = self.gamma2_ori * ratio

    def dynamic_weights_calc(self, net, inputs, targets, split_index, labeled_weight=1, margin=0):
        stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, 'gamma1': self.gamma1, 'gamma2': self.gamma2}
        with torch.no_grad():
            outputs = net(inputs).softmax(dim=1).clone().detach()
            decision_current = outputs.argmax(1)
            decision_pseudo = targets.argmax(1).clone().detach()
            confidence_current = outputs.max(1).values
            confidence_pseudo = targets.max(1).values.clone().detach()
            temp = decision_pseudo.unsqueeze(1).clone().detach()
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)
            dynamic_weights = torch.ones_like(decision_current).float()
            disagreement = decision_current != decision_pseudo
            current_win = confidence_current > confidence_pseudo
            stats['disagree'] = disagreement[:split_index].sum().int().item()
            stats['current_win'] = (disagreement * current_win)[:split_index].sum().int().item()
            if self.gamma1 == self.gamma2 == 0:
                return torch.ones(inputs.shape[0]), stats
            indices = ~disagreement
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma1
            indices = disagreement * current_win
            dynamic_weights[indices] = 0
            indices = disagreement * ~current_win
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma2
            dynamic_weights[split_index:] = labeled_weight
            stats['avg_weights'] = dynamic_weights.mean().item()
        return dynamic_weights, stats


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
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

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
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
        x = self.fc(x)
        return x


class _SimpleSegmentationModel(nn.Module):

    def __init__(self, backbone, classifier, aux_classifier=None, recon_head=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.recon_head = recon_head

    def forward(self, x):
        features = self.backbone(x)
        result = OrderedDict()
        x = features['out']
        x = self.classifier(x)
        result['out'] = x
        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            result['aux'] = x
        if self.recon_head is not None:
            x = features['recon']
            x = self.recon_head(x)
            result['recon'] = x
        return result


class DeepLab(_SimpleSegmentationModel):
    pass


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Head(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(DeepLabV3Head, self).__init__(ASPP(in_channels, [12, 24, 36]), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, num_classes, 1))


class ASPP_V2(nn.Module):

    def __init__(self, in_channels, num_classes, atrous_rates):
        super(ASPP_V2, self).__init__()
        self.convs = nn.ModuleList()
        for rates in atrous_rates:
            self.convs.append(nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=rates, dilation=rates, bias=True))

    def forward(self, x):
        res = self.convs[0](x)
        for i in range(len(self.convs) - 1):
            res += self.convs[i + 1](x)
            return res


class DeepLabV2Head(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(DeepLabV2Head, self).__init__(ASPP_V2(in_channels, num_classes, [6, 12, 18, 24]))


class ReconHead(nn.Sequential):

    def __init__(self, in_channels):
        super(ReconHead, self).__init__(nn.Conv2d(in_channels, int(in_channels / 2), 1, bias=False), nn.BatchNorm2d(int(in_channels / 2)), nn.ReLU(), nn.Conv2d(int(in_channels / 2), int(in_channels / 2), 1, bias=False), nn.BatchNorm2d(int(in_channels / 2)), nn.ReLU(), nn.ConvTranspose2d(int(in_channels / 2), int(in_channels / 2), 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(int(in_channels / 2)), nn.ReLU(), nn.Conv2d(int(in_channels / 2), int(int(in_channels / 4)), 1, bias=False), nn.BatchNorm2d(int(in_channels / 4)), nn.ReLU(), nn.Conv2d(int(in_channels / 4), int(in_channels / 4), 1, bias=False), nn.BatchNorm2d(int(in_channels / 4)), nn.ReLU(), nn.ConvTranspose2d(int(in_channels / 4), int(in_channels / 4), 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(int(in_channels / 4)), nn.ReLU(), nn.Conv2d(int(in_channels / 4), int(int(in_channels / 8)), 1, bias=False), nn.BatchNorm2d(int(in_channels / 8)), nn.ReLU(), nn.Conv2d(int(in_channels / 8), int(in_channels / 8), 1, bias=False), nn.BatchNorm2d(int(in_channels / 8)), nn.ReLU(), nn.ConvTranspose2d(int(in_channels / 8), int(in_channels / 8), 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(int(in_channels / 8)), nn.ReLU(), nn.Conv2d(int(in_channels / 8), 3, 1, bias=False))


class FCN(_SimpleSegmentationModel):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class FCNHead(nn.Sequential):

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(inter_channels, channels, 1)]
        super(FCNHead, self).__init__(*layers)


class DynamicLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma=0, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, split_index=None):
        statistics = {}
        total_loss = self.criterion_ce(inputs, targets)
        if split_index is None or self.gamma == 0:
            total_loss = total_loss.sum() / (targets != self.ignore_index).sum()
        else:
            probabilities = inputs.softmax(dim=1).clone().detach()
            indices = targets.unsqueeze(1).clone().detach()
            indices[indices == self.ignore_index] = 0
            probabilities = probabilities.gather(dim=1, index=indices).squeeze(1)
            probabilities[split_index:] = 1
            probabilities[:split_index] = probabilities[:split_index] ** self.gamma
            total_loss = (total_loss * probabilities).sum() / (targets != self.ignore_index).sum()
        statistics['dl'] = total_loss.item()
        return total_loss, statistics


class DynamicNaiveLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, split_index=None):
        if targets.dim() == 4:
            real_targets = targets[:, 0, :, :].long()
        else:
            real_targets = targets
        total_loss = self.criterion_ce(inputs, real_targets)
        stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, 'loss': total_loss.sum().item() / (real_targets != self.ignore_index).sum().item()}
        if split_index is None:
            total_loss = total_loss.sum() / (real_targets != self.ignore_index).sum()
        else:
            outputs = inputs.softmax(dim=1).clone().detach()
            temp = real_targets.unsqueeze(1).clone().detach()
            temp[temp == self.ignore_index] = 0
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)
            dynamic_weights = probabilities_current ** 5
            dynamic_weights[split_index:] = 1
            stats['avg_weights'] = dynamic_weights[real_targets != self.ignore_index].mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()
        return total_loss, stats


class FlipLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma1=0, gamma2=0, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, split_index=None):
        if targets.dim() == 4:
            real_targets = targets[:, 0, :, :].long()
        else:
            real_targets = targets
        if split_index is None or self.gamma1 == 0 and self.gamma2 == 0:
            total_loss = self.criterion_ce(inputs, real_targets)
            stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, 'loss': total_loss.sum().item() / (real_targets != self.ignore_index).sum().item()}
            total_loss = total_loss.sum() / (real_targets != self.ignore_index).sum()
        else:
            stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, 'loss': 0 / (real_targets != self.ignore_index).sum().item()}
            outputs = inputs.softmax(dim=1).clone().detach()
            decision_current = outputs.argmax(1)
            decision_pseudo = real_targets.clone().detach()
            confidence_current = outputs.max(1).values
            temp = real_targets.unsqueeze(1).clone().detach()
            temp[temp == self.ignore_index] = 0
            probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)
            confidence_pseudo = targets[:, 1, :, :].clone().detach()
            dynamic_weights = torch.ones_like(decision_current).float()
            disagreement = decision_current != decision_pseudo
            current_win = confidence_current > confidence_pseudo
            stats['disagree'] = (disagreement * (real_targets != self.ignore_index))[:split_index].sum().int().item()
            stats['current_win'] = (disagreement * current_win * (real_targets != self.ignore_index))[:split_index].sum().int().item()
            indices = ~disagreement
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma1
            indices = disagreement * current_win * (real_targets != self.ignore_index)
            dynamic_weights[indices] = (1 - confidence_pseudo[indices]) ** self.gamma2
            real_targets[:split_index][indices[:split_index]] = decision_current[:split_index][indices[:split_index]]
            indices = disagreement * ~current_win
            dynamic_weights[indices] = probabilities_current[indices] ** self.gamma2
            total_loss = self.criterion_ce(inputs, real_targets)
            stats['loss'] = total_loss.sum().item() / (real_targets != self.ignore_index).sum().item()
            dynamic_weights[split_index:] = 1
            stats['avg_weights'] = dynamic_weights[real_targets != self.ignore_index].mean().item()
            total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()
        return total_loss, stats


class OnlineLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, inputs, targets, split_index=None):
        outputs = inputs.softmax(dim=1).clone().detach()
        pseudo_targets = outputs.argmax(1)
        confidences = outputs.max(1).values
        pseudo_targets[confidences <= 0.9] = 255
        targets[:split_index] = pseudo_targets[:split_index]
        total_loss = self.criterion_ce(inputs, targets)
        stats = {'loss': total_loss.item()}
        return total_loss, stats


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'in_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ASPPConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ASPP_V2,
     lambda: ([], {'in_channels': 4, 'num_classes': 4, 'atrous_rates': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeepLabV2Head,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeepLabV3Head,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DynamicLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DynamicMutualLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DynamicNaiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FCNHead,
     lambda: ([], {'in_channels': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FlipLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_planes': 4, 'out_planes': 4, 'block': _mock_layer, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OnlineLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_voldemortX_DST_CBC(_paritybench_base):
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

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

