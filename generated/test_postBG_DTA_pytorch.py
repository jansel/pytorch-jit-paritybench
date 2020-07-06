import sys
_module = sys.modules[__name__]
del sys
augmentations = _module
misc = _module
consistency_losses = _module
datasets = _module
base = _module
visda = _module
dropout = _module
loggers = _module
losses = _module
main = _module
metrics = _module
misc = _module
models = _module
base = _module
resnet = _module
visda_architectures = _module
options = _module
ramps = _module
trainers = _module
dta_trainer = _module
utils = _module
vat = _module

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


from torch import nn as nn


from torch.nn import functional as F


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import Subset


import abc


import random


import torch.utils.data as data


from torchvision import transforms as transforms


import torch


from copy import deepcopy


from abc import ABC


from torchvision.transforms import ToTensor


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torch.backends.cudnn as cudnn


from collections import OrderedDict


import math


import torchvision.models.resnet as resnet


class AbstractConsistencyLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits1, logits2):
        raise NotImplementedError


class LossWithLogits(AbstractConsistencyLoss):

    def __init__(self, reduction='mean', loss_cls=nn.L1Loss):
        super().__init__(reduction)
        self.loss_with_softmax = loss_cls(reduction=reduction)

    def forward(self, logits1, logits2):
        loss = self.loss_with_softmax(F.softmax(logits1, dim=1), F.softmax(logits2, dim=1))
        return loss


class DiscrepancyLossWithLogits(AbstractConsistencyLoss):

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = LossWithLogits(reduction=reduction, loss_cls=nn.L1Loss)

    def forward(self, logits1, logits2):
        return self.loss(logits1, logits2)


class KLDivLossWithLogits(AbstractConsistencyLoss):

    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.kl_div_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits1, logits2):
        return self.kl_div_loss(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1))


class EntropyLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits):
        p = F.softmax(logits, dim=1)
        elementwise_entropy = -p * F.log_softmax(logits, dim=1)
        if self.reduction == 'none':
            return elementwise_entropy
        sum_entropy = torch.sum(elementwise_entropy, dim=1)
        if self.reduction == 'sum':
            return sum_entropy
        return torch.mean(sum_entropy)


class ClassBalanceLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits):
        p = F.softmax(logits, dim=1)
        cls_balance = -torch.mean(torch.log(torch.mean(p, 0) + 1e-06))
        return cls_balance


def l2_normalize(d):
    d_reshaped = d.view(d.size(0), -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-08
    return d


class VirtualAdversarialPerturbationGenerator(nn.Module):

    def __init__(self, feature_extractor, classifier, xi=1e-06, eps=3.5, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.kl_div = KLDivLossWithLogits()

    def forward(self, inputs):
        with disable_tracking_bn_stats(self.feature_extractor):
            with disable_tracking_bn_stats(self.classifier):
                features, _ = self.feature_extractor(inputs)
                logits = self.classifier(features).detach()
                d = l2_normalize(torch.randn_like(inputs))
                for _ in range(self.ip):
                    x_hat = inputs + self.xi * d
                    x_hat.requires_grad = True
                    features_hat, _ = self.feature_extractor(x_hat)
                    logits_hat = self.classifier(features_hat)
                    adv_distance = self.kl_div(logits_hat, logits)
                    adv_distance.backward()
                    d = l2_normalize(x_hat.grad)
                    self.feature_extractor.zero_grad()
                r_adv = d * self.eps
                return r_adv.detach(), logits


class LDSLoss(nn.Module):

    def __init__(self, model, xi=1e-06, eps=2.0, ip=1):
        super().__init__()
        self.model = model
        self.vap_generator = VirtualAdversarialPerturbationGenerator(model, xi=xi, eps=eps, ip=ip)
        self.kl_div = KLDivLossWithLogits()

    def forward(self, inputs):
        r_adv, logits = self.vap_generator(inputs)
        adv_inputs = inputs + r_adv
        adv_logits = self.model(adv_inputs)
        lds_loss = self.kl_div(adv_logits, logits)
        return lds_loss


class AbstractModel(nn.Module, ABC):

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        super().load_state_dict(new_state_dict, strict=strict)

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def stash_grad(self, grad_dict):
        for k, v in self.named_parameters():
            if k in grad_dict:
                grad_dict[k] += v.grad.clone()
            else:
                grad_dict[k] = v.grad.clone()
        self.zero_grad()
        return grad_dict

    def restore_grad(self, grad_dict):
        for k, v in self.named_parameters():
            grad = grad_dict[k] if k in grad_dict else torch.zeros_like(v.grad)
            if v.grad is None:
                v.grad = grad
            else:
                v.grad += grad


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = dropout

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
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ResNetLower(AbstractModel):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetLower, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_dropout_after_first_bottleneck=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_dropout_after_first_bottleneck=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        dropout_module = nn.Dropout2d(0.1) if use_dropout_after_first_bottleneck else None
        layers.append(block(self.inplanes, planes, stride, downsample, dropout=dropout_module))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def set_bn_momentum(self, momentum):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = momentum

    def freeze_bn(self, freeze_bn):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval() if freeze_bn else m.train()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        h1 = self.layer4(x)
        h2 = self.layer4(x)
        return h1, h2

    def load_imagenet_state_dict(self, state_dict, strict=True):
        current_state_dict = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('layer4.2') or k.startswith('fc.'))}
        current_state_dict.update(state_dict)
        super().load_state_dict(current_state_dict, strict=strict)


class ResNetUpper(AbstractModel):

    def __init__(self, in_features=1000, num_classes=12):
        super().__init__()
        self.inplanes = 512 * 4
        self.layer4_last_conv = self._make_single_layer(Bottleneck, 512)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512 * 4, 1000)
        self.fc2 = nn.Linear(1000, in_features)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features, num_classes)
        self.n_classes = num_classes
        self.drop_size = in_features

    def _make_single_layer(self, block, planes):
        return block(self.inplanes, planes)

    def load_imagenet_state_dict(self, state_dict, strict=True):
        current_state_dict = self.state_dict()
        state_dict = {k.replace('layer4.2', 'layer4_last_conv'): v for k, v in state_dict.items() if k.startswith('layer4.2')}
        current_state_dict.update(state_dict)
        super().load_state_dict(current_state_dict, strict=strict)

    def forward(self, x, mask=None):
        x = self.layer4_last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        if mask is not None:
            x = mask * x
        x = self.fc3(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ClassBalanceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiscrepancyLossWithLogits,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (EntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KLDivLossWithLogits,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LossWithLogits,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_postBG_DTA_pytorch(_paritybench_base):
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

