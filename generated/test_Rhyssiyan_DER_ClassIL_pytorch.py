import sys
_module = sys.modules[__name__]
del sys
main = _module
main = _module
main = _module
main = _module
main = _module
inclearn = _module
convnet = _module
cifar_resnet = _module
classifier = _module
imbalance = _module
modified_resnet_cifar = _module
network = _module
preact_resnet = _module
resnet = _module
utils = _module
datasets = _module
data = _module
dataset = _module
learn = _module
pretrain = _module
models = _module
align = _module
base = _module
incmodel = _module
tools = _module
data_utils = _module
factory = _module
memory = _module
metrics = _module
results_utils = _module
scheduler = _module
utils = _module

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


import copy


import time


import logging


import numpy as np


import random


import torch


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import init


from torch.nn.parameter import Parameter


from torch.nn import functional as F


from torch.nn import Module


from torch import nn


from torch.optim.lr_scheduler import CosineAnnealingLR


import torch.utils.model_zoo as model_zoo


from torch.optim import SGD


from copy import deepcopy


import warnings


from torch.utils.data import DataLoader


from torch.utils.data.sampler import SubsetRandomSampler


from torch.utils.data.sampler import WeightedRandomSampler


from torchvision import datasets


from torchvision import transforms


from torchvision.datasets.folder import pil_loader


from scipy.spatial.distance import cdist


from torch.nn import DataParallel


import abc


from torch import optim


import numbers


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import ReduceLROnPlateau


class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.featureSize = 64

    def forward(self, x):
        residual = x
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes, channels=3):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()
        self.featureSize = 64
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        self.num_classes = num_classes
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, feature=False, T=1, labels=False, scale=None, keep=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forwardFeature(self, x):
        pass


class CosineClassifier(Module):

    def __init__(self, in_features, n_classes, sigma=True):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.weight = Parameter(torch.Tensor(n_classes, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class BiC(nn.Module):

    def __init__(self, lr, scheduling, lr_decay_factor, weight_decay, batch_size, epochs):
        super(BiC, self).__init__()
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.lr = lr
        self.scheduling = scheduling
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = weight_decay
        self.class_specific = False
        self.batch_size = batch_size
        self.epochs = epochs
        self.bic_flag = False

    def reset(self, lr=None, scheduling=None, lr_decay_factor=None, weight_decay=None, n_classes=-1):
        with torch.no_grad():
            if lr is None:
                lr = self.lr
            if scheduling is None:
                scheduling = self.scheduling
            if lr_decay_factor is None:
                lr_decay_factor = self.lr_decay_factor
            if weight_decay is None:
                weight_decay = self.weight_decay
            if self.class_specific:
                assert n_classes != -1
                self.beta = torch.nn.Parameter(torch.ones(n_classes))
                self.gamma = torch.nn.Parameter(torch.zeros(n_classes))
            else:
                self.beta = torch.nn.Parameter(torch.ones(1))
                self.gamma = torch.nn.Parameter(torch.zeros(1))
            self.optimizer = torch.optim.SGD([self.beta, self.gamma], lr=lr, momentum=0.9, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, scheduling, gamma=lr_decay_factor)

    def extract_preds_and_targets(self, model, loader):
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                preds.append(model(x)['logit'])
                targets.append(y)
        return torch.cat(preds), torch.cat(targets)

    def update(self, logger, task_size, model, loader, loss_criterion=None):
        if task_size == 0:
            logger.info('no new task for BiC!')
            return
        if loss_criterion is None:
            loss_criterion = F.cross_entropy
        self.bic_flag = True
        logger.info('Begin BiC ...')
        model.eval()
        for epoch in range(self.epochs):
            preds_, targets_ = self.extract_preds_and_targets(model, loader)
            order = np.arange(preds_.shape[0])
            np.random.shuffle(order)
            preds, targets = preds_.clone(), targets_.clone()
            preds, targets = preds[order], targets[order]
            _loss = 0.0
            _correct = 0
            _count = 0
            for start in range(0, preds.shape[0], self.batch_size):
                if start + self.batch_size < preds.shape[0]:
                    out = preds[start:start + self.batch_size, :].clone()
                    lbls = targets[start:start + self.batch_size]
                else:
                    out = preds[start:, :].clone()
                    lbls = targets[start:]
                if self.class_specific is False:
                    out1 = out[:, :-task_size].clone()
                    out2 = out[:, -task_size:].clone()
                    outputs = torch.cat((out1, out2 * self.beta + self.gamma), 1)
                else:
                    outputs = out * self.beta + self.gamma
                loss = loss_criterion(outputs, lbls)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                _, pred = outputs.max(1)
                _correct += (pred == lbls).sum()
                _count += lbls.size(0)
                _loss += loss.item() * outputs.shape[0]
            logger.info('epoch {} loss {:4f} acc {:4f}'.format(epoch, _loss / preds.shape[0], _correct / _count))
            self.scheduler.step()
        logger.info('beta {:.4f} gamma {:.4f}'.format(self.beta.cpu().item(), self.gamma.cpu().item()))

    @torch.no_grad()
    def post_process(self, preds, task_size):
        if self.class_specific is False:
            if task_size != 0:
                preds[:, -task_size:] = preds[:, -task_size:] * self.beta + self.gamma
        else:
            preds = preds * self.beta + self.gamma
        return preds


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, remove_last_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu

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
        if not self.remove_last_relu:
            out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

    def __init__(self, block, layers, nf=64, zero_init_residual=True, dataset='cifar', start_class=0, remove_last_relu=False):
        super(ResNet, self).__init__()
        self.remove_last_relu = remove_last_relu
        self.inplanes = nf
        if 'cifar' in dataset:
            self.conv1 = nn.Sequential(nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True))
        elif 'imagenet' in dataset:
            if start_class == 0:
                self.conv1 = nn.Sequential(nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            else:
                self.conv1 = nn.Sequential(nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 1 * nf, layers[0])
        self.layer2 = self._make_layer(block, 2 * nf, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * nf, layers[3], stride=2, remove_last_relu=remove_last_relu)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 8 * nf * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def reset_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class WA(object):

    def __init__(self):
        self.gamma = None

    @torch.no_grad()
    def update(self, classifier, task_size):
        old_weight_norm = torch.norm(classifier.weight[:-task_size], p=2, dim=1)
        new_weight_norm = torch.norm(classifier.weight[-task_size:], p=2, dim=1)
        self.gamma = old_weight_norm.mean() / new_weight_norm.mean()
        None

    @torch.no_grad()
    def post_process(self, logits, task_size):
        logits[:, -task_size:] = logits[:, -task_size:] * self.gamma
        return logits


class BasicNet(nn.Module):

    def __init__(self, convnet_type, cfg, nf=64, use_bias=False, init='kaiming', device=None, dataset='cifar100'):
        super(BasicNet, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']
        if self.der:
            None
            self.convnets = nn.ModuleList()
            self.convnets.append(factory.get_convnet(convnet_type, nf=nf, dataset=dataset, start_class=self.start_class, remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type, nf=nf, dataset=dataset, remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None
        self.n_classes = 0
        self.ntask = 0
        self.device = device
        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == 'bic':
                self.postprocessor = BiC(cfg['postprocessor']['lr'], cfg['postprocessor']['scheduling'], cfg['postprocessor']['lr_decay_factor'], cfg['postprocessor']['weight_decay'], cfg['postprocessor']['batch_size'], cfg['postprocessor']['epochs'])
            elif cfg['postprocessor']['type'].lower() == 'wa':
                self.postprocessor = WA()
        else:
            self.postprocessor = None
        self

    def forward(self, x):
        if self.classifier is None:
            raise Exception('Add some classes before training.')
        if self.der:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)
        logits = self.classifier(features)
        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1
        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)
        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type, nf=self.nf, dataset=self.dataset, start_class=self.start_class, remove_last_relu=self.remove_last_relu)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)
        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        del self.classifier
        self.classifier = fc
        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)
        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)
        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias
        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias)
            if self.init == 'kaiming':
                nn.init.kaiming_normal_(classifier.weight, nonlinearity='linear')
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)
        return classifier


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, remove_last_relu=False):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.remove_last_relu = remove_last_relu
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        out = self.bn3(out)
        if not self.remove_last_relu:
            out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, nf=64, zero_init_residual=True, dataset='cifar', start_class=0, remove_last_relu=False):
        super(PreActResNet, self).__init__()
        self.in_planes = nf
        self.dataset = dataset
        self.remove_last_relu = remove_last_relu
        if 'cifar' in dataset:
            self.conv1 = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * nf, num_blocks[3], stride=2, remove_last_relu=remove_last_relu)
        self.out_dim = 8 * nf
        if 'cifar' in dataset:
            self.avgpool = nn.AvgPool2d(4)
        elif 'imagenet' in dataset:
            self.avgpool = nn.AvgPool2d(7)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, remove_last_relu=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        if remove_last_relu:
            for i in range(len(strides) - 1):
                layers.append(block(self.in_planes, planes, strides[i]))
                self.in_planes = planes * block.expansion
            layers.append(block(self.in_planes, planes, strides[-1], remove_last_relu=True))
            self.in_planes = planes * block.expansion
        else:
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
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CosineClassifier,
     lambda: ([], {'in_features': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DownsampleA,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'stride': 2}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetBasicblock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Rhyssiyan_DER_ClassIL_pytorch(_paritybench_base):
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

