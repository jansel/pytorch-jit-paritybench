import sys
_module = sys.modules[__name__]
del sys
cifar_train = _module
imagenet_lt_data = _module
imagenet_lt_test = _module
imagenet_lt_train = _module
imbalance_cifar = _module
inaturalist_data = _module
inaturalist_train = _module
losses = _module
RSG = _module
models = _module
densenet_cifar = _module
resnet = _module
resnet_cifar = _module
resnext_cifar = _module
utils = _module
places_data = _module
places_test = _module
places_train = _module
utils = _module
RSG = _module

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


import random


import time


import warnings


import numpy as np


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from torch.optim import lr_scheduler


from sklearn.metrics import confusion_matrix


from collections import OrderedDict


from torch.utils import data


from torchvision import transforms as T


from torchvision.datasets import ImageFolder


import torchvision


import math


import torch.nn.functional as F


import torch.nn.init as init


from torch.nn import Parameter


from torch.autograd import Variable


from torch.nn import init


import matplotlib


import matplotlib.pyplot as plt


from sklearn.utils.multiclass import unique_labels


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, weight=None, gamma=0.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class RSG(nn.Module):

    def __init__(self, n_center=3, feature_maps_shape=[32, 16, 16], num_classes=10, contrastive_module_dim=128, head_class_lists=[], transfer_strength=1.0, epoch_thresh=100):
        super(RSG, self).__init__()
        self.num_classes = num_classes
        self.C, self.H, self.W = feature_maps_shape
        self.n_center = n_center
        self.pooling = nn.AvgPool2d(self.H)
        self.linear = nn.Parameter(torch.randn(num_classes, self.C, n_center))
        self.bias = nn.Parameter(torch.ones(num_classes, n_center))
        self.centers = nn.Parameter(torch.zeros(num_classes, n_center, self.C))
        self.softmax = nn.Softmax(dim=1)
        self.strength = transfer_strength
        self.epoch_thresh = epoch_thresh
        self.contrastive_module_dim = contrastive_module_dim
        self.vec_transformation_module = nn.Sequential(nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1))
        self.contrastive_module = nn.Sequential(nn.Conv2d(self.C * 2, contrastive_module_dim, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(contrastive_module_dim, contrastive_module_dim, kernel_size=3, stride=1, padding=1), nn.AvgPool2d(self.H))
        self.contrastive_fc = nn.Linear(self.contrastive_module_dim, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def compute_cesc_loss(self, feature_maps, c, gamma, target, epoch):
        num, C, H, W = feature_maps.size()
        gamma = gamma.unsqueeze(1)
        if epoch <= self.epoch_thresh:
            feature1 = feature_maps[0:num // 2, :, :, :]
            feature2 = feature_maps[num // 2:num, :, :, :]
            target1 = target[0:num // 2]
            target2 = target[num // 2:num]
            feature_cat = torch.cat([feature1, feature2], dim=1)
            target_cat = torch.eq(target1, target2).long()
            pair_fea = self.contrastive_module(feature_cat).contiguous().view(-1, self.contrastive_module_dim)
            pair_pred = self.contrastive_fc(pair_fea)
            loss = torch.sum(torch.bmm(gamma, torch.pow(feature_maps.unsqueeze(1).expand(-1, c.size()[1], -1, -1, -1) - c, 2).view(num, self.n_center, -1))) / num + F.cross_entropy(pair_pred, target_cat)
        else:
            loss = torch.sum(torch.bmm(gamma, torch.pow(feature_maps.unsqueeze(1).expand(-1, c.size()[1], -1, -1, -1) - c, 2).view(num, self.n_center, -1))) / num
        return loss

    def to_one_hot_vector(self, num_class, label):
        label = label.cpu().numpy()
        b = np.zeros((label.shape[0], num_class))
        b[np.arange(label.shape[0]), label] = 1
        b = torch.from_numpy(b)
        return b

    def compute_mv_loss(self, origin_feature, origin_center, target_center, target_features, gamma_head, target, gamma_tail):
        c = origin_center.detach()
        num, C, H, W = target_features.size()
        gamma_h = gamma_head.detach()
        gamma_t = gamma_tail.detach()
        ori_f = origin_feature.detach()
        c_ = target_center.detach()
        for p in self.contrastive_module.parameters():
            p.requires_grad = False
        for p in self.contrastive_fc.parameters():
            p.requires_grad = False
        index = gamma_h.argmax(dim=1)
        index_ = gamma_t.argmax(dim=1)
        index = self.to_one_hot_vector(self.n_center, index).unsqueeze(1)
        index_ = self.to_one_hot_vector(self.n_center, index_).unsqueeze(1)
        c_o = torch.bmm(index, c.view(-1, self.n_center, self.H * self.W * self.C).double()).view(ori_f.size())
        c_t = torch.bmm(index_, c_.view(-1, self.n_center, self.H * self.W * self.C).double()).view(ori_f.size())
        var_map = ori_f - c_o.float()
        var_map_t = self.vec_transformation_module(var_map)
        target_features = target_features
        target_features_vector = target_features - c_t.float()
        target_features_f = target_features + var_map_t
        target_features_norm = F.normalize(target_features_vector.view(-1, self.C), dim=1)
        var_map_norm = F.normalize(var_map_t.view(-1, self.C), dim=1)
        paired = torch.cat([ori_f, var_map_t], dim=1)
        pair_fea = self.contrastive_module(paired).contiguous().view(-1, self.contrastive_module_dim)
        pair_pred = self.contrastive_fc(pair_fea)
        loss = F.cross_entropy(pair_pred, torch.zeros(num).long()) + (torch.sum(torch.abs(torch.norm(var_map_t.view(-1, self.C), dim=1) - torch.norm(var_map.view(-1, self.C), dim=1))) + torch.sum(torch.abs(target_features_norm * var_map_norm - torch.ones(target_features_norm.size())))) / num
        return loss, target_features_f

    def forward(self, feature_maps, head_class_lists, target, epoch):
        maps_detach = feature_maps.detach()
        total = target.size()[0]
        num_head_list = len(head_class_lists)
        index_head = []
        index_tail = []
        head_class_lists_tensor = torch.Tensor(head_class_lists)
        head_class_lists_tensor = head_class_lists_tensor.unsqueeze(0).repeat(total, 1)
        target_expand = target.unsqueeze(1).repeat(1, num_head_list)
        index_head = torch.sum((target_expand == head_class_lists_tensor).long(), dim=1)
        index_tail = 1 - index_head
        index_head_ = torch.eq(index_head, 1)
        index_tail_ = torch.eq(index_tail, 1)
        maps_detach_p = self.pooling(maps_detach).view(-1, self.C)
        target_select = target.unsqueeze(1)
        linear = self.linear[target_select, :, :].view(-1, self.C, self.n_center)
        bias = self.bias[target_select, :].view(-1, self.n_center)
        maps_detach_fc = torch.bmm(maps_detach_p.unsqueeze(1), linear).view(-1, self.n_center) + bias
        gamma = self.softmax(maps_detach_fc)
        centers_ = self.centers[target_select, :, :].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])
        loss_cesc = self.compute_cesc_loss(maps_detach, centers_, gamma, target, epoch)
        loss_mv_total = torch.zeros(loss_cesc.size())
        maps_tail = maps_detach[index_tail_, :, :, :]
        maps_head = maps_detach[index_head_, :, :, :]
        target_tail = target[index_tail_]
        target_head = target[index_head_]
        segment = 1
        num_tail = maps_tail.size()[0]
        num_head = maps_head.size()[0]
        if num_tail != 0 and num_head != 0 and epoch > self.epoch_thresh:
            if num_head >= num_tail:
                segment = int(num_head * self.strength / num_tail)
                if segment == 0:
                    segment = 1
                for j in range(0, segment):
                    latent_2 = maps_tail
                    feature_origin = maps_head[j * num_tail:(j + 1) * num_tail, :, :, :]
                    maps_head_p = self.pooling(feature_origin).view(-1, self.C)
                    target_head_select = target_head[j * num_tail:(j + 1) * num_tail].unsqueeze(1)
                    linear = self.linear[target_head_select, :, :].view(-1, self.C, self.n_center)
                    bias = self.bias[target_head_select, :].view(-1, self.n_center)
                    maps_head_fc = torch.bmm(maps_head_p.unsqueeze(1), linear).view(-1, self.n_center) + bias
                    gamma_head = self.softmax(maps_head_fc)
                    center_origin = self.centers[target_head_select, :, :].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])
                    maps_tail_p = self.pooling(latent_2).view(-1, self.C)
                    target_tail_select = target_tail.unsqueeze(1)
                    linear_ = self.linear[target_tail_select, :, :].view(-1, self.C, self.n_center)
                    bias_ = self.bias[target_tail_select, :].view(-1, self.n_center)
                    maps_tail_fc = torch.bmm(maps_tail_p.unsqueeze(1), linear_).view(-1, self.n_center) + bias_
                    gamma_tail = self.softmax(maps_tail_fc)
                    target_center = self.centers[target_tail_select, :, :].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])
                    loss_mv, feature_f = self.compute_mv_loss(feature_origin, center_origin, target_center, latent_2, gamma_head, target_tail, gamma_tail)
                    loss_mv_total += loss_mv
                    feature_maps = torch.cat((feature_maps, feature_f), dim=0)
                    target = torch.cat((target, target_tail), dim=0)
            else:
                segment = int(num_tail * self.strength / num_head)
                if segment == 0:
                    segment = 1
                for j in range(0, segment):
                    latent_2 = maps_tail[j * num_head:(j + 1) * num_head, :, :, :]
                    feature_origin = maps_head
                    maps_head_p = self.pooling(feature_origin).view(-1, self.C)
                    target_head_select = target_head.unsqueeze(1)
                    linear = self.linear[target_head_select, :, :].view(-1, self.C, self.n_center)
                    bias = self.bias[target_head_select, :].view(-1, self.n_center)
                    maps_head_fc = torch.bmm(maps_head_p.unsqueeze(1), linear).view(-1, self.n_center) + bias
                    gamma_head = self.softmax(maps_head_fc)
                    center_origin = self.centers[target_head_select, :, :].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])
                    maps_tail_p = self.pooling(latent_2).view(-1, self.C)
                    target_tail_select = target_tail[j * num_head:(j + 1) * num_head].unsqueeze(1)
                    linear_ = self.linear[target_tail_select, :, :].view(-1, self.C, self.n_center)
                    bias_ = self.bias[target_tail_select, :].view(-1, self.n_center)
                    maps_tail_fc = torch.bmm(maps_tail_p.unsqueeze(1), linear_).view(-1, self.n_center) + bias_
                    gamma_tail = self.softmax(maps_tail_fc)
                    target_center = self.centers[target_tail_select, :, :].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])
                    loss_mv, feature_f = self.compute_mv_loss(feature_origin, center_origin, target_center, latent_2, gamma_head, target_tail, gamma_tail)
                    feature_maps = torch.cat((feature_maps, feature_f), dim=0)
                    loss_mv_total += loss_mv
                    target = torch.cat((target, target_tail[j * num_head:(j + 1) * num_head]), dim=0)
        return feature_maps, loss_cesc, loss_mv_total / segment, target


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

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


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, use_norm=False, head_tail_ratio=0.3, transfer_strength=1.0, phase_train=True, epoch_thresh=0):
        super(DenseNet, self).__init__()
        if bottleneck:
            nDenseBlocks = int((depth - 4) / 6)
        else:
            nDenseBlocks = int((depth - 4) / 3)
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.phase_train = phase_train
        if use_norm:
            self.fc = NormedLinear(nChannels, nClasses)
        else:
            self.fc = nn.Linear(nChannels, nClasses)
        if self.phase_train:
            self.head_lists = [x for x in range(int(nClasses * head_tail_ratio))]
            self.RSG = RSG(n_center=15, feature_maps_shape=[312, 8, 8], num_classes=nClasses, contrastive_module_dim=256, head_class_lists=self.head_lists, transfer_strength=transfer_strength, epoch_thresh=epoch_thresh)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x, epoch=0, batch_target=None, phase_train=True):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        if phase_train:
            out, cesc_total, loss_mv_total, combine_target = self.RSG.forward(out, self.head_lists, batch_target, epoch)
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = self.fc(out)
        if phase_train:
            return out, cesc_total, loss_mv_total, combine_target
        else:
            return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), 'constant', 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, head_lists=[], zero_init_residual=False, groups=1, width_per_group=64, phase_train=True, epoch_thresh=0, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.phase_train = phase_train
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
        if self.phase_train:
            self.head_lists = head_lists
            self.RSG = RSG(n_center=15, feature_maps_shape=[256 * block.expansion, 14, 14], num_classes=num_classes, contrastive_module_dim=256, head_class_lists=self.head_lists, epoch_thresh=epoch_thresh)
        self.fc_ = NormedLinear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
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

    def _forward_impl(self, x, epoch=0, batch_target=None, phase_train=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if phase_train:
            x, cesc_total, loss_mv_total, combine_target = self.RSG.forward(x, self.head_lists, batch_target, epoch)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_(x)
        if phase_train:
            return x, cesc_total, loss_mv_total, combine_target
        else:
            return x

    def forward(self, x, epoch=0, batch_target=None, phase_train=True):
        return self._forward_impl(x, epoch, batch_target, phase_train)


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False, head_tail_ratio=0.3, transfer_strength=1.0, phase_train=True, epoch_thresh=0):
        super(ResNet_s, self).__init__()
        self.in_planes = 16
        self.phase_train = phase_train
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.linear = NormedLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        if self.phase_train:
            self.head_lists = [x for x in range(int(num_classes * head_tail_ratio))]
            self.RSG = RSG(n_center=15, feature_maps_shape=[32, 16, 16], num_classes=num_classes, contrastive_module_dim=256, head_class_lists=self.head_lists, transfer_strength=transfer_strength, epoch_thresh=epoch_thresh)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, epoch=0, batch_target=None, phase_train=True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        if phase_train:
            out, cesc_total, loss_mv_total, combine_target = self.RSG.forward(out, self.head_lists, batch_target, epoch)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if phase_train:
            return out, cesc_total, loss_mv_total, combine_target
        else:
            return out


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality
        self.conv_reduce = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D * C)
        self.conv_conv = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D * C)
        self.conv_expand = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):

    def __init__(self, block, depth, cardinality, base_width, num_classes, use_norm=False, head_tail_ratio=0.3, transfer_strength=1.0, phase_train=True, epoch_thresh=0):
        super(CifarResNeXt, self).__init__()
        assert (depth - 2) % 9 == 0
        layer_blocks = (depth - 2) // 9
        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes
        self.phase_train = phase_train
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        if use_norm:
            self.classifier = NormedLinear(256 * block.expansion, num_classes)
        else:
            self.classifier = nn.Linear(256 * block.expansion, num_classes)
        if self.phase_train:
            self.head_lists = [x for x in range(int(num_classes * head_tail_ratio))]
            self.RSG = RSG(n_center=15, feature_maps_shape=[512, 16, 16], num_classes=num_classes, contrastive_module_dim=256, head_class_lists=self.head_lists, transfer_strength=transfer_strength, epoch_thresh=epoch_thresh)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x, epoch=0, batch_target=None, phase_train=True):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        if phase_train:
            x, cesc_total, loss_mv_total, combine_target = self.RSG.forward(x, self.head_lists, batch_target, epoch)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        if phase_train:
            return out, cesc_total, loss_mv_total, combine_target
        else:
            return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LambdaLayer,
     lambda: ([], {'lambd': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (SingleLayer,
     lambda: ([], {'nChannels': 4, 'growthRate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'nChannels': 4, 'nOutChannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Jianf_Wang_RSG(_paritybench_base):
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

