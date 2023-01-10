import sys
_module = sys.modules[__name__]
del sys
APCNN = _module
APINet = _module
BCNN = _module
Baseline = _module
CBCNN = _module
CIN = _module
CrossX = _module
DCL = _module
InterpPartsNet = _module
MGE_CNN = _module
MPN = _module
NTSNet = _module
OSMENet = _module
PairConfusion = _module
PeerLearning = _module
ProtoTreeNet = _module
S3N = _module
Examples = _module
config = _module
dataset = _module
collate_fn = _module
dataset = _module
dataset_DCL = _module
sampler = _module
transforms = _module
model = _module
backbone = _module
resnet = _module
vgg = _module
APINet_loss = _module
CIN_loss = _module
CrossX_loss = _module
DCL_loss = _module
InterpParts_loss = _module
MAMC_loss = _module
NTS_loss = _module
S3N_loss = _module
loss = _module
pair_confusion = _module
peer_learning_loss = _module
APCNN = _module
APINet = _module
BCNN = _module
CBCNN = _module
CIN = _module
CrossX = _module
DCL = _module
Interp_Parts = _module
MGE = _module
grad_cam = _module
MPNCOV = _module
NTSNet = _module
NTS_Net = _module
anchors = _module
resnet = _module
OSME = _module
PeerLearningNet = _module
ProtoTreeNet = _module
ProtoTree = _module
branch = _module
l2conv = _module
leaf = _module
node = _module
prototree = _module
S3N = _module
methods = _module
nms = _module
registry = _module
utils = _module
test = _module
train = _module
repository = _module
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


import torch


import torch.nn as nn


import numpy as np


from torchvision.transforms import autoaugment


from torchvision.transforms import transforms


from torchvision.transforms.functional import InterpolationMode


from torch.utils.data.dataloader import DataLoader


from torchvision import transforms


import torchvision


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torchvision.transforms


from torch import nn


from torch.utils.data.dataloader import default_collate


import pandas as pd


from torch.utils.data.sampler import WeightedRandomSampler


import random


import torch.utils.data as data


from torch.utils.data.sampler import BatchSampler


import math


import numbers


from typing import Tuple


from torch import Tensor


from torchvision.transforms import functional as F


from typing import Type


from typing import Any


from typing import Callable


from typing import Union


from typing import List


from typing import Optional


from torch.hub import load_state_dict_from_url


from typing import Dict


from typing import cast


from scipy import stats


import torch.fft as afft


from torch.autograd import Variable


import torch.utils.model_zoo as model_zoo


import copy


from torch.autograd import Function


from torch.nn import functional as F


import logging


import time


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float=0.5, alpha: float=1.0, inplace: bool=False) ->None:
        super().__init__()
        assert num_classes > 0, 'Please provide a valid positive value for the num_classes.'
        assert alpha > 0, "Alpha param can't be zero."
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, data) ->Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        batch = data['img']
        target = data['label']
        if batch.ndim != 4:
            raise ValueError(f'Batch ndim should be 4. Got {batch.ndim}')
        if target.ndim != 1:
            raise ValueError(f'Target ndim should be 1. Got {target.ndim}')
        if not batch.is_floating_point():
            raise TypeError(f'Batch dtype should be a float tensor. Got {batch.dtype}.')
        if target.dtype != torch.int64:
            raise TypeError(f'Target dtype should be torch.int64. Got {target.dtype}')
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        if torch.rand(1).item() >= self.p:
            return {'img': batch, 'label': target}
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = F.get_image_size(batch)
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))
        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        return {'img': batch, 'label': target}

    def __repr__(self) ->str:
        s = self.__class__.__name__ + '('
        s += 'num_classes={num_classes}'
        s += ', p={p}'
        s += ', alpha={alpha}'
        s += ', inplace={inplace}'
        s += ')'
        return s.format(**self.__dict__)


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float=0.5, alpha: float=1.0, inplace: bool=False) ->None:
        super().__init__()
        assert num_classes > 0, 'Please provide a valid positive value for the num_classes.'
        assert alpha > 0, "Alpha param can't be zero."
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, data) ->Tuple[Tensor, Tensor]:
        """
        Args:
            data['img'] (Tensor): Float tensor of size (B, C, H, W)
            data['label'] (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        batch = data['img']
        target = data['label']
        if batch.ndim != 4:
            raise ValueError(f'Batch ndim should be 4. Got {batch.ndim}')
        if target.ndim != 1:
            raise ValueError(f'Target ndim should be 1. Got {target.ndim}')
        if not batch.is_floating_point():
            raise TypeError(f'Batch dtype should be a float tensor. Got {batch.dtype}.')
        if target.dtype != torch.int64:
            raise TypeError(f'Target dtype should be torch.int64. Got {target.dtype}')
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        if torch.rand(1).item() >= self.p:
            return {'img': batch, 'label': target}
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        return {'img': batch, 'label': target}

    def __repr__(self) ->str:
        s = self.__class__.__name__ + '('
        s += 'num_classes={num_classes}'
        s += ', p={p}'
        s += ', alpha={alpha}'
        s += ', inplace={inplace}'
        s += ')'
        return s.format(**self.__dict__)


class MixupCutmixCollateFn(nn.Module):

    def __init__(self, num_classes):
        super(MixupCutmixCollateFn, self).__init__()
        self.mixupcutmix = transforms.RandomChoice([RandomMixup(num_classes=num_classes, p=1.0, alpha=0.2), RandomCutmix(num_classes=num_classes, p=1.0, alpha=1.0)])

    def __call__(self, batch):
        return self.mixupcutmix(default_collate(batch))


class RandomSwap(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = int(size), int(size)
        else:
            assert len(size) == 2, 'Please provide only two dimensions (h, w) for size.'
            self.size = size

    def __call__(self, img):
        return self.swap(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    def swap(self, img, crop):

        def crop_image(image, cropnum):
            width, high = image.size
            crop_x = [int(width / cropnum[0] * i) for i in range(cropnum[0] + 1)]
            crop_y = [int(high / cropnum[1] * i) for i in range(cropnum[1] + 1)]
            im_list = []
            for j in range(len(crop_y) - 1):
                for i in range(len(crop_x) - 1):
                    im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
            return im_list
        widthcut, highcut = img.size
        img = img.crop((10, 10, widthcut - 10, highcut - 10))
        images = crop_image(img, crop)
        pro = 5
        if pro >= 5:
            tmpx = []
            tmpy = []
            count_x = 0
            count_y = 0
            k = 1
            RAN = 2
            for i in range(crop[1] * crop[0]):
                tmpx.append(images[i])
                count_x += 1
                if len(tmpx) >= k:
                    tmp = tmpx[count_x - RAN:count_x]
                    random.shuffle(tmp)
                    tmpx[count_x - RAN:count_x] = tmp
                if count_x == crop[0]:
                    tmpy.append(tmpx)
                    count_x = 0
                    count_y += 1
                    tmpx = []
                if len(tmpy) >= k:
                    tmp2 = tmpy[count_y - RAN:count_y]
                    random.shuffle(tmp2)
                    tmpy[count_y - RAN:count_y] = tmp2
            random_im = []
            for line in tmpy:
                random_im.extend(line)
            width, high = img.size
            iw = int(width / crop[0])
            ih = int(high / crop[1])
            toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
            x = 0
            y = 0
            for i in random_im:
                i = i.resize((iw, ih), Image.ANTIALIAS)
                toImage.paste(i, (x * iw, y * ih))
                x += 1
                if x == crop[0]:
                    x = 0
                    y += 1
        else:
            toImage = img
        toImage = toImage.resize((widthcut, highcut))
        return toImage


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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
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
        feature1 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.Dropout(p=0.5)(x)
        feature2 = x
        x = self.fc(x)
        return x, feature1, feature2


def initialize_weights(m) ->None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, val=0)


class VGG(nn.Module):

    def __init__(self, features: nn.Module, num_classes: int=1000, init_weights: bool=True) ->None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class APINetLoss(nn.Module):

    def __init__(self, config):
        super(APINetLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.rank_loss = nn.MarginRankingLoss(margin=0.05)
        self.softmax_layer = nn.Softmax(dim=1)

    def __call__(self, output, target):
        self_logits, other_logits, labels1, labels2 = output
        device = labels1.device
        batch_size = self_logits.shape[0] // 2
        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
        softmax_loss = self.ce_loss(logits, targets)
        self_scores = self.softmax_layer(self_logits)[torch.arange(2 * batch_size).long(), torch.cat([labels1, labels2], dim=0)]
        other_scores = self.softmax_layer(other_logits)[torch.arange(2 * batch_size).long(), torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones((2 * batch_size,))
        rank_loss = self.rank_loss(self_scores, other_scores, flag)
        loss = softmax_loss + rank_loss
        return loss


class CINLoss(nn.Module):

    def __init__(self, config):
        super(CINLoss, self).__init__()
        self.alpha = config.alpha if 'alpha' in config else 2.0
        self.beta = config.beta if 'beta' in config else 0.5
        self.channel = config.channel if 'channel' in config else 2048
        self.feature_size = config.feature_size if 'feature_size' in config else 7 * 7
        self.r_channel = config.r_channel if 'r_channel' in config else 512
        self.pdist = nn.PairwiseDistance(p=2)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.h = nn.Linear(self.channel * self.feature_size, self.r_channel)
        self.apply(initialize_weights)

    def __call__(self, output, target):
        if not isinstance(output, tuple):
            return self.ce_loss(output, target)
        Z, Z_CCI = output
        B, C, WH = Z_CCI.size()
        loss_ce = self.ce_loss(Z, target)
        Z_AB = Z_CCI.view(B, -1)
        Z_AB = self.h(Z_AB)
        pair_label = target[:B // 2] == target[B // 2]
        loss_cont_1 = torch.sum(torch.pow(self.pdist(Z_AB[:B // 2][pair_label], Z_AB[B // 2:][pair_label]), 2))
        loss_cont_2 = self.beta - self.pdist(Z_AB[:B // 2][~pair_label], Z_AB[B // 2:][~pair_label])
        loss_cont_2[loss_cont_2 < 0] = 0
        loss_cont_2 = torch.pow(loss_cont_1, 2)
        loss_cont = loss_cont_1 + loss_cont_2
        loss = loss_ce + self.alpha * loss_cont
        return loss


class RegularLoss(nn.Module):

    def __init__(self, gamma=0, num_parts=1):
        super(RegularLoss, self).__init__()
        self.num_parts = num_parts
        self.gamma = gamma

    def forward(self, x):
        assert isinstance(x, list), 'parts features should be presented in a list'
        corr_matrix = torch.zeros(self.num_parts, self.num_parts)
        for i in range(self.num_parts):
            x[i] = x[i].squeeze()
            x[i] = torch.div(x[i], x[i].norm(dim=1, keepdim=True))
        for i in range(self.num_parts):
            for j in range(self.num_parts):
                corr_matrix[i, j] = torch.mean(torch.mm(x[i], x[j].t()))
                if i == j:
                    corr_matrix[i, j] = 1.0 - corr_matrix[i, j]
        regloss = torch.mul(torch.sum(torch.triu(corr_matrix)), self.gamma)
        return regloss


class CrossXLoss(nn.Module):

    def __init__(self, config):
        super(CrossXLoss, self).__init__()
        self.num_parts = config.num_parts
        self.gamma = config.gamma
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.ulti_loss = RegularLoss(gamma=self.gamma[0], num_parts=self.num_parts)
        self.plty_loss = RegularLoss(gamma=self.gamma[1], num_parts=self.num_parts)
        self.cmbn_loss = RegularLoss(gamma=self.gamma[2], num_parts=self.num_parts)
        self.kl_loss = nn.KLDivLoss(reduction='sum')

    def __call__(self, outputs, target):
        if self.num_parts == 1:
            loss = self.ce_loss(outputs, target)
        else:
            outputs_ulti, outputs_plty, outputs_cmbn, ulti_ftrs, plty_ftrs, cmbn_ftrs = outputs
            outs = outputs_ulti + outputs_plty + outputs_cmbn
            cls_loss = self.ce_loss(outs, target)
            reg_loss_cmbn = self.cmbn_loss(cmbn_ftrs)
            outputs_cmbn = F.log_softmax(outputs_cmbn, 1)
            reg_loss_ulti = self.ulti_loss(ulti_ftrs)
            reg_loss_plty = self.plty_loss(plty_ftrs)
            outputs_plty = F.log_softmax(outputs_plty, 1)
            outputs_ulti = F.softmax(outputs_ulti, 1)
            kl_loss = (self.kl_loss(outputs_plty, outputs_ulti) + self.kl_loss(outputs_cmbn, outputs_ulti)) / target.size(0)
            loss = reg_loss_ulti + reg_loss_plty + reg_loss_cmbn + kl_loss + cls_loss
        return loss


class DCLLoss(nn.Module):

    def __init__(self, config):
        super(DCLLoss, self).__init__()
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.add_loss = nn.L1Loss()

    def __call__(self, outputs, labels, labels_swap, swap_law):
        loss_ce = self.ce_loss(outputs[0], labels)
        loss_swap = self.ce_loss(outputs[1], labels_swap)
        loss_law = self.add_loss(outputs[2], swap_law)
        loss = self.alpha * loss_ce + self.beta * loss_swap + self.gamma * loss_law
        return loss


def GaussianKernel(radius, std):
    """
    Generate a gaussian blur kernel based on the given radius and std.

    Args
    ----------
    radius: int
        Radius of the Gaussian kernel. Center not included.
    std: float
        Standard deviation of the Gaussian kernel.

    Returns
    ----------
    weight: torch.FloatTensor, [2 * radius + 1, 2 * radius + 1]
        Output Gaussian kernel.

    """
    size = 2 * radius + 1
    weight = torch.ones(size, size)
    weight.requires_grad = False
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            dis = i * i + j * j
            weight[i + radius][j + radius] = np.exp(-dis / (2 * std * std))
    weight = weight / weight.sum()
    return weight


def update_prior_dist(batch_size, alpha, beta, device=None):
    """
    Update the samples of prior distribution due to the change of batchsize.

    Args
    ----------
    batch_size: int
        Current batch size.
    alpha: float
        Parameter of Beta distribution.
    beta: float
        Parameter of Beta distribution.

    """
    global prior_dist
    grid_points = torch.arange(1.0, 2 * batch_size, 2.0).float() / (2 * batch_size)
    grid_points_np = grid_points.cpu().numpy()
    grid_points_icdf = stats.beta.ppf(grid_points_np, a=alpha, b=beta)
    prior_dist = torch.tensor(grid_points_icdf).float().unsqueeze(1)


def ShapingLoss(assign, radius, std, num_parts, alpha, beta, eps=1e-05):
    """
    Wasserstein shaping loss for Bernoulli distribution.

    Args
    ----------
    assign: torch.cuda.FloatTensor, [batch_size, num_parts, height, width]
        Assignment map for grouping.
    radius: int
        Radius for the Gaussian kernel.
    std: float
        Standard deviation for the Gaussian kernel.
    num_parts: int
        Number of object parts in the current model.
    alpha: float
        Parameter of Beta distribution.
    beta: float
        Parameter of Beta distribution.
    eps:
        Epsilon for rescaling the distribution.

    Returns
    ----------
    loss: torch.cuda.FloatTensor, [1, ]
        Average Wasserstein shaping loss for the current minibatch.

    """
    global prev_bs, prior_dist
    batch_size = assign.shape[0]
    device = assign.device
    if radius == 0:
        assign_smooth = assign
    else:
        weight = GaussianKernel(radius, std)
        weight = weight.contiguous().view(1, 1, 2 * radius + 1, 2 * radius + 1).expand(num_parts, 1, 2 * radius + 1, 2 * radius + 1)
        assign_smooth = F.conv2d(assign, weight, groups=num_parts)
    part_occ = F.adaptive_max_pool2d(assign_smooth, (1, 1)).squeeze(2).squeeze(2)
    emp_dist, _ = part_occ.sort(dim=0, descending=False)
    if batch_size != prev_bs:
        update_prior_dist(batch_size, alpha, beta, device)
    emp_dist = (emp_dist + eps).log()
    prior_dist = (prior_dist + eps).log()
    output_nk = (emp_dist - prior_dist).abs()
    loss = output_nk.mean()
    return loss


class InterpPartsLoss(nn.Module):

    def __init__(self, config):
        super(InterpPartsLoss, self).__init__()
        self.radius = config.radius if 'radius' in config else 2
        self.std = config.std if 'std' in config else 0.4
        self.num_parts = config.num_parts if 'num_parts' in config else 5
        self.alpha = config.alpha if 'alpha' in config else 1
        self.beta = config.beta if 'beta' in config else 0.001
        self.coeff = config.coeff if 'coeff' in config else 0.5
        self.ce_loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        logits, att, assign = output
        loss_ce = self.ce_loss(logits, target)
        shaping_loss = ShapingLoss(assign, self.radius, self.std, self.num_parts, self.alpha, self.beta)
        loss = loss_ce + self.coeff * shaping_loss
        return loss


class NPairsLoss(nn.Module):
    """N-pairs loss as explained in equation 11 of MAMC paper.

    Reference:
        Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition
    """

    def __init__(self):
        super(NPairsLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, part_num, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        b, p, _ = inputs.size()
        n = b * p
        inputs = inputs.contiguous().view(n, -1)
        inputs = F.normalize(inputs, p=2, dim=1)
        targets = torch.repeat_interleave(targets, p)
        parts = torch.arange(p).repeat(b)
        prod = torch.mm(inputs, inputs.t())
        parts = parts
        same_class_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        same_atten_mask = parts.expand(n, n).eq(parts.expand(n, n).t())
        s_sasc = same_class_mask & same_atten_mask
        s_sadc = ~same_class_mask & same_atten_mask
        s_dasc = same_class_mask & ~same_atten_mask
        s_dadc = ~same_class_mask & ~same_atten_mask
        loss_sasc = 0
        loss_sadc = 0
        loss_dasc = 0
        for i in range(n):
            pos = prod[i][s_sasc[i]]
            neg = prod[i][s_sadc[i] | s_dasc[i] | s_dadc[i]]
            n_pos = pos.size(0)
            n_neg = neg.size(0)
            pos = pos.repeat(n_neg, 1).t()
            neg = neg.repeat(n_pos, 1)
            loss_sasc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))
            pos = prod[i][s_sadc[i]]
            neg = prod[i][s_dadc[i]]
            n_pos = pos.size(0)
            n_neg = neg.size(0)
            pos = pos.repeat(n_neg, 1).t()
            neg = neg.repeat(n_pos, 1)
            loss_sadc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))
            pos = prod[i][s_dasc[i]]
            neg = prod[i][s_dadc[i]]
            n_pos = pos.size(0)
            n_neg = neg.size(0)
            pos = pos.repeat(n_neg, 1).t()
            neg = neg.repeat(n_pos, 1)
            loss_dasc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))
        return (loss_sasc + loss_sadc + loss_dasc) / n


class MAMCLoss(nn.Module):

    def __init__(self, config):
        super(MAMCLoss, self).__init__()
        self.lambda_a = config.lambda_a if 'lambda_a' in config else 0.5
        self.use_mamc = config.use_mamc if 'use_mamc' in config else True
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.npair_loss = NPairsLoss()

    def forward(self, inputs, targets):
        pred, x_part = inputs
        loss_ce = self.ce_loss(pred, targets)
        if not self.use_mamc:
            return loss_ce
        loss_npair = self.npair_loss(x_part, targets)
        return loss_ce + self.lambda_a * loss_npair


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [(-temp[i][targets[i].item()]) for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=6):
    loss = torch.zeros(1)
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size


class NTSLoss(nn.Module):

    def __init__(self, config):
        super(NTSLoss, self).__init__()
        self.PROPOSAL_NUM = config.proposal_num
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.list_loss = list_loss
        self.rank_loss = ranking_loss

    def __call__(self, outputs, targets):
        raw_logits, concat_logits, part_logits, _, top_n_prob = outputs
        batch_size = targets.size(0)
        part_loss = self.list_loss(part_logits.view(batch_size * self.PROPOSAL_NUM, -1), targets.unsqueeze(1).repeat(1, self.PROPOSAL_NUM).view(-1))
        part_loss = part_loss.view(batch_size, self.PROPOSAL_NUM)
        raw_loss = self.ce_loss(raw_logits, targets)
        concat_loss = self.ce_loss(concat_logits, targets)
        rank_loss = self.rank_loss(top_n_prob, part_loss, self.PROPOSAL_NUM)
        partcls_loss = self.ce_loss(part_logits.view(batch_size * self.PROPOSAL_NUM, -1), targets.unsqueeze(1).repeat(1, self.PROPOSAL_NUM).view(-1))
        loss = raw_loss + rank_loss + concat_loss + partcls_loss
        return loss


class MultiSmoothLoss(nn.Module):
    """Multi smooth loss.
    """

    def __init__(self, config):
        self.smooth_ratio = config.smooth_ratio

    def __call__(self, output, target, loss_weight=None, weight=None, size_average=True, ignore_index=-100, reduce=True):
        assert isinstance(output, tuple), 'input is less than 2'
        weight_loss = torch.ones(len(output))
        if loss_weight is not None:
            for item in loss_weight.items():
                weight_loss[int(item[0])] = item[1]
        loss = 0
        for i in range(0, len(output)):
            if i in [1, len(output) - 1]:
                prob = F.log_softmax(output[i], dim=1)
                ymask = prob.data.new(prob.size()).zero_()
                ymask = ymask.scatter_(1, target.view(-1, 1), 1)
                ymask = self.smooth_ratio * ymask + (1 - self.smooth_ratio) * (1 - ymask) / (output[i].shape[1] - 1)
                loss_tmp = -weight_loss[i] * (prob * ymask).sum(1).mean()
            else:
                loss_tmp = weight_loss[i] * F.cross_entropy(output[i], target, weight, ignore_index=ignore_index, reduction='mean')
            loss += loss_tmp
        return loss


class PairwiseConfusionLoss(nn.Module):

    def __init__(self, config):
        super(PairwiseConfusionLoss, self).__init__()
        self.lambda_a = config.lambda_a if 'lambda_a' in config else 10
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, features, labels):
        batch_size = features.size(0)
        if float(batch_size) % 2 != 0:
            raise Exception('Incorrect batch size provided')
        batch_left = features[:int(0.5 * batch_size)]
        batch_right = features[int(0.5 * batch_size):]
        label_left = labels[:int(0.5 * batch_size)]
        label_right = labels[int(0.5 * batch_size):]
        loss = torch.norm((batch_left - batch_right).abs(), 2, 1)
        loss = loss * (label_left != label_right)
        loss = loss.sum() / float(batch_size)
        loss_ce = self.cross_entropy(features, labels)
        return loss_ce + self.lambda_a * loss


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SimpleFPA(nn.Module):

    def __init__(self, in_planes, out_planes):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(SimpleFPA, self).__init__()
        self.channels_cond = in_planes
        self.conv_master = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)
        self.conv_gpb = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        x_master = self.conv_master(x)
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        out = x_master + x_gpb
        return out


class PyramidFeatures(nn.Module):
    """Feature pyramid module with top-down feature pathway"""

    def __init__(self, B2_size, B3_size, B4_size, B5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        self.P5_1 = SimpleFPA(B5_size, feature_size)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_1 = nn.Conv2d(B4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1 = nn.Conv2d(B3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        B3, B4, B5 = inputs
        P5_x = self.P5_1(B5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2)
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_1(B4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_1(B3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        return [P3_x, P4_x, P5_x]


class ChannelGate(nn.Module):
    """generation channel attention mask"""

    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels // 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels // 16, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = torch.sigmoid(self.conv2(x))
        return x


class SpatialGate(nn.Module):
    """generation spatial attention mask"""

    def __init__(self, out_channels):
        super(SpatialGate, self).__init__()
        self.conv = nn.ConvTranspose2d(out_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)


class PyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256):
        super(PyramidAttentions, self).__init__()
        self.A3_1 = SpatialGate(channel_size)
        self.A3_2 = ChannelGate(channel_size)
        self.A4_1 = SpatialGate(channel_size)
        self.A4_2 = ChannelGate(channel_size)
        self.A5_1 = SpatialGate(channel_size)
        self.A5_2 = ChannelGate(channel_size)

    def forward(self, inputs):
        F3, F4, F5 = inputs
        A3_spatial = self.A3_1(F3)
        A3_channel = self.A3_2(F3)
        A3 = A3_spatial * F3 + A3_channel * F3
        A4_spatial = self.A4_1(F4)
        A4_channel = self.A4_2(F4)
        A4_channel = (A4_channel + A3_channel) / 2
        A4 = A4_spatial * F4 + A4_channel * F4
        A5_spatial = self.A5_1(F5)
        A5_channel = self.A5_2(F5)
        A5_channel = (A5_channel + A4_channel) / 2
        A5 = A5_spatial * F5 + A5_channel * F5
        return [A3, A4, A5, A3_spatial, A4_spatial, A5_spatial]


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Repository(dict):
    """
    A dict format repository to register module.
    Repository can also manage config node.
    """

    def __init__(self, *args, **kwargs):
        super(Repository, self).__init__(*args, **kwargs)

    def register(self, module):
        assert module.__name__ not in self
        self[module.__name__] = module
        return module


MODEL = Repository()


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


class APINet(nn.Module):

    def __init__(self, config):
        super(APINet, self).__init__()
        self.num_classes = config.num_classes
        resnet = resnet101(pretrained=True)
        layers = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        self.avg = nn.AvgPool2d(kernel_size=7, stride=1)
        self.map1 = nn.Linear(2048 * 2, 512)
        self.map2 = nn.Linear(512, 2048)
        self.fc = nn.Linear(2048, self.num_classes)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.device = None

    def forward(self, images, targets=None, flag='train'):
        self.device = images.device
        batch_size = images.size(0) * 2
        conv_out = self.backbone(images)
        pool_out = self.avg(conv_out).squeeze()
        if flag == 'train':
            intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(pool_out, targets)
            features1 = torch.cat([pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat([pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)
            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)
            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)
            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)
            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1
            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2
            logit1_self = self.fc(self.drop(features1_self))
            logit1_other = self.fc(self.drop(features1_other))
            logit2_self = self.fc(self.drop(features2_self))
            logit2_other = self.fc(self.drop(features2_other))
            self_logits = torch.zeros(2 * batch_size, 200)
            other_logits = torch.zeros(2 * batch_size, 200)
            self_logits[:batch_size] = logit1_self
            self_logits[batch_size:] = logit2_self
            other_logits[:batch_size] = logit1_other
            other_logits[batch_size:] = logit2_other
            return self_logits, other_logits, labels1, labels2
        elif flag == 'val':
            return self.fc(pool_out)

    def get_pairs(self, embeddings, labels):
        distance_matrix = pdist(embeddings).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().reshape(-1, 1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs == False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)
        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)
        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])
        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]
            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]
        intra_labels = torch.from_numpy(intra_labels).long()
        intra_pairs = torch.from_numpy(intra_pairs).long()
        inter_labels = torch.from_numpy(inter_labels).long()
        inter_pairs = torch.from_numpy(inter_pairs).long()
        return intra_pairs, inter_pairs, intra_labels, inter_labels


class BilinearPooling(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, x):
        batch_size = x.size(0)
        channel_size = x.size(1)
        feature_size = x.size(2) * x.size(3)
        x = x.view(batch_size, channel_size, feature_size)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size
        x = x.view(batch_size, -1)
        x = torch.sqrt(x + 1e-05)
        x = torch.nn.functional.normalize(x)
        return x


BACKBONE = Repository()


def make_layers(cfg: List[Union[str, int]], batch_norm: bool=False) ->nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) ->VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@BACKBONE.register
def vgg16(pretrained: bool=False, progress: bool=True, **kwargs: Any) ->VGG:
    """VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


class BCNN(nn.Module):

    def __init__(self, config):
        super(BCNN, self).__init__()
        self.stage = config.stage if 'stage' in config else 2
        self.backbone = vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2][0])
        self.bilinear_pooling = BilinearPooling()
        self.classifier = nn.Linear(512 ** 2, config.num_classes)
        self.classifier.apply(initialize_weights)
        if self.stage == 1:
            for params in self.backbone.parameters():
                params.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        if self.stage == 1:
            x = x.detach()
        x = self.bilinear_pooling(x)
        x = self.classifier(x)
        return x


class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.

    Args:

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.

        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.

        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.

        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool=True, rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1
        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(rand_h_1, rand_s_1, self.output_dim))
        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1
        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(rand_h_2, rand_s_2, self.output_dim))

    def forward(self, bottom1, bottom2=None):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        if bottom2 is None:
            bottom2 = bottom1.clone()
        assert bottom1.size(1) == self.input_dim1 and bottom2.size(1) == self.input_dim2
        if self.sparse_sketch_matrix1.device != bottom1.device:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2
        batch_size, _, height, width = bottom1.size()
        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)
        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)
        fft1 = afft.fft(sketch_1)
        fft2 = afft.fft(sketch_2)
        fft_product = fft1 * fft2
        cbp_flat = afft.ifft(fft_product).real
        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)
        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)
        cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-10)
        cbp = torch.nn.functional.normalize(cbp)
        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert rand_h.ndim == 1 and rand_s.ndim == 1 and len(rand_h) == len(rand_s)
        assert np.all(rand_h >= 0) and np.all(rand_h < output_dim)
        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis], rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()


class CBCNN(nn.Module):

    def __init__(self, config):
        super(CBCNN, self).__init__()
        self.config = config
        in_channel = config.input_channel
        out_channel = config.output_channel
        self.backbone = vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2][0])
        self.bilinear_pooling = CompactBilinearPooling(in_channel, in_channel, out_channel)
        self.classifier = nn.Linear(out_channel, config.num_classes)
        self.classifier.apply(initialize_weights)

    def forward(self, x):
        x = self.backbone(x)
        if self.config.stage == 1:
            x = x.detach()
        x = self.bilinear_pooling(x)
        x = self.classifier(x)
        return x


class ChannelInteractionModule(nn.Module):
    """Channel Interaction Network
    """

    def __init__(self, in_channel=2048, spatial_size=(7, 7)):
        super(ChannelInteractionModule, self).__init__()
        self.in_channel = in_channel
        self.spatial_size = spatial_size
        WH = self.spatial_size[0] * self.spatial_size[1]
        self.softmax = nn.Softmax()
        self.conv = nn.Conv2d(self.in_channel, self.in_channel, 3, 1, 1)
        self.fc = nn.Linear(2 * self.in_channel * WH, 1)

    def forward(self, x):
        B, C, W, H = x.size()
        assert B % 2 == 0, 'batch size should not be odd!'
        x = x.view(B, C, W * H)
        bilinear_matrix = torch.bmm(x, torch.transpose(x, 1, 2)) / (W * H)
        W_SCI = F.softmax(-bilinear_matrix, dim=2)
        Y = torch.bmm(W_SCI, x)
        Y = self.conv(Y.view(B, C, W, H))
        Y = Y.view(B, C, W * H)
        Z = Y + x
        if not self.training:
            return Z
        y = Y.view(B, -1)
        y_a = torch.cat((y[:B // 2], y[B // 2:]), dim=1)
        y_b = torch.cat((y[B // 2:], y[:B // 2]), dim=1)
        eta = self.fc(y_a)
        gamma = self.fc(y_b)
        weight = torch.cat((eta, gamma), dim=0)
        W_SCI_BA = torch.cat((W_SCI[B // 2:], W_SCI[:B // 2]), dim=0)
        W_CCI = torch.abs(W_SCI - weight.view(-1, 1, 1) * W_SCI_BA)
        Y_CCI = torch.bmm(W_CCI, x)
        Y_CCI = self.conv(Y_CCI.view(B, C, W, H))
        Y_CCI = Y_CCI.view(B, C, W * H)
        Z_CCI = Y_CCI + x
        return Z, Z_CCI


class CINClassifier(nn.Module):
    """Channel Interaction Network Classifier
    """

    def __init__(self, in_channel=2048, num_classes=200):
        super(CINClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channel, num_classes)

    def forward(self, x):
        if isinstance(x, tuple):
            Z, Z_CCI = x
            Z = torch.squeeze(self.pool(Z))
            Z = self.classifier(Z)
            return Z, Z_CCI
        else:
            x = torch.squeeze(self.pool(x))
            x = self.classifier(x)
            return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class CIN(nn.Module):

    def __init__(self, config):
        super(CIN, self).__init__()
        self.num_classes = config.num_classes if 'num_classes' in config else 200
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.ChannelInteraction = ChannelInteractionModule(in_channel=2048, spatial_size=(7, 7))
        self.classifier = CINClassifier(in_channel=2048, num_classes=self.num_classes)
        self.ChannelInteraction.apply(initialize_weights)
        self.classifier.apply(initialize_weights)

    def forward(self, x):
        x = self.backbone(x)
        x = self.ChannelInteraction(x)
        x = self.classifier(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MELayer(nn.Module):

    def __init__(self, channel, reduction=16, nparts=1):
        super(MELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nparts = nparts
        parts = list()
        for part in range(self.nparts):
            parts.append(nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid()))
        self.parts = nn.Sequential(*parts)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        meouts = list()
        for i in range(self.nparts):
            meouts.append(x * self.parts[i](y).view(b, c, 1, 1))
        return meouts


class DCL(nn.Module):

    def __init__(self, config):
        super(DCL, self).__init__()
        self.num_classes = config.num_classes
        self.cls_2 = config.cls_2
        self.cls_2xmul = config.cls_2xmul
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=True)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        if self.cls_2:
            self.classifier_swap = nn.Linear(2048, 2, bias=False)
        if self.cls_2xmul:
            self.classifier_swap = nn.Linear(2048, 2 * self.num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        mask = self.Convmask(x)
        mask = self.avgpool2(mask)
        mask = torch.tanh(mask)
        mask = mask.view(mask.size(0), -1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = [self.classifier(x), self.classifier_swap(x), mask]
        return out


class GroupingUnit(nn.Module):

    def __init__(self, in_channels, num_parts):
        super(GroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels, 1, 1))
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-05)
        else:
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, inputs):
        assert inputs.dim() == 4
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels
        grouping_centers = self.weight.contiguous().view(1, self.num_parts, self.in_channels).expand(batch_size, self.num_parts, self.in_channels)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        x = x.permute(0, 2, 1)
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)
        c = grouping_centers
        sum_ass = torch.sum(assign, dim=2, keepdim=True)
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-05)
        sigma = (beta / 2).sqrt()
        out = (qx / sum_ass - c) / sigma.unsqueeze(0).unsqueeze(2)
        assign = assign.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        outputs = nn.functional.normalize(out, dim=2)
        outputs_t = outputs.permute(0, 2, 1)
        return outputs_t, assign

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.num_parts) + ')'


class Bottleneck1x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
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


class Classifier(nn.Module):

    def __init__(self, in_panel, out_panel, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_panel, out_panel, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim() == 1:
            logit = logit.unsqueeze(0)
        return logit


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, feature_extractor, classifier, target_layers):
        self.model = model
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.feature_extractor._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_extractor, classifier, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, feature_extractor, classifier, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        conv5_pool = self.model.pool(output).squeeze()
        logits = self.model.classifier(conv5_pool)
        output = logits
        return target_activations, output


class GradCam:

    def __init__(self, model, feature_extractor, classifier, target_layer_names):
        self.model = model
        self.flag = model.training
        self.model.eval()
        self.extractor = ModelOutputs(self.model, feature_extractor, classifier, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        input.requires_grad = True
        features, output = self.extractor(input)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        if index is None:
            index = torch.argmax(output, dim=-1)
        one_hot = torch.zeros_like(output)
        one_hot[[torch.arange(output.size(0)), index]] = 1
        one_hot = torch.sum(one_hot * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1]
        grads_val = F.relu(grads_val)
        weights = grads_val.mean(-1).mean(-1)
        if self.flag:
            self.model.train()
        return weights.clone().detach()


def get_bbox(x, conv5, layer_weights, rate=0.3, img_size=448):
    mask_size = img_size // 32
    conv5_cam = conv5.clone().detach().view(conv5.size(0), conv5.size(1), -1) * layer_weights.unsqueeze(-1)
    mask_sum = conv5_cam.sum(1, keepdim=True).view(conv5.size(0), 1, mask_size, mask_size)
    mask_sum = F.interpolate(mask_sum, size=(img_size, img_size), mode='bilinear', align_corners=True)
    mask_sum = mask_sum.view(mask_sum.size(0), -1)
    x_range = mask_sum.max(-1, keepdim=True)[0] - mask_sum.min(-1, keepdim=True)[0]
    mask_sum = (mask_sum - mask_sum.min(-1, keepdim=True)[0]) / x_range
    mask = torch.sign(torch.sign(mask_sum - rate) + 1)
    mask = mask.view(mask.size(0), 1, img_size, img_size)
    input_box = torch.zeros_like(x)
    xy_list = []
    for k in torch.arange(x.size(0)):
        indices = mask[k].nonzero()
        y1, x1 = indices.min(dim=0)[0][-2:]
        y2, x2 = indices.max(dim=0)[0][-2:]
        tmp = x[k, :, y1:y2, x1:x2]
        if x1 == x2 or y1 == y2:
            tmp = x[k, :, :, :]
        input_box[k] = F.interpolate(tmp.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=True).clone().detach()
        xy_list.append([x1, x2, y1, y2])
    return input_box, xy_list


def l2_norm_v2(input):
    input_size = input.size()
    _output = input / torch.norm(input, p=2, dim=-1, keepdim=True)
    output = _output.view(input_size)
    return output


class LocalCamNet(nn.Module):

    def __init__(self, config=None):
        super(LocalCamNet, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.box_thred = config.box_thred
        self.image_size = config.image_size
        basenet = resnet50(pretrained=True)
        self.conv4 = nn.Sequential(*list(basenet.children())[:-3])
        self.conv5 = nn.Sequential(*list(basenet.children())[-3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(2048, self.num_classes, bias=True)
        self.conv4_box = copy.deepcopy(self.conv4)
        self.conv5_box = copy.deepcopy(self.conv5)
        self.classifier_box = Classifier(2048, self.num_classes, bias=True)
        self.conv4_box_2 = copy.deepcopy(self.conv4)
        self.conv5_box_2 = copy.deepcopy(self.conv5)
        self.classifier_box_2 = Classifier(2048, self.num_classes, bias=True)
        self.conv6_1 = nn.Conv2d(1024, 10 * self.num_classes, 1, 1, 1)
        self.conv6_2 = nn.Conv2d(1024, 10 * self.num_classes, 1, 1, 1)
        self.conv6 = nn.Conv2d(1024, 10 * self.num_classes, 1, 1, 1)
        self.cls_part_1 = Classifier(10 * self.num_classes, self.num_classes, bias=True)
        self.cls_part_2 = Classifier(10 * self.num_classes, self.num_classes, bias=True)
        self.cls_part = Classifier(10 * self.num_classes, self.num_classes, bias=True)
        self.cls_cat_1 = Classifier(2048 + 10 * self.num_classes, self.num_classes, bias=True)
        self.cls_cat_2 = Classifier(2048 + 10 * self.num_classes, self.num_classes, bias=True)
        self.cls_cat = Classifier(2048 + 10 * self.num_classes, self.num_classes, bias=True)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.cls_cat_a = Classifier(3 * (2048 + 10 * self.num_classes), self.num_classes, bias=True)
        self.conv4_gate = copy.deepcopy(self.conv4)
        self.conv5_gate = copy.deepcopy(self.conv5)
        self.cls_gate = nn.Sequential(Classifier(2048, 512, bias=True), Classifier(512, 3, bias=True))

    def forward(self, x, y=None, is_vis=False, vis_idx=None, gt_top=None):
        b = x.size(0)
        conv4 = self.conv4(x)
        conv5 = self.conv5(conv4)
        conv5_pool = self.pool(conv5).view(b, -1)
        logits = self.classifier(conv5_pool)
        pool_conv6 = self.pool_max(F.relu(self.conv6(conv4.detach()))).view(b, -1)
        pool_cat = torch.cat((10 * l2_norm_v2(conv5_pool.detach()), 10 * l2_norm_v2(pool_conv6.detach())), dim=1)
        logits_max = self.cls_part(pool_conv6)
        logits_cat = self.cls_cat(pool_cat)
        with torch.enable_grad():
            layer_weights = None
            self.grad_cam = GradCam(model=self, feature_extractor=self.conv5, classifier=self.classifier, target_layer_names=['2'])
            target_index = y
            if is_vis and not vis_idx is None:
                if vis_idx == 1:
                    target_index = gt_top
            layer_weights = self.grad_cam(conv4.detach(), target_index)
            self.grad_cam = None
        input_box, box_xy = get_bbox(x, conv5, layer_weights, rate=self.box_thred, img_size=self.image_size)
        conv4_box = self.conv4_box(input_box.detach())
        conv5_box = self.conv5_box(conv4_box)
        conv5_box_pool = self.pool(conv5_box).view(b, -1)
        logits_box = self.classifier_box(conv5_box_pool)
        pool_conv6_1 = self.pool_max(F.relu(self.conv6_1(conv4_box.detach()))).view(b, -1)
        pool_cat_1 = torch.cat((10 * l2_norm_v2(conv5_box_pool.detach()), 10 * l2_norm_v2(pool_conv6_1.detach())), dim=1)
        logits_max_1 = self.cls_part_1(pool_conv6_1)
        logits_cat_1 = self.cls_cat_1(pool_cat_1)
        with torch.enable_grad():
            layer_weights_2 = None
            self.grad_cam = GradCam(model=self, feature_extractor=self.conv5_box, classifier=self.classifier_box, target_layer_names=['2'])
            target_index = y
            if is_vis and not vis_idx is None:
                if vis_idx == 2:
                    target_index = gt_top
            layer_weights_2 = self.grad_cam(conv4_box.detach(), target_index)
            self.grad_cam = None
        input_box_2, box_xy_2 = get_bbox(input_box, conv5_box, layer_weights_2, rate=self.box_thred, img_size=self.image_size)
        conv4_box_2 = self.conv4_box_2(input_box_2.detach())
        conv5_box_2 = self.conv5_box_2(conv4_box_2)
        conv5_box_pool_2 = self.pool(conv5_box_2).view(b, -1)
        logits_box_2 = self.classifier_box_2(conv5_box_pool_2)
        pool_conv6_2 = self.pool_max(F.relu(self.conv6_2(conv4_box_2.detach()))).view(b, -1)
        pool_cat_2 = torch.cat((10 * l2_norm_v2(conv5_box_pool_2.detach()), 10 * l2_norm_v2(pool_conv6_2.detach())), dim=1)
        logits_max_2 = self.cls_part_2(pool_conv6_2)
        logits_cat_2 = self.cls_cat_2(pool_cat_2)
        conv5_gate = self.conv5_gate(self.conv4_gate(x))
        pool_gate = self.pool(conv5_gate).view(b, -1)
        pr_gate = F.softmax(self.cls_gate(pool_gate), dim=1)
        logits_gate = torch.stack([logits_cat.detach(), logits_cat_1.detach(), logits_cat_2.detach()], dim=-1)
        logits_gate = logits_gate * pr_gate.view(pr_gate.size(0), 1, pr_gate.size(1))
        logits_gate = logits_gate.sum(-1)
        logits_list = [logits, logits_max, logits_cat, logits_box, logits_max_1, logits_cat_1, logits_box_2, logits_max_2, logits_cat_2, logits_gate]
        outputs = {'logits': logits_list, 'pr_gate': pr_gate}
        return outputs

    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters()) + list(self.conv4.parameters()) + list(self.conv5_box.parameters()) + list(self.conv4_box.parameters()) + list(self.conv5_box_2.parameters()) + list(self.conv4_box_2.parameters()) + list(self.conv5_gate.parameters()) + list(self.conv4_gate.parameters())
        extractor_params_ids = list(map(id, extractor_params))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params


class Covpool(Function):

    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        I_hat = -1.0 / M / M * torch.ones(M, M, device=x.device) + 1.0 / M * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        ctx.save_for_backward(input, I_hat)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        grad_input = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize, dim, h, w)
        return grad_input


class Sqrtm(Function):

    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = 1.0 / 3.0 * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False, device=x.device).type(dtype)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, iterN, 1, 1).type(dtype)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            YZY = 0.5 * Y[:, iterN - 2, :, :].bmm(I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]))
        y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, YZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :])) - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(Y[:, iterN - 2, :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) - Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) - ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) - Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) - dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i] - grad_aux[i] / (normA[i] * normA[i])) * torch.ones(dim, device=x.device).diag().type(dtype)
        return grad_input, None


class Triuvec(Function):

    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        x = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero()
        y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(dtype)
        y = x[:, index]
        ctx.save_for_backward(input, index)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        grad_input = torch.zeros(batchSize, dim * dim, device=x.device, requires_grad=False).type(dtype)
        grad_input[:, index] = grad_output
        grad_input = grad_input.reshape(batchSize, dim, dim)
        return grad_input


class MPNCOV(nn.Module):
    """Matrix power normalized Covariance pooling (MPNCOV)
       implementation of fast MPN-COV (i.e.,iSQRT-COV)
       https://arxiv.org/abs/1712.01034

    Args:
        iterNum: #iteration of Newton-schulz method
        is_sqrt: whether perform matrix square root or not
        is_vec: whether the output is a vector or not
        input_dim: the #channel of input feature
        dimension_reduction: if None, it will not use 1x1 conv to
                              reduce the #channel of feature.
                             if 256 or others, the #channel of feature
                              will be reduced to 256 or others.
    """

    def __init__(self, iter_num=3, is_sqrt=True, is_vec=True, input_dim=2048, dimension_reduction=None):
        super(MPNCOV, self).__init__()
        self.iterNum = iter_num
        self.is_sqrt = is_sqrt
        self.is_vec = is_vec
        self.dr = dimension_reduction
        if self.dr is not None:
            self.conv_dr_block = nn.Sequential(nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.dr), nn.ReLU(inplace=True))
        output_dim = self.dr if self.dr else input_dim
        if self.is_vec:
            self.output_dim = int(output_dim * (output_dim + 1) / 2)
        else:
            self.output_dim = int(output_dim * output_dim)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _cov_pool(self, x):
        return Covpool.apply(x)

    def _sqrtm(self, x):
        return Sqrtm.apply(x, self.iterNum)

    def _triuvec(self, x):
        return Triuvec.apply(x)

    def forward(self, x):
        if self.dr is not None:
            x = self.conv_dr_block(x)
        x = self._cov_pool(x)
        if self.is_sqrt:
            x = self._sqrtm(x)
        if self.is_vec:
            x = self._triuvec(x)
        return x


class MPN(nn.Module):

    def __init__(self, config):
        super(MPN, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pool = MPNCOV(config.iter_num, config.is_sqrt, config.is_vec, config.input_dim, config.dimension_reduction)
        self.classifier = nn.Linear(self.pool.output_dim, config.num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ProposalNet(nn.Module):

    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


_default_anchors_setting = dict(layer='p3', stride=32, size=48, scale=[2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], aspect_ratio=[0.667, 1, 1.5]), dict(layer='p4', stride=64, size=96, scale=[2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], aspect_ratio=[0.667, 1, 1.5]), dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], aspect_ratio=[0.667, 1, 1.5])


def generate_default_anchor_maps(input_shape=(448, 448), anchors_setting=None):
    """
    generate default anchor

    :param anchors_setting: all informations of anchors
    :param input_shape: shape of input images, e.g. (h, w)
    :return: center_anchors: # anchors * 4 (oy, ox, h, w)
             edge_anchors: # anchors * 4 (y0, x0, y1, x1)
             anchor_area: # anchors * 1 (area)
    """
    if anchors_setting is None:
        anchors_setting = _default_anchors_setting
    center_anchors = np.zeros((0, 4), dtype=np.float32)
    edge_anchors = np.zeros((0, 4), dtype=np.float32)
    anchor_areas = np.zeros((0,), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)
    for anchor_info in anchors_setting:
        stride = anchor_info['stride']
        size = anchor_info['size']
        scales = anchor_info['scale']
        aspect_ratios = anchor_info['aspect_ratio']
        output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
        output_map_shape = output_map_shape.astype(np.int)
        output_shape = tuple(output_map_shape) + (4,)
        ostart = stride / 2.0
        oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
        oy = oy.reshape(output_shape[0], 1)
        ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
        ox = ox.reshape(1, output_shape[1])
        center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
        center_anchor_map_template[:, :, 0] = oy
        center_anchor_map_template[:, :, 1] = ox
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                center_anchor_map = center_anchor_map_template.copy()
                center_anchor_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
                center_anchor_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5
                edge_anchor_map = np.concatenate((center_anchor_map[..., :2] - center_anchor_map[..., 2:4] / 2.0, center_anchor_map[..., :2] + center_anchor_map[..., 2:4] / 2.0), axis=-1)
                anchor_area_map = center_anchor_map[..., 2] * center_anchor_map[..., 3]
                center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))
    return center_anchors, edge_anchors, anchor_areas


def hard_nms(cdds, topn=10, iou_thresh=0.25):
    if not (type(cdds).__module__ == 'numpy' and len(cdds.shape) == 2 and cdds.shape[1] >= 5):
        raise TypeError('edge_box_map should be N * 5+ ndarray')
    cdds = cdds.copy()
    indices = np.argsort(cdds[:, 0])
    cdds = cdds[indices]
    cdd_results = []
    res = cdds
    while res.any():
        cdd = res[-1]
        cdd_results.append(cdd)
        if len(cdd_results) == topn:
            return np.array(cdd_results)
        res = res[:-1]
        start_max = np.maximum(res[:, 1:3], cdd[1:3])
        end_min = np.minimum(res[:, 3:5], cdd[3:5])
        lengths = end_min - start_max
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1]) * (res[:, 4] - res[:, 2]) + (cdd[3] - cdd[1]) * (cdd[4] - cdd[2]) - intersec_map)
        res = res[iou_map_cur < iou_thresh]
    return np.array(cdd_results)


class NTSNet(nn.Module):

    def __init__(self, config):
        super(NTSNet, self).__init__()
        self.topN = config.proposal_num
        self.proposal_num = config.proposal_num
        self.CAT_NUM = config.cat_num
        self.image_size = config.image_size
        self.pretrained_model = resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)
        self.proposal_net = ProposalNet()
        self.concat_net = nn.Linear(2048 * (self.CAT_NUM + 1), 200)
        self.partcls_net = nn.Linear(512 * 4, 200)
        _, edge_anchors, _ = generate_default_anchor_maps(input_shape=(self.image_size, self.image_size))
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1) for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int64)
        top_n_index = torch.from_numpy(top_n_index)
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224])
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear', align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :self.CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]


class OSME_block(torch.nn.Module):

    def __init__(self, channels, ratio):
        torch.nn.Module.__init__(self)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.block = nn.Sequential(nn.Linear(channels, channels // ratio), nn.ReLU(inplace=True), nn.Linear(channels // ratio, channels), nn.Sigmoid())

    def forward(self, x):
        N, C, _, _ = x.size()
        z = self.avg_pool(x).squeeze()
        m = self.block(z)
        s = m.view(N, C, 1, 1) * x
        return s


class OSME(torch.nn.Module):

    def __init__(self, in_channels, out_channels=1024, feature_shape=(7, 7), num_attention=2):
        torch.nn.Module.__init__(self)
        reduce_ratio = 16
        fc_in_channels = in_channels * feature_shape[0] * feature_shape[1] if isinstance(feature_shape, tuple) else in_channels * feature_shape * feature_shape
        self.blocks = nn.ModuleList([OSME_block(in_channels, reduce_ratio) for _ in range(num_attention)])
        self.fcs = nn.ModuleList([nn.Linear(fc_in_channels, out_channels) for _ in range(num_attention)])

    def forward(self, x):
        """
        :param x: [N D W H]
        :return: [N C], [N P C]
        """
        N, C, _, _ = x.size()
        s = [block(x) for block in self.blocks]
        features = [fc(s[i].view(N, -1)) for i, fc in enumerate(self.fcs)]
        return sum(features), torch.stack(features, dim=1)


class OSMENet(nn.Module):

    def __init__(self, config):
        super(OSMENet, self).__init__()
        self.config = config
        self.num_attention = config.num_attention
        self.num_classes = config.num_classes
        resnet = resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.osme = OSME(2048, 1024, feature_shape=7, num_attention=self.num_attention)
        self.classifier = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x1, x_part = self.osme(x)
        out = self.classifier(x1)
        return out, x_part


class PeerLearningNet(nn.Module):

    def __init__(self, config):
        super(PeerLearningNet, self).__init__()
        self.base_model = MODEL.get(config.base_model.name)(config.base_model)
        self.base_model2 = copy.deepcopy(self.base_model)
        self.base_model2.classifier.apply(initialize_weights)

    def forward(self, x):
        out1 = self.base_model(x)
        out2 = self.base_model2(x)
        return out1, out2


class Node(nn.Module):

    def __init__(self, index: int):
        super().__init__()
        self._index = index

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def index(self) ->int:
        return self._index

    @property
    def size(self) ->int:
        raise NotImplementedError

    @property
    def nodes(self) ->set:
        return self.branches.union(self.leaves)

    @property
    def leaves(self) ->set:
        raise NotImplementedError

    @property
    def branches(self) ->set:
        raise NotImplementedError

    @property
    def nodes_by_index(self) ->dict:
        raise NotImplementedError

    @property
    def num_branches(self) ->int:
        return len(self.branches)

    @property
    def num_leaves(self) ->int:
        return len(self.leaves)

    @property
    def depth(self) ->int:
        raise NotImplementedError


class Branch(Node):

    def __init__(self, index: int, l: Node, r: Node, args):
        super().__init__(index)
        self.l = l
        self.r = r
        self._log_probabilities = args.log_probabilities

    def forward(self, xs: torch.Tensor, **kwargs):
        batch_size = xs.size(0)
        node_attr = kwargs.setdefault('attr', dict())
        if not self._log_probabilities:
            pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
        else:
            pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
        ps = self.g(xs, **kwargs)
        if not self._log_probabilities:
            node_attr[self, 'ps'] = ps
            node_attr[self.l, 'pa'] = (1 - ps) * pa
            node_attr[self.r, 'pa'] = ps * pa
            l_dists, _ = self.l.forward(xs, **kwargs)
            r_dists, _ = self.r.forward(xs, **kwargs)
            ps = ps.view(batch_size, 1)
            return (1 - ps) * l_dists + ps * r_dists, node_attr
        else:
            node_attr[self, 'ps'] = ps
            x = torch.abs(ps) + 1e-07
            oneminusp = torch.where(x < np.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))
            node_attr[self.l, 'pa'] = oneminusp + pa
            node_attr[self.r, 'pa'] = ps + pa
            l_dists, _ = self.l.forward(xs, **kwargs)
            r_dists, _ = self.r.forward(xs, **kwargs)
            ps = ps.view(batch_size, 1)
            oneminusp = oneminusp.view(batch_size, 1)
            logs_stacked = torch.stack((oneminusp + l_dists, ps + r_dists))
            return torch.logsumexp(logs_stacked, dim=0), node_attr

    def g(self, xs: torch.Tensor, **kwargs):
        out_map = kwargs['out_map']
        conv_net_output = kwargs['conv_net_output']
        out = conv_net_output[out_map[self]]
        return out.squeeze(dim=1)

    @property
    def size(self) ->int:
        return 1 + self.l.size + self.r.size

    @property
    def leaves(self) ->set:
        return self.l.leaves.union(self.r.leaves)

    @property
    def branches(self) ->set:
        return {self}.union(self.l.branches).union(self.r.branches)

    @property
    def nodes_by_index(self) ->dict:
        return {self.index: self, **self.l.nodes_by_index, **self.r.nodes_by_index}

    @property
    def num_branches(self) ->int:
        return 1 + self.l.num_branches + self.r.num_branches

    @property
    def num_leaves(self) ->int:
        return self.l.num_leaves + self.r.num_leaves

    @property
    def depth(self) ->int:
        return self.l.depth + 1


class L2Conv2D(nn.Module):
    """
    Convolutional layer that computes the squared L2 distance instead of the conventional inner product. 
    """

    def __init__(self, num_prototypes, num_features, w_1, h_1):
        """
        Create a new L2Conv2D layer
        :param num_prototypes: The number of prototypes in the layer
        :param num_features: The number of channels in the input features
        :param w_1: Width of the prototypes
        :param h_1: Height of the prototypes
        """
        super().__init__()
        prototype_shape = num_prototypes, num_features, w_1, h_1
        self.prototype_vectors = nn.Parameter(torch.randn(prototype_shape), requires_grad=True)

    def forward(self, xs):
        """
        Perform convolution over the input using the squared L2 distance for all prototypes in the layer
        :param xs: A batch of input images obtained as output from some convolutional neural network F. Following the
                   notation from the paper, let the shape of xs be (batch_size, D, W, H), where
                     - D is the number of output channels of the conv net F
                     - W is the width of the convolutional output of F
                     - H is the height of the convolutional output of F
        :return: a tensor of shape (batch_size, num_prototypes, W, H) obtained from computing the squared L2 distances
                 for patches of the input using all prototypes
        """
        ones = torch.ones_like(self.prototype_vectors, device=xs.device)
        xs_squared_l2 = F.conv2d(xs ** 2, weight=ones)
        ps_squared_l2 = torch.sum(self.prototype_vectors ** 2, dim=(1, 2, 3))
        ps_squared_l2 = ps_squared_l2.view(-1, 1, 1)
        xs_conv = F.conv2d(xs, weight=self.prototype_vectors)
        distance = xs_squared_l2 + ps_squared_l2 - 2 * xs_conv
        distance = torch.sqrt(torch.abs(distance) + 1e-14)
        if torch.isnan(distance).any():
            raise Exception('Error: NaN values! Using the --log_probabilities flag might fix this issue')
        return distance


class Leaf(Node):

    def __init__(self, index: int, num_classes: int, args):
        super().__init__(index)
        if args.disable_derivative_free_leaf_optim:
            self._dist_params = nn.Parameter(torch.randn(num_classes), requires_grad=True)
        elif args.kontschieder_normalization:
            self._dist_params = nn.Parameter(torch.ones(num_classes), requires_grad=False)
        else:
            self._dist_params = nn.Parameter(torch.zeros(num_classes), requires_grad=False)
        self._log_probabilities = args.log_probabilities
        self._kontschieder_normalization = args.kontschieder_normalization

    def forward(self, xs: torch.Tensor, **kwargs):
        batch_size = xs.size(0)
        node_attr = kwargs.setdefault('attr', dict())
        if not self._log_probabilities:
            node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
        else:
            node_attr.setdefault((self, 'pa'), torch.zeros(batch_size, device=xs.device))
        dist = self.distribution()
        dist = dist.view(1, -1)
        dists = torch.cat((dist,) * batch_size, dim=0)
        node_attr[self, 'ds'] = dists
        return dists, node_attr

    def distribution(self) ->torch.Tensor:
        if not self._kontschieder_normalization:
            if self._log_probabilities:
                return F.log_softmax(self._dist_params, dim=0)
            else:
                return F.softmax(self._dist_params - torch.max(self._dist_params), dim=0)
        elif self._log_probabilities:
            return torch.log(self._dist_params / torch.sum(self._dist_params) + 1e-10)
        else:
            return self._dist_params / torch.sum(self._dist_params)

    @property
    def requires_grad(self) ->bool:
        return self._dist_params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._dist_params.requires_grad = val

    @property
    def size(self) ->int:
        return 1

    @property
    def leaves(self) ->set:
        return {self}

    @property
    def branches(self) ->set:
        return set()

    @property
    def nodes_by_index(self) ->dict:
        return {self.index: self}

    @property
    def num_branches(self) ->int:
        return 0

    @property
    def num_leaves(self) ->int:
        return 1

    @property
    def depth(self) ->int:
        return 0


def min_pool2d(xs, **kwargs):
    return -F.max_pool2d(-xs, **kwargs)


class ProtoTree(nn.Module):
    ARGUMENTS = ['depth', 'num_features', 'W1', 'H1', 'log_probabilities']
    SAMPLING_STRATEGIES = ['distributed', 'sample_max', 'greedy']

    def __init__(self, args):
        super().__init__()
        args = self.init_args(args)
        assert args.height > 0
        assert args.num_classes > 0
        self._num_classes = args.num_classes
        self._root = self._init_tree(args.num_classes, args)
        self.num_features = args.num_features
        self.num_prototypes = self.num_branches
        self.prototype_shape = args.W1, args.H1, args.num_features
        self._parents = dict()
        self._set_parents()
        self._log_probabilities = args.log_probabilities
        self._kontschieder_normalization = args.kontschieder_normalization
        self._kontschieder_train = args.kontschieder_train
        self._out_map = {n: i for i, n in zip(range(2 ** args.height - 1), self.branches)}
        self.prototype_layer = L2Conv2D(self.num_prototypes, self.num_features, args.W1, args.H1)

    def init_args(self, args):
        args.defrost()
        if 'log_probabilities' not in args:
            args.log_probabilities = False
        if 'kontschieder_normalization' not in args:
            args.kontschieder_normalization = False
        if 'kontschieder_train' not in args:
            args.kontschieder_train = False
        if 'disable_derivative_free_leaf_optim' not in args:
            args.disable_derivative_free_leaf_optim = False
        args.freeze()
        return args

    @property
    def root(self) ->Node:
        return self._root

    @property
    def leaves_require_grad(self) ->bool:
        return any([leaf.requires_grad for leaf in self.leaves])

    @leaves_require_grad.setter
    def leaves_require_grad(self, val: bool):
        for leaf in self.leaves:
            leaf.requires_grad = val

    @property
    def prototypes_require_grad(self) ->bool:
        return self.prototype_layer.prototype_vectors.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.prototype_vectors.requires_grad = val

    def forward(self, xs: torch.Tensor, features: torch.Tensor, sampling_strategy: str=SAMPLING_STRATEGIES[0], **kwargs) ->tuple:
        assert sampling_strategy in ProtoTree.SAMPLING_STRATEGIES
        bs, D, W, H = features.shape
        """
            COMPUTE THE PROTOTYPE SIMILARITIES GIVEN THE COMPUTED FEATURES
        """
        distances = self.prototype_layer(features)
        min_distances = min_pool2d(distances, kernel_size=(W, H))
        min_distances = min_distances.view(bs, self.num_prototypes)
        if not self._log_probabilities:
            similarities = torch.exp(-min_distances)
        else:
            similarities = -min_distances
        kwargs['conv_net_output'] = similarities.chunk(similarities.size(1), dim=1)
        kwargs['out_map'] = dict(self._out_map)
        """
            PERFORM A FORWARD PASS THROUGH THE TREE GIVEN THE COMPUTED SIMILARITIES
        """
        out, attr = self._root.forward(xs, **kwargs)
        info = dict()
        info['pa_tensor'] = {n.index: attr[n, 'pa'].unsqueeze(1) for n in self.nodes}
        info['ps'] = {n.index: attr[n, 'ps'].unsqueeze(1) for n in self.branches}
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[0]:
            return out, info
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[1]:
            batch_size = xs.size(0)
            leaves = list(self.leaves)
            pas = [attr[l, 'pa'].view(batch_size, 1) for l in leaves]
            dss = [attr[l, 'ds'].view(batch_size, 1, self._num_classes) for l in leaves]
            pas = torch.cat(tuple(pas), dim=1)
            dss = torch.cat(tuple(dss), dim=1)
            ix = torch.argmax(pas, dim=1).long()
            dists = []
            for j, i in zip(range(dss.shape[0]), ix):
                dists += [dss[j][i].view(1, -1)]
            dists = torch.cat(tuple(dists), dim=0)
            info['out_leaf_ix'] = [leaves[i.item()].index for i in ix]
            return dists, info
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[2]:
            batch_size = xs.size(0)
            threshold = 0.5 if not self._log_probabilities else np.log(0.5)
            routing = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                node = self._root
                while node in self.branches:
                    routing[i] += [node]
                    if attr[node, 'ps'][i].item() > threshold:
                        node = node.r
                    else:
                        node = node.l
                routing[i] += [node]
            dists = [attr[path[-1], 'ds'][0] for path in routing]
            dists = torch.cat([dist.unsqueeze(0) for dist in dists], dim=0)
            info['out_leaf_ix'] = [path[-1].index for path in routing]
            return dists, info
        raise Exception('Sampling strategy not recognized!')

    @property
    def depth(self) ->int:
        d = lambda node: 1 if isinstance(node, Leaf) else 1 + max(d(node.l), d(node.r))
        return d(self._root)

    @property
    def size(self) ->int:
        return self._root.size

    @property
    def nodes(self) ->set:
        return self._root.nodes

    @property
    def nodes_by_index(self) ->dict:
        return self._root.nodes_by_index

    @property
    def node_depths(self) ->dict:

        def _assign_depths(node, d):
            if isinstance(node, Leaf):
                return {node: d}
            if isinstance(node, Branch):
                return {node: d, **_assign_depths(node.r, d + 1), **_assign_depths(node.l, d + 1)}
        return _assign_depths(self._root, 0)

    @property
    def branches(self) ->set:
        return self._root.branches

    @property
    def leaves(self) ->set:
        return self._root.leaves

    @property
    def num_branches(self) ->int:
        return self._root.num_branches

    @property
    def num_leaves(self) ->int:
        return self._root.num_leaves

    def save(self, directory_path: str):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        with open(directory_path + '/model.pth', 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        with open(directory_path + '/model_state.pth', 'wb') as f:
            torch.save(self.state_dict(), f)
        with open(directory_path + '/tree.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + '/model.pth')

    def _init_tree(self, num_classes, args) ->Node:

        def _init_tree_recursive(i: int, d: int) ->Node:
            if d == args.height:
                return Leaf(i, num_classes, args)
            else:
                left = _init_tree_recursive(i + 1, d + 1)
                return Branch(i, left, _init_tree_recursive(i + left.size + 1, d + 1), args)
        return _init_tree_recursive(0, 0)

    def _set_parents(self) ->None:
        self._parents.clear()
        self._parents[self._root] = None

        def _set_parents_recursively(node: Node):
            if isinstance(node, Branch):
                self._parents[node.r] = node
                self._parents[node.l] = node
                _set_parents_recursively(node.r)
                _set_parents_recursively(node.l)
                return
            if isinstance(node, Leaf):
                return
            raise Exception('Unrecognized node type!')
        _set_parents_recursively(self._root)

    def path_to(self, node: Node):
        assert node in self.leaves or node in self.branches
        path = [node]
        while isinstance(self._parents[node], Node):
            node = self._parents[node]
            path = [node] + path
        return path


def initialize_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))


class ProtoTreeNet(torch.nn.Module):

    def __init__(self, config):
        super(ProtoTreeNet, self).__init__()
        self.config = config
        resnet = resnet50(pretrained=True)
        state_dict = self.get_inat_resnet50_weight(config.backbone.pretrain)
        resnet.load_state_dict(state_dict, strict=False)
        neck_conv_in_channels = [i for i in resnet.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.neck_conv = nn.Sequential(nn.Conv2d(in_channels=neck_conv_in_channels, out_channels=config.num_features, kernel_size=1, bias=False), nn.Sigmoid())
        self.tree = ProtoTree(args=config)
        with torch.no_grad():
            torch.nn.init.normal_(self.tree.prototype_layer.prototype_vectors, mean=0.5, std=0.1)
            self.neck_conv.apply(initialize_weights_xavier)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck_conv(features)
        pred, info = self.tree(x, features)
        return pred, info

    def get_inat_resnet50_weight(self, pretrain):
        model_dict = torch.load(pretrain)
        new_model = copy.deepcopy(model_dict)
        for k in model_dict.keys():
            if k.startswith('module.backbone.cb_block'):
                splitted = k.split('cb_block')
                new_model['layer4.2' + splitted[-1]] = model_dict[k]
                del new_model[k]
            elif k.startswith('module.backbone.rb_block'):
                del new_model[k]
            elif k.startswith('module.backbone.'):
                splitted = k.split('backbone.')
                new_model[splitted[-1]] = model_dict[k]
                del new_model[k]
            elif k.startswith('module.classifier'):
                del new_model[k]
        return new_model


class KernelGenerator(nn.Module):

    def __init__(self, size, offset=None):
        super(KernelGenerator, self).__init__()
        self.size = self._pair(size)
        xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
        if offset is None:
            offset_x = offset_y = size // 2
        else:
            offset_x, offset_y = self._pair(offset)
        self.factor = torch.from_numpy(-(np.power(xx - offset_x, 2) + np.power(yy - offset_y, 2)) / 2).float()

    @staticmethod
    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    def forward(self, theta):
        pow2 = torch.pow(theta * self.size[0], 2)
        kernel = 1.0 / (2 * np.pi * pow2) * torch.exp(self.factor / pow2)
        return kernel / kernel.max()


class ScaleLayer(nn.Module):

    def __init__(self, init_value=0.001):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def _mean_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)


def kernel_generate(theta, size, offset=None):
    return KernelGenerator(size, offset)(theta)


def makeGaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4
        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset:-offset, offset:-offset]
        element_map = element_map
        _, indices = F.max_pool2d(padded_maps, kernel_size=win_size, stride=1, return_indices=True, ceil_mode=True)
        peak_map = indices == element_map
        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = peak_map & mask
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / peak_map.view(batch_size, num_channels, -1).sum(2)
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1) / (peak_map.view(batch_size, num_channels, -1).sum(2).view(batch_size, num_channels, 1, 1) + 1e-06)
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)


class S3N(nn.Module):

    def __init__(self, config):
        super(S3N, self).__init__()
        self.config = config
        num_classes = self.config.num_classes
        self.backbone = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.num_features = 2048
        self.grid_size = 31
        self.padding_size = 30
        self.global_size = self.grid_size + 2 * self.padding_size
        self.input_size_net = self.config.image_size
        gaussian_weights = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1, fwhm=13))
        self.base_ratio = self.config.base_ratio
        self.radius = ScaleLayer(self.config.radius)
        self.radius_inv = ScaleLayer(self.config.radius_inv)
        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        self.filter.weight[0].data[:, :, :] = gaussian_weights
        self.P_basis = torch.zeros(2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k, i, j] = k * (i - self.padding_size) / (self.grid_size - 1.0) + (1.0 - k) * (j - self.padding_size) / (self.grid_size - 1.0)
        self.raw_classifier = nn.Linear(2048, num_classes)
        self.sampler_buffer = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(2048), nn.ReLU())
        self.sampler_classifier = nn.Linear(2048, num_classes)
        self.sampler_buffer1 = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(2048), nn.ReLU())
        self.sampler_classifier1 = nn.Linear(2048, num_classes)
        self.con_classifier = nn.Linear(int(self.num_features * 3), num_classes)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.map_origin = nn.Conv2d(2048, num_classes, 1, 1, 0)

    def create_grid(self, x):
        P = torch.autograd.Variable(torch.zeros(1, 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size), requires_grad=False)
        P[0, :, :, :] = self.P_basis
        P = P.expand(x.size(0), 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size, self.global_size)
        all_filter = self.filter(x_mul).view(-1, 2, self.grid_size, self.grid_size)
        x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        x_filter = x_filter / p_filter
        y_filter = y_filter / p_filter
        xgrids = x_filter * 2 - 1
        ygrids = y_filter * 2 - 1
        xgrids = torch.clamp(xgrids, min=-1, max=1)
        ygrids = torch.clamp(ygrids, min=-1, max=1)
        xgrids = xgrids.view(-1, 1, self.grid_size, self.grid_size)
        ygrids = ygrids.view(-1, 1, self.grid_size, self.grid_size)
        grid = torch.cat((xgrids, ygrids), 1)
        grid = F.interpolate(grid, size=(self.input_size_net, self.input_size_net), mode='bilinear', align_corners=True)
        grid = torch.transpose(grid, 1, 2)
        grid = torch.transpose(grid, 2, 3)
        return grid

    def generate_map(self, input_x, class_response_maps, p):
        N, C, H, W = class_response_maps.size()
        device = input_x.device
        score_pred, sort_number = torch.sort(F.softmax(F.adaptive_avg_pool2d(class_response_maps, 1), dim=1), dim=1, descending=True)
        gate_score = (score_pred[:, 0:5] * torch.log(score_pred[:, 0:5])).sum(1)
        xs = []
        xs_inv = []
        for idx_i in range(N):
            if gate_score[idx_i] > -0.2:
                decide_map = class_response_maps[idx_i, sort_number[idx_i, 0], :, :]
            else:
                decide_map = class_response_maps[idx_i, sort_number[idx_i, 0:5], :, :].mean(0)
            min_value, max_value = decide_map.min(), decide_map.max()
            decide_map = (decide_map - min_value) / (max_value - min_value)
            peak_list, aggregation = peak_stimulation(decide_map, win_size=3, peak_filter=_mean_filter)
            decide_map = decide_map.squeeze(0).squeeze(0)
            score = [decide_map[item[2], item[3]] for item in peak_list]
            x = [item[3] for item in peak_list]
            y = [item[2] for item in peak_list]
            if score == []:
                temp = torch.zeros(1, 1, self.grid_size, self.grid_size)
                temp += self.base_ratio
                xs.append(temp)
                continue
            peak_num = torch.arange(len(score))
            temp = self.base_ratio
            temp_w = self.base_ratio
            if p == 0:
                for i in peak_num:
                    temp += score[i] * kernel_generate(self.radius(torch.sqrt(score[i])), H, (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0)
                    temp_w += 1 / score[i] * kernel_generate(self.radius_inv(torch.sqrt(score[i])), H, (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0)
            elif p == 1:
                for i in peak_num:
                    rd = random.uniform(0, 1)
                    if score[i] > rd:
                        temp += score[i] * kernel_generate(self.radius(torch.sqrt(score[i])), H, (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0)
                    else:
                        temp_w += 1 / score[i] * kernel_generate(self.radius_inv(torch.sqrt(score[i])), H, (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0)
            elif p == 2:
                index = score.index(max(score))
                temp += score[index] * kernel_generate(self.radius(torch.sqrt(score[index])), H, (x[index].item(), y[index].item())).unsqueeze(0).unsqueeze(0)
                index = score.index(min(score))
                temp_w += 1 / score[index] * kernel_generate(self.radius_inv(torch.sqrt(score[index])), H, (x[index].item(), y[index].item())).unsqueeze(0).unsqueeze(0)
            if type(temp) == float:
                temp += torch.zeros(1, 1, self.grid_size, self.grid_size)
            xs.append(temp)
            if type(temp_w) == float:
                temp_w += torch.zeros(1, 1, self.grid_size, self.grid_size)
            xs_inv.append(temp_w)
        xs = torch.cat(xs, 0)
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)
        grid = self.create_grid(xs_hm)
        x_sampled_zoom = F.grid_sample(input_x, grid, align_corners=True)
        xs_inv = torch.cat(xs_inv, 0)
        xs_hm_inv = nn.ReplicationPad2d(self.padding_size)(xs_inv)
        grid_inv = self.create_grid(xs_hm_inv)
        x_sampled_inv = F.grid_sample(input_x, grid_inv, align_corners=True)
        return x_sampled_zoom, x_sampled_inv

    def forward(self, input_x, p):
        self.map_origin.weight.data.copy_(self.raw_classifier.weight.data.unsqueeze(-1).unsqueeze(-1))
        self.map_origin.bias.data.copy_(self.raw_classifier.bias.data)
        feature_raw = self.features(input_x)
        agg_origin = self.raw_classifier(self.avg(feature_raw).view(-1, 2048))
        with torch.no_grad():
            class_response_maps = F.interpolate(self.map_origin(feature_raw), size=self.grid_size, mode='bilinear', align_corners=True)
        x_sampled_zoom, x_sampled_inv = self.generate_map(input_x, class_response_maps, p)
        feature_D = self.sampler_buffer(self.features(x_sampled_zoom))
        agg_sampler = self.sampler_classifier(self.avg(feature_D).view(-1, 2048))
        feature_C = self.sampler_buffer1(self.features(x_sampled_inv))
        agg_sampler1 = self.sampler_classifier1(self.avg(feature_C).view(-1, 2048))
        aggregation = self.con_classifier(torch.cat([self.avg(feature_raw).view(-1, 2048), self.avg(feature_D).view(-1, 2048), self.avg(feature_C).view(-1, 2048)], 1))
        return aggregation, agg_origin, agg_sampler, agg_sampler1


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BilinearPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CIN,
     lambda: ([], {'config': _mock_config(num_classes=4)}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (CINLoss,
     lambda: ([], {'config': _mock_config(alpha=4, beta=4, channel=4, feature_size=4, r_channel=4)}),
     lambda: ([], {'output': torch.rand([4, 4]), 'target': torch.rand([4, 4])}),
     False),
    (ChannelInteractionModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {}),
     False),
    (Classifier,
     lambda: ([], {'in_panel': 4, 'out_panel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CompactBilinearPooling,
     lambda: ([], {'input_dim1': 4, 'input_dim2': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DCL,
     lambda: ([], {'config': _mock_config(num_classes=4, cls_2=4, cls_2xmul=4)}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GroupingUnit,
     lambda: ([], {'in_channels': 4, 'num_parts': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KernelGenerator,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Conv2D,
     lambda: ([], {'num_prototypes': 4, 'num_features': 4, 'w_1': 4, 'h_1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Leaf,
     lambda: ([], {'index': 4, 'num_classes': 4, 'args': _mock_config(disable_derivative_free_leaf_optim=_mock_config(), log_probabilities=4, kontschieder_normalization=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MPNCOV,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (OSME_block,
     lambda: ([], {'channels': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PairwiseConfusionLoss,
     lambda: ([], {'config': _mock_config(lambda_a=4)}),
     lambda: ([torch.rand([4, 2, 4, 4]), torch.rand([4, 2, 4, 4])], {}),
     True),
    (ProposalNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {}),
     True),
    (PyramidFeatures,
     lambda: ([], {'B2_size': 4, 'B3_size': 4, 'B4_size': 4, 'B5_size': 4}),
     lambda: ([(torch.rand([4, 4, 16, 16]), torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (RegularLoss,
     lambda: ([], {}),
     lambda: ([[torch.rand([4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]], {}),
     True),
    (S3N,
     lambda: ([], {'config': _mock_config(num_classes=4, image_size=4, base_ratio=4, radius=4, radius_inv=4)}),
     lambda: ([torch.rand([4, 3, 64, 64]), 0], {}),
     False),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleFPA,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpatialGate,
     lambda: ([], {'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Hawkeye_FineGrained_Hawkeye(_paritybench_base):
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

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

