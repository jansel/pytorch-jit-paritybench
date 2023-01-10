import sys
_module = sys.modules[__name__]
del sys
conv4 = _module
resnet_large = _module
resnet_small = _module
datamanager = _module
essential_script = _module
deepcluster = _module
deepinfomax = _module
relationnet = _module
rotationnet = _module
simclr = _module
standard = _module
test = _module
train_linear_evaluation = _module
train_unsupervised = _module
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


import collections


import math


import random


import torchvision.datasets as dset


import torchvision.transforms as transforms


import numpy as np


import torchvision


import time


from torch.optim import SGD


from torch.optim import Adam


import torch.nn.functional as F


from torch import nn


import torchvision.transforms.functional


import torch.optim


class Conv4(torch.nn.Module):
    """A simple 4 layers CNN.
    Used as backbone.    
    """

    def __init__(self):
        super(Conv4, self).__init__()
        self.feature_size = 64
        self.name = 'conv4'
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(8), torch.nn.ReLU(), torch.nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(16), torch.nn.ReLU(), torch.nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1))
        self.flatten = torch.nn.Flatten()
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.flatten(h)
        return h


def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
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
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, channels=[16, 32, 64], flatten=True):
        super(ResNet, self).__init__()
        self.name = 'resnet'
        self.flatten = flatten
        self.channels = channels
        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, channels[0], layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if flatten:
            self.feature_size = channels[2] * block.expansion
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
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.flatten:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        return x


class RelationalReasoning(torch.nn.Module):
    """Self-Supervised Relational Reasoning.
  Essential implementation of the method, which uses
  the 'cat' aggregation function (the most effective),
  and can be used with any backbone.
  """

    def __init__(self, backbone, feature_size=64):
        super(RelationalReasoning, self).__init__()
        self.backbone = backbone
        self.relation_head = torch.nn.Sequential(torch.nn.Linear(feature_size * 2, 256), torch.nn.BatchNorm1d(256), torch.nn.LeakyReLU(), torch.nn.Linear(256, 1))

    def aggregate(self, features, K):
        relation_pairs_list = list()
        targets_list = list()
        size = int(features.shape[0] / K)
        shifts_counter = 1
        for index_1 in range(0, size * K, size):
            for index_2 in range(index_1 + size, size * K, size):
                pos_pair = torch.cat([features[index_1:index_1 + size], features[index_2:index_2 + size]], 1)
                neg_pair = torch.cat([features[index_1:index_1 + size], torch.roll(features[index_2:index_2 + size], shifts=shifts_counter, dims=0)], 1)
                relation_pairs_list.append(pos_pair)
                relation_pairs_list.append(neg_pair)
                targets_list.append(torch.ones(size, dtype=torch.float32))
                targets_list.append(torch.zeros(size, dtype=torch.float32))
                shifts_counter += 1
                if shifts_counter >= size:
                    shifts_counter = 1
        relation_pairs = torch.cat(relation_pairs_list, 0)
        targets = torch.cat(targets_list, 0)
        return relation_pairs, targets

    def train(self, tot_epochs, train_loader):
        optimizer = torch.optim.Adam([{'params': self.backbone.parameters()}, {'params': self.relation_head.parameters()}])
        BCE = torch.nn.BCEWithLogitsLoss()
        self.backbone.train()
        self.relation_head.train()
        for epoch in range(tot_epochs):
            for i, (data_augmented, _) in enumerate(train_loader):
                K = len(data_augmented)
                x = torch.cat(data_augmented, 0)
                optimizer.zero_grad()
                features = self.backbone(x)
                relation_pairs, targets = self.aggregate(features, K)
                score = self.relation_head(relation_pairs).squeeze()
                loss = BCE(score, targets)
                loss.backward()
                optimizer.step()
                predicted = torch.round(torch.sigmoid(score))
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = 100.0 * correct / float(len(targets))
                if i % 100 == 0:
                    None


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Model(torch.nn.Module):

    def __init__(self, feature_extractor):
        super(Model, self).__init__()
        self.net = nn.Sequential(collections.OrderedDict([('feature_extractor', feature_extractor)]))
        self.head = nn.Sequential(collections.OrderedDict([('linear1', nn.Linear(feature_extractor.feature_size, 256)), ('bn1', nn.BatchNorm1d(256)), ('relu', nn.LeakyReLU()), ('linear2', nn.Linear(256, 64))]))
        self.optimizer = Adam([{'params': self.net.parameters(), 'lr': 0.001}, {'params': self.head.parameters(), 'lr': 0.001}])

    def return_loss_fn(self, x, t=0.5, eps=1e-08):
        n = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x @ x.t() / (n * n.t()).clamp(min=eps)
        x = torch.exp(x / t)
        idx = torch.arange(x.size()[0])
        idx[::2] += 1
        idx[1::2] -= 1
        x = x[idx]
        x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
        return -torch.log(x.mean())

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.net.train()
        self.head.train()
        loss_meter = AverageMeter()
        statistics_dict = {}
        for i, (data, data_augmented, _) in enumerate(train_loader):
            data = torch.stack(data_augmented, dim=1)
            d = data.size()
            train_x = data.view(d[0] * 2, d[2], d[3], d[4])
            self.optimizer.zero_grad()
            features = self.net(train_x)
            tot_pairs = int(features.shape[0] * features.shape[0])
            embeddings = self.head(features)
            loss = self.return_loss_fn(embeddings)
            loss_meter.update(loss.item(), features.shape[0])
            loss.backward()
            self.optimizer.step()
            if i == 0:
                statistics_dict['batch_size'] = data.shape[0]
                statistics_dict['tot_pairs'] = tot_pairs
        elapsed_time = time.time() - start_time
        None
        return loss_meter.avg, -loss_meter.avg

    def save(self, file_path='./checkpoint.dat'):
        feature_extractor_state_dict = self.net.feature_extractor.state_dict()
        head_state_dict = self.head.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({'backbone': feature_extractor_state_dict, 'head': head_state_dict, 'optimizer': optimizer_state_dict}, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.net.feature_extractor.load_state_dict(checkpoint['backbone'])
        self.head.load_state_dict(checkpoint['head'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class Encoder(nn.Module):
    """The encoder class.

    Takes a feature extractor and returns the representaion
    produced by it (y), and additionally the feature-maps of the
    very first layer (M).
    """

    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.y_size = feature_extractor.feature_size
        if self.feature_extractor.name == 'conv4':
            self.M_channels = 8
        elif self.feature_extractor.name == 'resnet':
            self.M_channels = feature_extractor.channels[0]
        elif self.feature_extractor.name == 'resnet_large':
            self.M_channels = feature_extractor.channels[0]
        else:
            raise ValueError('[ERROR][DeepInfoMax] The network type ' + str(self.feature_extractor.name) + ' is not supported!')
        None

    def forward_resnet_large(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        M = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(M)
        x = self.feature_extractor.layer1(x)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        x = self.feature_extractor.layer4(x)
        x = self.feature_extractor.avgpool(x)
        x = torch.flatten(x, 1)
        return x, M

    def forward_resnet(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        M = self.feature_extractor.relu(x)
        x = self.feature_extractor.layer1(M)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        if self.feature_extractor.flatten:
            x = self.feature_extractor.avgpool(x)
            x = x.view(x.size(0), -1)
        return x, M

    def forward_conv4(self, x):
        x = self.feature_extractor.layer1.conv(x)
        x = self.feature_extractor.layer1.bn(x)
        M = self.feature_extractor.layer1.relu(x)
        x = self.feature_extractor.layer1.avgpool(M)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        x = self.feature_extractor.layer4(x)
        if self.feature_extractor.is_flatten:
            x = self.feature_extractor.flatten(x)
        return x, M

    def forward(self, x):
        if self.feature_extractor.name == 'conv4':
            return self.forward_conv4(x)
        elif self.feature_extractor.name == 'resnet':
            return self.forward_resnet(x)
        elif self.feature_extractor.name == 'resnet_large':
            return self.forward_resnet_large(x)
        else:
            raise ValueError('[ERROR][DeepInfoMax] The network type ' + str(self.feature_extractor.name) + ' is not supported!')


class GlobalDiscriminator(nn.Module):

    def __init__(self, y_size, M_channels):
        super().__init__()
        self.c0 = nn.Conv2d(M_channels, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d(16)
        self.l0 = nn.Linear(32 * 16 * 16 + y_size, 256)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = self.avgpool(h)
        h = h.view(M.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    """The local discriminator class.

    A network that analyses the relation between the
    output of the encoder y, and the feature map M.
    It is called "local" because it compares y with
    each one of the features in M. So if M is a [64, 6, 6]
    feature map, and y is a [32] vector, the comparison is
    done concatenating y along each one of the 6x6 features
    in M: 
    (i) [32] -> [64, 1, 1]; (ii) [32] -> [64, 1, 2]
    ... (xxxvi) [32] -> [64, 6, 6]. 
    This can be efficiently done expanding y to have same 
    dimensionality as M such that:
    [32] torch.expand -> [32, 6, 6]
    and then concatenate on the channel dimension:
    [32, 6, 6] torch.cat(axis=0) -> [64, 6, 6] = [96, 6, 6]
    The tensor is then feed to the local discriminator.
    """

    def __init__(self, y_size, M_channels):
        super().__init__()
        self.c0 = nn.Conv2d(y_size + M_channels, 256, kernel_size=1)
        self.c1 = nn.Conv2d(256, 256, kernel_size=1)
        self.c2 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    """The prior discriminator class.

    This discriminate between a vector drawn from random uniform,
    and the vector y obtained as output of the encoder.
    It enforces y to be close to a uniform distribution.
    """

    def __init__(self, y_size):
        super().__init__()
        self.l0 = nn.Linear(y_size, 512)
        self.l1 = nn.Linear(512, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class DeepInfoMaxLoss(nn.Module):

    def __init__(self, y_size, M_channels, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        if alpha != 0.0:
            self.global_d = GlobalDiscriminator(y_size, M_channels)
        if beta != 0.0:
            self.local_d = LocalDiscriminator(y_size, M_channels)
        if gamma != 0.0:
            self.prior_d = PriorDiscriminator(y_size)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):
        if self.beta != 0.0:
            y_expanded = y.unsqueeze(-1).unsqueeze(-1)
            y_expanded = y_expanded.expand(-1, -1, M.shape[2], M.shape[3])
            y_M = torch.cat((M, y_expanded), dim=1)
            y_M_prime = torch.cat((M_prime, y_expanded), dim=1)
            Ej = -F.softplus(-self.local_d(y_M)).mean()
            Em = F.softplus(self.local_d(y_M_prime)).mean()
            LOCAL = (Em - Ej) * self.beta
        else:
            LOCAL = 0.0
        if self.alpha != 0.0:
            Ej = -F.softplus(-self.global_d(y, M)).mean()
            Em = F.softplus(self.global_d(y, M_prime)).mean()
            GLOBAL = (Em - Ej) * self.alpha
        else:
            GLOBAL = 0.0
        if self.gamma != 0.0:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = -(term_a + term_b) * self.gamma
        else:
            PRIOR = 0.0
        return LOCAL + GLOBAL + PRIOR


class DIM(nn.Module):

    def __init__(self, feature_extractor, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.encoder = Encoder(feature_extractor)
        y_size = self.encoder.y_size
        M_channels = self.encoder.M_channels
        self.loss_fn = DeepInfoMaxLoss(y_size, M_channels, alpha, beta, gamma)
        if torch.cuda.is_available():
            self.encoder = self.encoder
            self.loss_fn = self.loss_fn
        self.optimizer = Adam([{'params': self.encoder.parameters(), 'lr': 0.0001}, {'params': self.loss_fn.parameters(), 'lr': 0.0001}])

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.encoder.train()
        self.loss_fn.train()
        loss_meter = AverageMeter()
        for i, (data, _) in enumerate(train_loader):
            if torch.cuda.is_available():
                data = data
            self.optimizer.zero_grad()
            y, M = self.encoder(data)
            M_prime = torch.cat([M[1:], M[0].unsqueeze(0)], dim=0)
            loss = self.loss_fn(y, M, M_prime)
            loss_meter.update(loss.item(), data.shape[0])
            loss.backward()
            self.optimizer.step()
        elapsed_time = time.time() - start_time
        None
        return loss_meter.avg, -loss_meter.avg

    def save(self, file_path='./checkpoint.dat'):
        feature_extractor_state_dict = self.encoder.feature_extractor.state_dict()
        loss_fn_state_dict = self.loss_fn.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({'backbone': feature_extractor_state_dict, 'loss_fn': loss_fn_state_dict, 'optimizer': optimizer_state_dict}, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.encoder.feature_extractor.load_state_dict(checkpoint['backbone'])
        self.loss_fn.load_state_dict(checkpoint['loss_fn'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class FocalLoss(torch.nn.Module):
    """Sigmoid focal cross entropy loss.
  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
    Args:
      gamma: exponent of the modulating factor (1 - p_t)^gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives,
           with alpha in [0, 1] for class 1 and 1-alpha for class 0. 
           In practice alpha may be set by inverse class frequency,
           so that for a low number of positives, its weight is high.
    """
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self.BCEWithLogits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, prediction_tensor, target_tensor):
        """Compute loss function.
    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets.
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
        per_entry_cross_ent = self.BCEWithLogits(prediction_tensor, target_tensor)
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = target_tensor * prediction_probabilities + (1 - target_tensor) * (1 - prediction_probabilities)
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha)
        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
        return torch.mean(focal_cross_entropy_loss)


class StandardModel(torch.nn.Module):

    def __init__(self, feature_extractor, num_classes, tot_epochs=200):
        super(StandardModel, self).__init__()
        self.num_classes = num_classes
        self.tot_epochs = tot_epochs
        self.feature_extractor = feature_extractor
        feature_size = feature_extractor.feature_size
        self.classifier = nn.Linear(feature_size, num_classes)
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = SGD([{'params': self.feature_extractor.parameters(), 'lr': 0.1, 'momentum': 0.9}, {'params': self.classifier.parameters(), 'lr': 0.1, 'momentum': 0.9}])
        self.optimizer_lineval = Adam([{'params': self.classifier.parameters(), 'lr': 0.001}])
        self.optimizer_finetune = Adam([{'params': self.feature_extractor.parameters(), 'lr': 0.001, 'weight_decay': 1e-05}, {'params': self.classifier.parameters(), 'lr': 0.0001, 'weight_decay': 1e-05}])

    def forward(self, x, detach=False):
        if detach:
            out = self.feature_extractor(x).detach()
        else:
            out = self.feature_extractor(x)
        out = self.classifier(out)
        return out

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.feature_extractor.train()
        self.classifier.train()
        if epoch == int(self.tot_epochs * 0.5) or epoch == int(self.tot_epochs * 0.75):
            for i_g, g in enumerate(self.optimizer.param_groups):
                g['lr'] *= 0.1
                None
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for i, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data, target
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = 100.0 * correct / float(len(target))
            accuracy_meter.update(accuracy.item(), len(target))
        elapsed_time = time.time() - start_time
        None
        return loss_meter.avg, accuracy_meter.avg

    def linear_evaluation(self, epoch, train_loader):
        self.feature_extractor.eval()
        self.classifier.train()
        minibatch_iter = tqdm.tqdm(train_loader, desc=f'(Epoch {epoch}) Minibatch')
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data, target
            self.optimizer_lineval.zero_grad()
            output = self.forward(data, detach=True)
            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer_lineval.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = 100.0 * correct / float(len(target))
            accuracy_meter.update(accuracy.item(), len(target))
            minibatch_iter.set_postfix({'loss': loss_meter.avg, 'acc': accuracy_meter.avg})
        return loss_meter.avg, accuracy_meter.avg

    def finetune(self, epoch, train_loader):
        self.feature_extractor.train()
        self.classifier.train()
        if epoch == int(self.tot_epochs * 0.5) or epoch == int(self.tot_epochs * 0.75):
            for i_g, g in enumerate(self.optimizer_finetune.param_groups):
                g['lr'] *= 0.1
                None
        minibatch_iter = tqdm.tqdm(train_loader, desc=f'(Epoch {epoch}) Minibatch')
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data, target
            self.optimizer_finetune.zero_grad()
            output = self.forward(data)
            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer_finetune.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = 100.0 * correct / float(len(target))
            accuracy_meter.update(accuracy.item(), len(target))
            minibatch_iter.set_postfix({'loss': loss_meter.avg, 'acc': accuracy_meter.avg})
        return loss_meter.avg, accuracy_meter.avg

    def test(self, test_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        with torch.no_grad():
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data, target
                output = self.forward(data)
                loss = self.ce(output, target)
                loss_meter.update(loss.item(), len(target))
                pred = output.argmax(-1)
                correct = pred.eq(target.view_as(pred)).cpu().sum()
                accuracy = 100.0 * correct / float(len(target))
                accuracy_meter.update(accuracy.item(), len(target))
        return loss_meter.avg, accuracy_meter.avg

    def return_embeddings(self, data_loader, portion=0.5):
        self.feature_extractor.eval()
        embeddings_list = []
        target_list = []
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if torch.cuda.is_available():
                    data, target = data, target
                features = self.feature_extractor(data)
                embeddings_list.append(features)
                target_list.append(target)
                if i >= int(len(data_loader) * portion):
                    break
        return torch.cat(embeddings_list, dim=0).cpu().detach().numpy(), torch.cat(target_list, dim=0).cpu().detach().numpy()

    def save(self, file_path='./checkpoint.dat'):
        state_dict = self.classifier.state_dict()
        feature_extractor_state_dict = self.feature_extractor.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        optimizer_lineval_state_dict = self.optimizer_lineval.state_dict()
        optimizer_finetune_state_dict = self.optimizer_finetune.state_dict()
        torch.save({'classifier': state_dict, 'backbone': feature_extractor_state_dict, 'optimizer': optimizer_state_dict, 'optimizer_lineval': optimizer_lineval_state_dict, 'optimizer_finetune': optimizer_finetune_state_dict}, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.feature_extractor.load_state_dict(checkpoint['backbone'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer_lineval.load_state_dict(checkpoint['optimizer_lineval'])
        self.optimizer_finetune.load_state_dict(checkpoint['optimizer_finetune'])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalDiscriminator,
     lambda: ([], {'y_size': 4, 'M_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 64, 64])], {}),
     True),
    (LocalDiscriminator,
     lambda: ([], {'y_size': 4, 'M_channels': 4}),
     lambda: ([torch.rand([4, 8, 64, 64])], {}),
     True),
    (PriorDiscriminator,
     lambda: ([], {'y_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mpatacchiola_self_supervised_relational_reasoning(_paritybench_base):
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

