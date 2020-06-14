import sys
_module = sys.modules[__name__]
del sys
ReIDdatasets = _module
main = _module
resnet = _module
trainers = _module
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


import torch.nn.functional as F


import math


import torch.utils.model_zoo as model_zoo


import torch


from torch.optim.lr_scheduler import MultiStepLR


from scipy.spatial.distance import cdist


import time


import numpy as np


from scipy.spatial.distance import pdist


from collections import OrderedDict


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last
        =False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

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
        if not self.is_last:
            out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            is_last=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, is_last=is_last))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_maps = self.layer4(x)
        x = self.avgpool(feature_maps)
        x = x.view(x.size(0), -1)
        feature = x.renorm(2, 0, 1e-05).mul(100000.0)
        w = self.fc.weight
        ww = w.renorm(2, 0, 1e-05).mul(100000.0)
        sim = feature.mm(ww.t())
        return feature, sim, feature_maps


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, residual_transform=None,
        output_activation='relu', norm='batch'):
        super(ResNetBasicblock, self).__init__()
        self.norm = norm
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        if norm == 'batch':
            self.bn_a = nn.BatchNorm2d(planes)
        elif norm == 'instance':
            self.bn_a = nn.InstanceNorm2d(planes)
        else:
            assert False, 'norm must be batch or instance'
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        if norm == 'batch':
            self.bn_b = nn.BatchNorm2d(planes)
        elif norm == 'instance':
            self.bn_b = nn.InstanceNorm2d(planes)
        else:
            assert False, 'norm must be batch or instance'
        self.residual_transform = residual_transform
        self.output_activation = nn.ReLU(
            ) if output_activation == 'relu' else nn.Tanh()

    def forward(self, x):
        residual = x
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
        if self.residual_transform is not None:
            residual = self.residual_transform(x)
        if residual.size()[1] > basicblock.size()[1]:
            residual = residual[:, :basicblock.size()[1], :, :]
        output = self.output_activation(residual + basicblock)
        return output


class AgentLoss(torch.nn.Module):

    def __init__(self):
        super(AgentLoss, self).__init__()

    def forward(self, features, agents, labels):
        """
        :param features: shape=(BS, dim)
        :param agents: shape=(n_class, dim)
        :param labels: shape=(BS, dim)
        :return:
        """
        respective_agents = agents[labels]
        similarity_matrix = (features * respective_agents).sum(dim=1)
        loss = 1 - similarity_matrix.mean()
        return loss


def dist_idx_to_pair_idx(d, i):
    """
    :param d: number of samples
    :param i: np.array
    :return:
    """
    if i.size == 0:
        return None
    b = 1 - 2 * d
    x = np.floor((-b - np.sqrt(b ** 2 - 8 * i)) / 2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    return x, y


class DiscriminativeLoss(torch.nn.Module):

    def __init__(self, mining_ratio=0.001):
        super(DiscriminativeLoss, self).__init__()
        self.mining_ratio = mining_ratio
        self.register_buffer('n_pos_pairs', torch.Tensor([0]))
        self.register_buffer('rate_TP', torch.Tensor([0]))
        self.moment = 0.1
        self.initialized = False

    def init_threshold(self, pairwise_agreements):
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = sorted_agreements[-pos]
        self.register_buffer('threshold', torch.Tensor([t]))
        self.initialized = True

    def forward(self, features, multilabels, labels):
        """
        :param features: shape=(BS, dim)
        :param multilabels: (BS, n_class)
        :param labels: (BS,)
        :return:
        """
        P, N = self._partition_sets(features.detach(), multilabels, labels)
        if P is None:
            pos_exponant = torch.Tensor([1])
            num = 0
        else:
            sdist_pos_pairs = []
            for i, j in zip(P[0], P[1]):
                sdist_pos_pair = (features[i] - features[j]).pow(2).sum()
                sdist_pos_pairs.append(sdist_pos_pair)
            pos_exponant = torch.exp(-torch.stack(sdist_pos_pairs)).mean()
            num = -torch.log(pos_exponant)
        if N is None:
            neg_exponant = torch.Tensor([0.5])
        else:
            sdist_neg_pairs = []
            for i, j in zip(N[0], N[1]):
                sdist_neg_pair = (features[i] - features[j]).pow(2).sum()
                sdist_neg_pairs.append(sdist_neg_pair)
            neg_exponant = torch.exp(-torch.stack(sdist_neg_pairs)).mean()
        den = torch.log(pos_exponant + neg_exponant)
        loss = num + den
        return loss

    def _partition_sets(self, features, multilabels, labels):
        """
        partition the batch into confident positive, hard negative and others
        :param features: shape=(BS, dim)
        :param multilabels: shape=(BS, n_class)
        :param labels: shape=(BS,)
        :return:
        P: positive pair set. tuple of 2 np.array i and j.
            i contains smaller indices and j larger indices in the batch.
            if P is None, no positive pair found in this batch.
        N: negative pair set. similar to P, but will never be None.
        """
        f_np = features.cpu().numpy()
        ml_np = multilabels.cpu().numpy()
        p_dist = pdist(f_np)
        p_agree = 1 - pdist(ml_np, 'minkowski', p=1) / 2
        sorting_idx = np.argsort(p_dist)
        n_similar = int(len(p_dist) * self.mining_ratio)
        similar_idx = sorting_idx[:n_similar]
        is_positive = p_agree[similar_idx] > self.threshold.item()
        pos_idx = similar_idx[is_positive]
        neg_idx = similar_idx[~is_positive]
        P = dist_idx_to_pair_idx(len(f_np), pos_idx)
        N = dist_idx_to_pair_idx(len(f_np), neg_idx)
        self._update_threshold(p_agree)
        self._update_buffers(P, labels)
        return P, N

    def _update_threshold(self, pairwise_agreements):
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = torch.Tensor([sorted_agreements[-pos]])
        self.threshold = self.threshold * (1 - self.moment) + t * self.moment

    def _update_buffers(self, P, labels):
        if P is None:
            self.n_pos_pairs = 0.9 * self.n_pos_pairs
            return 0
        n_pos_pairs = len(P[0])
        count = 0
        for i, j in zip(P[0], P[1]):
            count += labels[i] == labels[j]
        rate_TP = float(count) / n_pos_pairs
        self.n_pos_pairs = 0.9 * self.n_pos_pairs + 0.1 * n_pos_pairs
        self.rate_TP = 0.9 * self.rate_TP + 0.1 * rate_TP


class JointLoss(torch.nn.Module):

    def __init__(self, margin=1):
        super(JointLoss, self).__init__()
        self.margin = margin
        self.sim_margin = 1 - margin / 2

    def forward(self, features, agents, labels, similarity, features_target,
        similarity_target):
        """
        :param features: shape=(BS/2, dim)
        :param agents: shape=(n_class, dim)
        :param labels: shape=(BS/2,)
        :param features_target: shape=(BS/2, n_class)
        :return:
        """
        loss_terms = []
        arange = torch.arange(len(agents))
        zero = torch.Tensor([0])
        for f, l, s in zip(features, labels, similarity):
            loss_pos = (f - agents[l]).pow(2).sum()
            loss_terms.append(loss_pos)
            neg_idx = arange != l
            hard_agent_idx = neg_idx & (s > self.sim_margin)
            if torch.any(hard_agent_idx):
                hard_neg_sdist = (f - agents[hard_agent_idx]).pow(2).sum(dim=1)
                loss_neg = torch.max(zero, self.margin - hard_neg_sdist).mean()
                loss_terms.append(loss_neg)
        for f, s in zip(features_target, similarity_target):
            hard_agent_idx = s > self.sim_margin
            if torch.any(hard_agent_idx):
                hard_neg_sdist = (f - agents[hard_agent_idx]).pow(2).sum(dim=1)
                loss_neg = torch.max(zero, self.margin - hard_neg_sdist).mean()
                loss_terms.append(loss_neg)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total


class MultilabelLoss(torch.nn.Module):

    def __init__(self, batch_size, use_std=True):
        super(MultilabelLoss, self).__init__()
        self.use_std = use_std
        self.moment = batch_size / 10000
        self.initialized = False

    def init_centers(self, log_multilabels, views):
        """
        :param log_multilabels: shape=(N, n_class)
        :param views: (N,)
        :return:
        """
        univiews = torch.unique(views)
        mean_ml = []
        std_ml = []
        for v in univiews:
            ml_in_v = log_multilabels[views == v]
            mean = ml_in_v.mean(dim=0)
            std = ml_in_v.std(dim=0)
            mean_ml.append(mean)
            std_ml.append(std)
        center_mean = torch.mean(torch.stack(mean_ml), dim=0)
        center_std = torch.mean(torch.stack(std_ml), dim=0)
        self.register_buffer('center_mean', center_mean)
        self.register_buffer('center_std', center_std)
        self.initialized = True

    def _update_centers(self, log_multilabels, views):
        """
        :param log_multilabels: shape=(BS, n_class)
        :param views: shape=(BS,)
        :return:
        """
        univiews = torch.unique(views)
        means = []
        stds = []
        for v in univiews:
            ml_in_v = log_multilabels[views == v]
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            means.append(mean)
            if self.use_std:
                std = ml_in_v.std(dim=0)
                stds.append(std)
        new_mean = torch.mean(torch.stack(means), dim=0)
        self.center_mean = self.center_mean * (1 - self.moment
            ) + new_mean * self.moment
        if self.use_std:
            new_std = torch.mean(torch.stack(stds), dim=0)
            self.center_std = self.center_std * (1 - self.moment
                ) + new_std * self.moment

    def forward(self, log_multilabels, views):
        """
        :param log_multilabels: shape=(BS, n_class)
        :param views: shape=(BS,)
        :return:
        """
        self._update_centers(log_multilabels.detach(), views)
        univiews = torch.unique(views)
        loss_terms = []
        for v in univiews:
            ml_in_v = log_multilabels[views == v]
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            loss_mean = (mean - self.center_mean).pow(2).sum()
            loss_terms.append(loss_mean)
            if self.use_std:
                std = ml_in_v.std(dim=0)
                loss_std = (std - self.center_std).pow(2).sum()
                loss_terms.append(loss_std)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_KovenYu_MAR(_paritybench_base):
    pass
    def test_000(self):
        self._check(ResNetBasicblock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

