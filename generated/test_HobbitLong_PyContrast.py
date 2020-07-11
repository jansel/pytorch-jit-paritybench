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
util = _module
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


import numpy as np


import torch


from torchvision import datasets


from torchvision import transforms


import torch.nn as nn


import math


import torch.distributed as dist


import torch.backends.cudnn as cudnn


import time


from torch.nn.parallel import DistributedDataParallel as DDP


from collections import OrderedDict


import torch.utils.data.distributed


import torch.multiprocessing as mp


import torch.nn.functional as F


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


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = self.prob[large] - 1.0 + self.prob[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in (smaller + larger):
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob
        self.alias = self.alias

    def draw(self, N):
        """
        Draw N samples from multinomial
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)
        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())
        return oq + oj


class RGBMem(BaseMem):
    """Memory bank for single modality"""

    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5):
        super(RGBMem, self).__init__(K, T, m)
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial
        self.register_buffer('memory', torch.randn(n_data, n_dim))
        self.memory = F.normalize(self.memory)

    def forward(self, x, y, x_jig=None, all_x=None, all_y=None):
        """
        Args:
          x: feat on current node
          y: index on current node
          x_jig: jigsaw feat on current node
          all_x: gather of feats across nodes; otherwise use x
          all_y: gather of index across nodes; otherwise use y
        """
        bsz = x.size(0)
        n_dim = x.size(1)
        idx = self.multinomial.draw(bsz * (self.K + 1)).view(bsz, -1)
        idx.select(1, 0).copy_(y.data)
        w = torch.index_select(self.memory, 0, idx.view(-1))
        w = w.view(bsz, self.K + 1, n_dim)
        logits = self._compute_logit(x, w)
        if x_jig is not None:
            logits_jig = self._compute_logit(x_jig, w)
        labels = torch.zeros(bsz, dtype=torch.long)
        if all_x is not None and all_y is not None:
            self._update_memory(self.memory, all_x, all_y)
        else:
            self._update_memory(self.memory, x, y)
        if x_jig is not None:
            return logits, logits_jig, labels
        else:
            return logits, labels


class CMCMem(BaseMem):
    """Memory bank for two modalities, e.g. in CMC"""

    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5):
        super(CMCMem, self).__init__(K, T, m)
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial
        self.register_buffer('memory_1', torch.randn(n_data, n_dim))
        self.register_buffer('memory_2', torch.randn(n_data, n_dim))
        self.memory_1 = F.normalize(self.memory_1)
        self.memory_2 = F.normalize(self.memory_2)

    def forward(self, x1, x2, y, x1_jig=None, x2_jig=None, all_x1=None, all_x2=None, all_y=None):
        """
        Args:
          x1: feat of modal 1
          x2: feat of modal 2
          y: index on current node
          x1_jig: jigsaw feat of modal1
          x2_jig: jigsaw feat of modal2
          all_x1: gather of feats across nodes; otherwise use x1
          all_x2: gather of feats across nodes; otherwise use x2
          all_y: gather of index across nodes; otherwise use y
        """
        bsz = x1.size(0)
        n_dim = x1.size(1)
        idx = self.multinomial.draw(bsz * (self.K + 1)).view(bsz, -1)
        idx.select(1, 0).copy_(y.data)
        w1 = torch.index_select(self.memory_1, 0, idx.view(-1))
        w1 = w1.view(bsz, self.K + 1, n_dim)
        w2 = torch.index_select(self.memory_2, 0, idx.view(-1))
        w2 = w2.view(bsz, self.K + 1, n_dim)
        logits1 = self._compute_logit(x1, w2)
        logits2 = self._compute_logit(x2, w1)
        if x1_jig is not None and x2_jig is not None:
            logits1_jig = self._compute_logit(x1_jig, w2)
            logits2_jig = self._compute_logit(x2_jig, w1)
        labels = torch.zeros(bsz, dtype=torch.long)
        if all_x1 is not None and all_x2 is not None and all_y is not None:
            self._update_memory(self.memory_1, all_x1, all_y)
            self._update_memory(self.memory_2, all_x2, all_y)
        else:
            self._update_memory(self.memory_1, x1, y)
            self._update_memory(self.memory_2, x2, y)
        if x1_jig is not None and x2_jig is not None:
            return logits1, logits2, logits1_jig, logits2_jig, labels
        else:
            return logits1, logits2, labels


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


class RGBMoCo(BaseMoCo):
    """Single Modal (e.g., RGB) MoCo-style cache"""

    def __init__(self, n_dim, K=65536, T=0.07):
        super(RGBMoCo, self).__init__(K, T)
        self.register_buffer('memory', torch.randn(K, n_dim))
        self.memory = F.normalize(self.memory)

    def forward(self, q, k, q_jig=None, all_k=None):
        """
        Args:
          q: query on current node
          k: key on current node
          q_jig: jigsaw query
          all_k: gather of feats across nodes; otherwise use q
        """
        bsz = q.size(0)
        k = k.detach()
        queue = self.memory.clone().detach()
        logits = self._compute_logit(q, k, queue)
        if q_jig is not None:
            logits_jig = self._compute_logit(q_jig, k, queue)
        labels = torch.zeros(bsz, dtype=torch.long)
        all_k = all_k if all_k is not None else k
        self._update_memory(all_k, self.memory)
        self._update_pointer(all_k.size(0))
        if q_jig is not None:
            return logits, logits_jig, labels
        else:
            return logits, labels


class CMCMoCo(BaseMoCo):
    """MoCo-style memory for two modalities, e.g. in CMC"""

    def __init__(self, n_dim, K=65536, T=0.07):
        super(CMCMoCo, self).__init__(K, T)
        self.register_buffer('memory_1', torch.randn(K, n_dim))
        self.register_buffer('memory_2', torch.randn(K, n_dim))
        self.memory_1 = F.normalize(self.memory_1)
        self.memory_2 = F.normalize(self.memory_2)

    def forward(self, q1, k1, q2, k2, q1_jig=None, q2_jig=None, all_k1=None, all_k2=None):
        """
        Args:
          q1: q of modal 1
          k1: k of modal 1
          q2: q of modal 2
          k2: k of modal 2
          q1_jig: q jig of modal 1
          q2_jig: q jig of modal 2
          all_k1: gather of k1 across nodes; otherwise use k1
          all_k2: gather of k2 across nodes; otherwise use k2
        """
        bsz = q1.size(0)
        k1 = k1.detach()
        k2 = k2.detach()
        queue1 = self.memory_1.clone().detach()
        queue2 = self.memory_2.clone().detach()
        logits1 = self._compute_logit(q1, k2, queue2)
        logits2 = self._compute_logit(q2, k1, queue1)
        if q1_jig is not None and q2_jig is not None:
            logits1_jig = self._compute_logit(q1_jig, k2, queue2)
            logits2_jig = self._compute_logit(q2_jig, k1, queue1)
        labels = torch.zeros(bsz, dtype=torch.long)
        all_k1 = all_k1 if all_k1 is not None else k1
        all_k2 = all_k2 if all_k2 is not None else k2
        assert all_k1.size(0) == all_k2.size(0)
        self._update_memory(all_k1, self.memory_1)
        self._update_memory(all_k2, self.memory_2)
        self._update_pointer(all_k1.size(0))
        if q1_jig is not None and q2_jig is not None:
            return logits1, logits2, logits1_jig, logits2_jig, labels
        else:
            return logits1, logits2, labels


class Normalize(nn.Module):

    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, groups=2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, groups=2, bias=False)
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
        self.conv1_v1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_v2 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.base = int(64 * width)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, groups=2, bias=False), nn.BatchNorm2d(planes * block.expansion))
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


def resnest101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True, stem_width=64, avg_down=True, avd=True, avd_first=False, **kwargs)
    if pretrained:
        raise NotImplementedError('pretrained model not available')
    return model


def resnest50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True, stem_width=32, avg_down=True, avd=True, avd_first=False, **kwargs)
    if pretrained:
        raise NotImplementedError('pretrained model not available')
    return model


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


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnext101_32x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-101 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext101_32x4d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnext101_64x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-101 64x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return _resnet('resnext101_64x4d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnext152_32x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-152 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext152_32x4d', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext152_32x8d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-152 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext152_32x8d', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext152_64x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-152 64x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return _resnet('resnext152_64x4d', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


model_dict = {'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152, 'resnext50': resnext50_32x4d, 'resnext101v1': resnext101_32x4d, 'resnext101v2': resnext101_32x8d, 'resnext101v3': resnext101_64x4d, 'resnext152v1': resnext152_32x4d, 'resnext152v2': resnext152_32x8d, 'resnext152v3': resnext152_64x4d, 'resnest50': resnest50, 'resnest101': resnest101}


class RGBSingleHead(nn.Module):
    """RGB model with a single linear/mlp projection head"""

    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBSingleHead, self).__init__()
        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width
        self.encoder = model_dict[name](width=width)
        if head == 'linear':
            self.head = nn.Sequential(nn.Linear(dim_in, feat_dim), Normalize(2))
        elif head == 'mlp':
            self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim), Normalize(2))
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


class JigsawHead(nn.Module):
    """Jigswa + linear + l2norm"""

    def __init__(self, dim_in, dim_out, k=9, head='linear'):
        super(JigsawHead, self).__init__()
        if head == 'linear':
            self.fc1 = nn.Linear(dim_in, dim_out)
        elif head == 'mlp':
            self.fc1 = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, dim_out))
        else:
            raise NotImplementedError('JigSaw head not supported: {}'.format(head))
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


class RGBMultiHeads(RGBSingleHead):
    """RGB model with Multiple linear/mlp projection heads"""

    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBMultiHeads, self).__init__(name, head, feat_dim)
        self.head_jig = JigsawHead(dim_in=int(2048 * self.width), dim_out=feat_dim, head=head)

    def forward(self, x, x_jig=None, mode=0):
        if mode == 0:
            feat = self.head(self.encoder(x))
            feat_jig = self.head_jig(self.encoder(x_jig))
            return feat, feat_jig
        elif mode == 1:
            feat = self.head(self.encoder(x))
            return feat
        else:
            feat = self.encoder(x)
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
            self.head1 = nn.Sequential(nn.Linear(dim_in, feat_dim), Normalize(2))
            self.head2 = nn.Sequential(nn.Linear(dim_in, feat_dim), Normalize(2))
        elif head == 'mlp':
            self.head1 = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim), Normalize(2))
            self.head2 = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim), Normalize(2))
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


class CMCMultiHeads(CMCSingleHead):
    """CMC model with Multiple linear/mlp projection heads"""

    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(CMCMultiHeads, self).__init__(name, head, feat_dim)
        self.head1_jig = JigsawHead(dim_in=int(2048 * self.width), dim_out=feat_dim, head=head)
        self.head2_jig = JigsawHead(dim_in=int(2048 * self.width), dim_out=feat_dim, head=head)

    def forward(self, x, x_jig=None, mode=0):
        x1, x2 = torch.split(x, [1, 2], dim=1)
        feat1 = self.encoder1(x1)
        feat2 = self.encoder2(x2)
        if mode == 0:
            x1_jig, x2_jig = torch.split(x_jig, [1, 2], dim=1)
            feat1_jig = self.encoder1(x1_jig)
            feat2_jig = self.encoder2(x2_jig)
            feat1, feat2 = self.head1(feat1), self.head2(feat2)
            feat1_jig = self.head1_jig(feat1_jig)
            feat2_jig = self.head2_jig(feat2_jig)
            feat = torch.cat((feat1, feat2), dim=1)
            feat_jig = torch.cat((feat1_jig, feat2_jig), dim=1)
            return feat, feat_jig
        elif mode == 1:
            feat1, feat2 = self.head1(feat1), self.head2(feat2)
            return torch.cat((feat1, feat2), dim=1)
        else:
            return torch.cat((feat1, feat2), dim=1)


class DropBlock2D(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = F.sigmoid(atten, dim=1).view(batch, -1, 1, 1)
        if self.radix > 1:
            atten = torch.split(atten, channel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=2, bias=False)


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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CMCMoCo,
     lambda: ([], {'n_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Normalize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RGBMoCo,
     lambda: ([], {'n_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_HobbitLong_PyContrast(_paritybench_base):
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

