import sys
_module = sys.modules[__name__]
del sys
write_CUB_filelist = _module
write_miniImagenet_base_val_filelist = _module
write_miniImagenet_cross_filelist = _module
write_miniImagenet_filelist = _module
backbone = _module
conv4 = _module
resnet12 = _module
resnet18 = _module
wrn = _module
config = _module
dataset = _module
dropblock = _module
metric = _module
model = _module
utils = _module
main = _module

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


import math


import torch.nn as nn


import torch


import torch.nn.functional as F


from torchvision import transforms


from torch import nn


from torch.distributions import Bernoulli


from torch import nn as nn


from torch.nn import Parameter


from torch.nn import functional as F


import numpy as np


from torch.optim.lr_scheduler import _LRScheduler


import random


import time


from collections import defaultdict


import torch.optim


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.utils.tensorboard import SummaryWriter


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class ConvBlock(nn.Module):

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)
        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):

    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=i < 4)
            trunk.append(B)
        if flatten:
            trunk.append(Flatten())
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out


class DropBlock(nn.Module):

    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)))
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()
            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]
        offsets = torch.stack([torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), torch.arange(self.block_size).repeat(self.block_size)]).t()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).long(), offsets.long()), 1)
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()
            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.0
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
        block_mask = 1 - padded_mask
        return block_mask


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.C1 = conv3x3(inplanes, planes)
        self.BN1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.C2 = conv3x3(planes, planes)
        self.BN2 = nn.BatchNorm2d(planes)
        self.C3 = conv3x3(planes, planes)
        self.BN3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.drop_rate = drop_rate
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        num_batches_tracked = int(self.BN1.num_batches_tracked.cpu().data)
        residual = x
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class ResNet12(nn.Module):

    def __init__(self, block, avg_pool=True, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet12, self).__init__()
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
            self.final_feat_dim = 640
        else:
            self.final_feat_dim = [640, 5, 5]
        self.keep_avg_pool = avg_pool
        self.num_batches_tracked = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size)]
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class SimpleBlock(nn.Module):

    def __init__(self, indim, outdim, half_res, drop_rate, block_size, drop_block=False):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
        self.half_res = half_res
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
        for layer in self.parametrized_layers:
            init_layer(layer)
        self.drop_rate = drop_rate
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        num_batches_tracked = int(self.BN1.num_batches_tracked.cpu().data)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class ResNet18(nn.Module):

    def __init__(self, block, list_of_out_dims, flatten=True, drop_rate=0.1, dropblock_size=5):
        super(ResNet18, self).__init__()
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        init_layer(conv1)
        init_layer(bn1)
        trunk = [conv1, bn1, relu, pool1]
        self.trunk = nn.Sequential(*trunk)
        self.layer1 = self._make_layer(block, 64, list_of_out_dims[0], half_res=False, drop_rate=drop_rate, dropblock_size=dropblock_size)
        self.layer2 = self._make_layer(block, list_of_out_dims[0], list_of_out_dims[1], half_res=True, drop_rate=drop_rate, dropblock_size=dropblock_size)
        self.layer3 = self._make_layer(block, list_of_out_dims[1], list_of_out_dims[2], half_res=True, drop_rate=drop_rate, dropblock_size=dropblock_size, drop_block=True)
        self.layer4 = self._make_layer(block, list_of_out_dims[2], list_of_out_dims[3], half_res=True, drop_rate=drop_rate, dropblock_size=dropblock_size, drop_block=True)
        if flatten:
            self.avgpool = nn.AvgPool2d(7)
            self.Flatten = Flatten()
            self.final_feat_dim = list_of_out_dims[3]
        else:
            self.final_feat_dim = [list_of_out_dims[3], 7, 7]
        self.flatten = flatten
        self.num_batches_tracked = 0

    def _make_layer(self, block, indim, outdim, half_res=False, drop_rate=0.1, dropblock_size=5, drop_block=False):
        layers = [block(indim, outdim, half_res, drop_rate, dropblock_size, drop_block=drop_block), block(outdim, outdim, False, drop_rate, dropblock_size, drop_block=drop_block)]
        return nn.Sequential(*layers)

    def forward(self, x):
        self.num_batches_tracked += 1
        x = self.trunk(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.flatten:
            x = self.avgpool(x)
            x = self.Flatten(x)
        return x


class WRNBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(WRNBlock, self).__init__()
        self.BN1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.C1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.C2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.BN1(x))
        else:
            out = self.relu1(self.BN1(x))
        out = self.relu2(self.BN2(self.C1(out if self.equalInOut else x)))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.C2(out)
        short_out = x if self.equalInOut else self.convShortcut(x)
        out = out + short_out
        return out


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor=1, stride=1, drop_rate=0.5):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = WRNBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(n, nChannels[0], nChannels[1], block, stride, drop_rate)
        self.layer2 = self._make_layer(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        self.layer3 = self._make_layer(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.final_feat_dim = 640
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x


class CosineSimilarity(nn.Module):

    def __init__(self, in_features, out_features, scale_factor=30.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.weight = Parameter(torch.Tensor(out_features, in_features).float())
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        return cosine * self.scale_factor


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


class AddMarginProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    :return： (theta) - m
    """

    def __init__(self, in_features, out_features, scale_factor=30.0, margin=0.4):
        super(AddMarginProduct, self).__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        if label is None:
            return cosine * self.scale_factor
        phi = cosine - self.margin
        output = torch.where(one_hot(label, cosine.shape[1]).byte(), phi, cosine)
        output *= self.scale_factor
        return output


class SoftmaxMargin(nn.Module):
    """Implement of softmax with margin:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    """

    def __init__(self, in_features, out_features, scale_factor=5.0, margin=0.4):
        super(SoftmaxMargin, self).__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        z = F.linear(feature, self.weight)
        z -= z.min(dim=1, keepdim=True)[0]
        if label is None:
            return z * self.scale_factor
        phi = z - self.margin
        output = torch.where(one_hot(label, z.shape[1]).byte(), phi, z)
        output *= self.scale_factor
        return output


def get_linear_clf(metric_type, feature_dimension, num_classes, scale_factor=None, margin=None):
    if metric_type == 'softmax':
        classifier = nn.Linear(feature_dimension, num_classes)
    elif metric_type == 'cosine':
        classifier = metric.CosineSimilarity(feature_dimension, num_classes, scale_factor=scale_factor)
    elif metric_type == 'cosineface':
        classifier = metric.AddMarginProduct(feature_dimension, num_classes, scale_factor=scale_factor, margin=margin)
    elif metric_type == 'neg-softmax':
        classifier = metric.SoftmaxMargin(feature_dimension, num_classes, scale_factor=scale_factor, margin=margin)
    else:
        raise ValueError(f'Unknown metric type: "{metric_type}"')
    return classifier


class BaselineTrain(nn.Module):

    def __init__(self, model_func, num_class, metric_type, metric_params):
        super(BaselineTrain, self).__init__()
        self.feature = model_func()
        self.metric_type = metric_type
        self.classifier = get_linear_clf(metric_type, self.feature.final_feat_dim, num_class, **metric_params)

    def forward(self, x, y=None):
        feature = self.feature.forward(x)
        if self.metric_type in ['cosineface', 'neg-softmax']:
            scores = self.classifier.forward(feature, y)
        else:
            scores = self.classifier.forward(feature)
        return scores


def get_few_shot_label(n_way, n_data_per_way):
    return torch.from_numpy(np.repeat(range(n_way), n_data_per_way))


class BaselineFinetune(nn.Module):

    def __init__(self, n_way, n_support, metric_type, metric_params, finetune_params):
        super(BaselineFinetune, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.metric_type = metric_type
        self.metric_params = metric_params
        self.finetune_params = finetune_params

    def forward(self, z_all):
        z_all = z_all
        z_support = z_all[:, :self.n_support, :]
        z_query = z_all[:, self.n_support:, :]
        feature_dim = z_support.shape[-1]
        z_support = z_support.contiguous().view(-1, feature_dim)
        z_query = z_query.contiguous().view(-1, feature_dim)
        y_support = get_few_shot_label(self.n_way, self.n_support)
        linear_clf = get_linear_clf(self.metric_type, feature_dim, self.n_way, **self.metric_params)
        if self.finetune_params.optim == 'SGD':
            finetune_optimizer = torch.optim.SGD(linear_clf.parameters(), **self.finetune_params.sgd_params)
        else:
            raise ValueError(f'finetune optimzation not supported: {self.finetune_params.optim}')
        loss_function = nn.CrossEntropyLoss()
        batch_size = 4
        support_size = self.n_way * self.n_support
        for _ in range(self.finetune_params.iter):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                selected_id = torch.from_numpy(rand_id[i:min(i + batch_size, support_size)])
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                if self.metric_type in ['cosineface', 'neg-softmax']:
                    scores = linear_clf(z_batch, y_batch)
                else:
                    scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                finetune_optimizer.zero_grad()
                loss.backward()
                finetune_optimizer.step()
        scores = linear_clf(z_query)
        return scores


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddMarginProduct,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'indim': 4, 'outdim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNet,
     lambda: ([], {'depth': 1}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (CosineSimilarity,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropBlock,
     lambda: ([], {'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleBlock,
     lambda: ([], {'indim': 4, 'outdim': 4, 'half_res': 4, 'drop_rate': 0.5, 'block_size': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     False),
    (SoftmaxMargin,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (WRNBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_bl0_negative_margin_few_shot(_paritybench_base):
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

