import sys
_module = sys.modules[__name__]
del sys
fedtorch = _module
comms = _module
algorithms = _module
distributed = _module
federated = _module
afl = _module
centered = _module
fedavg = _module
fedgate = _module
misc = _module
qffl = _module
qsparse = _module
scaffold = _module
fedavg = _module
fedgate = _module
misc = _module
qsparse = _module
scaffold = _module
trainings = _module
distributed = _module
afl = _module
apfl = _module
afl = _module
apfl = _module
drfa = _module
main = _module
perfedme = _module
drfa = _module
main = _module
utils = _module
eval = _module
eval_centered = _module
flow_utils = _module
components = _module
comps = _module
criterion = _module
dataset = _module
datasets = _module
loader = _module
adult_loader = _module
federated_datasets = _module
libsvm_datasets = _module
partition = _module
prepare_data = _module
preprocess_toolkit = _module
metrics = _module
model = _module
models = _module
convex = _module
least_square = _module
logistic_regression = _module
robust_least_square = _module
robust_logistic_regression = _module
nonconvex = _module
cnn = _module
densenet = _module
mlp = _module
resnet = _module
rnn = _module
robust_mlp = _module
wideresnet = _module
optimizer = _module
optimizers = _module
adam = _module
learning = _module
sgd = _module
scheduler = _module
logs = _module
check_training = _module
checkpoint = _module
logging = _module
meter = _module
nodes = _module
nodes = _module
nodes_centered = _module
parameters = _module
tools = _module
get_summary = _module
load_console_records = _module
plot_utils = _module
auxiliary = _module
dict2obj = _module
init_config = _module
op_files = _module
op_paths = _module
topology = _module
main = _module
main_centered = _module
run_mpi = _module

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


from copy import deepcopy


import time


import torch


import torch.distributed as dist


import torch.nn as nn


import pandas as pd


from numpy import loadtxt


from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split


from torch.utils.data import Dataset


import warnings


from scipy.special import softmax


from sklearn.datasets import load_svmlight_file


import random


import torchvision.datasets as datasets


import torchvision.transforms as transforms


import torch.nn.functional as F


import math


from collections import OrderedDict


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from copy import copy


import re


class Least_square(nn.Module):

    def __init__(self, dataset):
        super(Least_square, self).__init__()
        self.dataset = dataset
        self._determine_problem_dims()
        self.fc = nn.Linear(in_features=self.num_features, out_features=self.num_classes, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _determine_problem_dims(self):
        if self.dataset == 'epsilon':
            self.num_features = 2000
            self.num_classes = 1
        elif self.dataset == 'url':
            self.num_features = 3231961
            self.num_classes = 1
        elif self.dataset == 'rcv1':
            self.num_features = 47236
            self.num_classes = 1
        elif self.dataset == 'MSD':
            self.num_features = 90
            self.num_classes = 1
        else:
            raise ValueError('convex methods only support epsilon, url, YearPredictionMSD and rcv1 for the moment')


class LinearMAFL(nn.Module):

    def __init__(self, in_features, middle_features, out_features=1):
        super(LinearMAFL, self).__init__()
        self.middle_features = middle_features
        self.in_features = in_features
        self.out_features = out_features
        if self.in_features == 0:
            self.in_features = self.middle_features
        self.Z = nn.Linear(self.in_features, self.middle_features, bias=False)
        self.W = nn.Linear(self.middle_features, self.out_features, bias=True)
        self.core_params = self.W.parameters()
        self.extra_params = self.Z.parameters()

    @property
    def weight(self):
        return torch.matmul(self.W.weight, self.Z.weight)

    @property
    def bias(self):
        return self.W.bias

    def forward(self, x):
        return self.W(self.Z(x))


class LogisticRegression(torch.nn.Module):

    def __init__(self, dataset):
        super(LogisticRegression, self).__init__()
        self.dataset = dataset
        self._determine_problem_dims()
        self.fc = nn.Linear(in_features=self.num_features, out_features=self.num_classes, bias=True)
        self._weight_initialization()

    def forward(self, x):
        if self.dataset in ['mnist', 'cifar10', 'cifar100', 'fashion_mnist', 'emnist', 'emnist_full']:
            x = x.view(-1, self.num_features)
        x = self.fc(x)
        return x

    def _determine_problem_dims(self):
        if self.dataset == 'epsilon':
            self.num_features = 2000
            self.num_classes = 2
        elif self.dataset == 'url':
            self.num_features = 3231961
            self.num_classes = 2
        elif self.dataset == 'rcv1':
            self.num_features = 47236
            self.num_classes = 2
        elif self.dataset == 'higgs':
            self.num_features = 28
            self.num_classes = 2
        elif self.dataset == 'mnist':
            self.num_features = 784
            self.num_classes = 10
        elif self.dataset == 'emnist':
            self.num_features = 784
            self.num_classes = 10
        elif self.dataset == 'emnist_full':
            self.num_features = 784
            self.num_classes = 62
        elif self.dataset == 'cifar10':
            self.num_features = 3072
            self.num_classes = 10
        elif self.dataset == 'cifar100':
            self.num_features = 3072
            self.num_classes = 100
        elif self.dataset == 'fashion_mnist':
            self.num_features = 784
            self.num_classes = 10
        elif self.dataset == 'synthetic':
            self.num_features = 60
            self.num_classes = 10
        elif self.dataset == 'adult':
            self.num_features = 14
            self.num_classes = 2
        else:
            raise ValueError('convex methods only support epsilon, url, rcv1, higgs, synthetic, mnist, fashion_mnist, cifar, and adult for the moment')

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()


class Robust_Least_Square(torch.nn.Module):

    def __init__(self, dataset):
        super(Robust_Least_Square, self).__init__()
        self.dataset = dataset
        self._determine_problem_dims()
        self.noise = torch.nn.Parameter(torch.randn(self.num_features) * 0.001, requires_grad=True)
        self.fc = nn.Linear(in_features=self.num_features, out_features=self.num_classes, bias=True)

    def forward(self, x):
        x = self.fc(x + self.noise)
        return x

    def _determine_problem_dims(self):
        if self.dataset == 'epsilon':
            self.num_features = 2000
            self.num_classes = 1
        elif self.dataset == 'url':
            self.num_features = 3231961
            self.num_classes = 1
        elif self.dataset == 'rcv1':
            self.num_features = 47236
            self.num_classes = 1
        elif self.dataset == 'MSD':
            self.num_features = 90
            self.num_classes = 1
        elif self.dataset == 'synthetic':
            self.num_features = 60
            self.num_classes = 1
        else:
            raise ValueError('convex methods only support epsilon, url, YearPredictionMSD and rcv1 for the moment')


class RobustLogisticRegression(torch.nn.Module):

    def __init__(self, dataset):
        super(RobustLogisticRegression, self).__init__()
        self.dataset = dataset
        self._determine_problem_dims()
        self.noise = torch.nn.Parameter(torch.randn(self.num_features) * 0.001, requires_grad=True)
        self.fc = nn.Linear(in_features=self.num_features, out_features=self.num_classes, bias=True)
        self._weight_initialization()

    def forward(self, x):
        if self.dataset in ['mnist', 'cifar10', 'cifar100', 'fashion_mnist', 'emnist', 'emnist_full']:
            x = x.view(-1, self.num_features)
        x = self.fc(x + self.noise)
        return x

    def _determine_problem_dims(self):
        if self.dataset == 'epsilon':
            self.num_features = 2000
            self.num_classes = 2
        elif self.dataset == 'url':
            self.num_features = 3231961
            self.num_classes = 2
        elif self.dataset == 'rcv1':
            self.num_features = 47236
            self.num_classes = 2
        elif self.dataset == 'higgs':
            self.num_features = 28
            self.num_classes = 2
        elif self.dataset == 'mnist':
            self.num_features = 784
            self.num_classes = 10
        elif self.dataset == 'emnist':
            self.num_features = 784
            self.num_classes = 10
        elif self.dataset == 'emnist_full':
            self.num_features = 784
            self.num_classes = 62
        elif self.dataset == 'cifar10':
            self.num_features = 3072
            self.num_classes = 10
        elif self.dataset == 'cifar100':
            self.num_features = 3072
            self.num_classes = 100
        elif self.dataset == 'fashion_mnist':
            self.num_features = 784
            self.num_classes = 10
        elif self.dataset == 'synthetic':
            self.num_features = 60
            self.num_classes = 10
        elif self.dataset == 'adult':
            self.num_features = 12
            self.num_classes = 2
        else:
            raise ValueError('convex methods only support epsilon, url, rcv1, higgs, synthetic, mnist, fashion_mnist, cifar, and adult for the moment')

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()


class CNN(nn.Module):

    def __init__(self, dataset):
        super(CNN, self).__init__()
        self.dataset = dataset
        self.num_classes = self._decide_num_classes()
        self.num_channels = self._decide_num_channels()
        self.conv1 = nn.Conv2d(self.num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.rep_out_dim = self._decide_output_representation_size()
        self.fc1 = nn.Linear(self.rep_out_dim, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

    def _decide_num_channels(self):
        if self.dataset in ['cifar10', 'cifar100']:
            return 3
        elif self.dataset in ['mnist', 'fashion_mnist', 'emnist', 'emnist_full']:
            return 1

    def _decide_num_classes(self):
        if self.dataset in ['cifar10', 'mnist', 'fashion_mnist', 'emnist']:
            return 10
        elif self.dataset == 'cifar100':
            return 100
        elif self.dataset == 'emnist_full':
            return 62

    def _decide_input_feature_size(self):
        if 'mnist' in self.dataset:
            return 28 * 28
        elif 'cifar' in self.dataset:
            return 32 * 32 * 3
        else:
            raise NotImplementedError

    def _decide_output_representation_size(self):
        if 'mnist' in self.dataset:
            return 4 * 4 * 50
        elif 'cifar' in self.dataset:
            return 5 * 5 * 50
        else:
            raise NotImplementedError

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.rep_out_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicLayer(nn.Module):

    def __init__(self, num_channels, growth_rate, drop_rate=0.0):
        super(BasicLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.droprate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_planes)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_planes * 4)
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


class Transition(nn.Module):

    def __init__(self, num_channels, num_out_channels, drop_rate=0.0):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_channels, num_out_channels, kernel_size=1, bias=False)
        self.droprate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, dataset, net_depth, growth_rate, bc_mode, compression, drop_rate):
        super(DenseNet, self).__init__()
        self.dataset = dataset
        self.num_classes = self._decide_num_classes()
        is_small_inputs = 'imagenet' not in self.dataset
        self.avgpool_size = 8 if is_small_inputs else 7
        assert 0 < compression <= 1, 'compression should be between 0 and 1.'
        if is_small_inputs:
            num_blocks = 3
            num_layers_per_block = (net_depth - (num_blocks + 1)) // num_blocks
            if bc_mode:
                num_layers_per_block = num_layers_per_block // 2
            block_config = [num_layers_per_block] * num_blocks
        else:
            model_params = {(121): [6, 12, 24, 16], (169): [6, 12, 32, 32], (201): [6, 12, 48, 32], (264): [6, 12, 64, 48]}
            assert net_depth not in model_params.keys()
            block_config = model_params[net_depth]
        num_channels = 2 * growth_rate
        if is_small_inputs:
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False))]))
        else:
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_channels)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))]))
        for ind, num_layers in enumerate(block_config):
            block = self._make_dense(num_channels, growth_rate, num_layers, bc_mode, drop_rate)
            self.features.add_module('denseblock%d' % (ind + 1), block)
            num_channels += num_layers * growth_rate
            num_out_channels = int(math.floor(num_channels * compression))
            if ind != len(block_config) - 1:
                trans = Transition(num_channels, num_out_channels, drop_rate)
                self.features.add_module('transition%d' % (ind + 1), trans)
                num_channels = num_out_channels
        self.features.add_module('norm_final', nn.BatchNorm2d(num_channels))
        self.classifier = nn.Linear(num_channels, self.num_classes)
        self._weight_initialization()

    def _decide_num_classes(self):
        if self.dataset == 'cifar10' or self.dataset == 'svhn':
            return 10
        elif self.dataset == 'cifar100':
            return 100
        elif self.dataset == 'imagenet':
            return 1000

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, num_channels, growth_rate, num_layers_per_block, bc_mode, drop_rate):
        layers = []
        for i in range(int(num_layers_per_block)):
            if bc_mode:
                layers.append(Bottleneck(num_channels, growth_rate, drop_rate))
            else:
                layers.append(BasicLayer(num_channels, growth_rate, drop_rate))
            num_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class MLP(nn.Module):

    def __init__(self, dataset, num_layers, hidden_size, drop_rate):
        super(MLP, self).__init__()
        self.dataset = dataset
        self.num_layers = num_layers
        self.num_classes = self._decide_num_classes()
        input_size = self._decide_input_feature_size()
        for i in range(1, self.num_layers + 1):
            in_features = input_size if i == 1 else hidden_size
            out_features = hidden_size
            layer = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features, track_running_stats=False), nn.ReLU(), nn.Dropout(p=drop_rate))
            setattr(self, 'layer{}'.format(i), layer)
        self.fc = nn.Linear(hidden_size, self.num_classes, bias=False)

    def _decide_num_classes(self):
        if self.dataset in ['cifar10', 'mnist', 'fashion_mnist', 'emnist']:
            return 10
        elif self.dataset == 'cifar100':
            return 100
        elif self.dataset == 'emnist_full':
            return 62
        elif self.dataset == 'adult':
            return 2

    def _decide_input_feature_size(self):
        if 'cifar' in self.dataset:
            return 32 * 32 * 3
        elif 'mnist' in self.dataset:
            return 28 * 28
        elif self.dataset == 'adult':
            return 14
        else:
            raise NotImplementedError

    def forward(self, x):
        out = x.view(x.size(0), -1)
        for i in range(1, self.num_layers + 1):
            out = getattr(self, 'layer{}'.format(i))(out)
        out = self.fc(out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = drop_rate
        self.equal_in_out = in_planes == out_planes
        self.conv_shortcut = not self.equal_in_out and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equal_in_out else self.conv_shortcut(x), out)


class ResNetBase(nn.Module):

    def _decide_num_classes(self):
        if self.dataset == 'cifar10' or self.dataset == 'svhn':
            return 10
        elif self.dataset == 'cifar100':
            return 100
        elif self.dataset == 'imagenet':
            return 1000

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self, block_fn, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block_fn.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block_fn.expansion))
        layers = []
        layers.append(block_fn(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_fn.expansion
        for i in range(1, block_num):
            layers.append(block_fn(self.inplanes, planes))
        return nn.Sequential(*layers)


class ResNet_imagenet(ResNetBase):

    def __init__(self, dataset, resnet_size):
        super(ResNet_imagenet, self).__init__()
        self.dataset = dataset
        model_params = {(18): {'block': BasicBlock, 'layers': [2, 2, 2, 2]}, (34): {'block': BasicBlock, 'layers': [3, 4, 6, 3]}, (50): {'block': Bottleneck, 'layers': [3, 4, 6, 3]}, (101): {'block': Bottleneck, 'layers': [3, 4, 23, 3]}, (152): {'block': Bottleneck, 'layers': [3, 8, 36, 3]}}
        block_fn = model_params[resnet_size]['block']
        block_nums = model_params[resnet_size]['layers']
        self.num_classes = self._decide_num_classes()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_block(block_fn=block_fn, planes=64, block_num=block_nums[0])
        self.layer2 = self._make_block(block_fn=block_fn, planes=128, block_num=block_nums[1], stride=2)
        self.layer3 = self._make_block(block_fn=block_fn, planes=256, block_num=block_nums[2], stride=2)
        self.layer4 = self._make_block(block_fn=block_fn, planes=512, block_num=block_nums[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=512 * block_fn.expansion, out_features=self.num_classes)
        self._weight_initialization()

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet_cifar(ResNetBase):

    def __init__(self, dataset, resnet_size):
        super(ResNet_cifar, self).__init__()
        self.dataset = dataset
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock
        self.num_classes = self._decide_num_classes()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_block(block_fn=block_fn, planes=16, block_num=block_nums)
        self.layer2 = self._make_block(block_fn=block_fn, planes=32, block_num=block_nums, stride=2)
        self.layer3 = self._make_block(block_fn=block_fn, planes=64, block_num=block_nums, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_features=64 * block_fn.expansion, out_features=self.num_classes)
        self._weight_initialization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RNN(nn.Module):

    def __init__(self, dataset, input_size, hidden_size, output_size, batch_size, n_layers=1):
        super(RNN, self).__init__()
        self.dataset = dataset
        if self.dataset not in ['shakespeare']:
            raise NotImplementedError
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.init_hidden(self.batch_size)

    def forward(self, input):
        if self.hidden.size(1) != input.size(0):
            self.init_hidden(input.size(0))
        if self.hidden.device != input.device:
            self.hidden = self.hidden
        input = self.encoder(input)
        output, h = self.gru(input, self.hidden.detach())
        self.hidden.data = h.data
        output = self.decoder(output)
        return output.permute(0, 2, 1)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        weight = next(self.parameters())
        self.hidden = weight.new_zeros(self.n_layers, batch_size, self.hidden_size)
        return


class RobustMLP(nn.Module):

    def __init__(self, dataset, num_layers, hidden_size, drop_rate):
        super(RobustMLP, self).__init__()
        self.dataset = dataset
        self.num_layers = num_layers
        self.num_classes = self._decide_num_classes()
        input_size = self._decide_input_feature_size()
        self.noise = torch.nn.Parameter(torch.randn(input_size) * 0.001, requires_grad=True)
        for i in range(1, self.num_layers + 1):
            in_features = input_size if i == 1 else hidden_size
            out_features = hidden_size
            layer = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Dropout(p=drop_rate))
            setattr(self, 'layer{}'.format(i), layer)
        self.fc = nn.Linear(hidden_size, self.num_classes, bias=False)

    def _decide_num_classes(self):
        if self.dataset in ['cifar10', 'mnist', 'fashion_mnist', 'emnist']:
            return 10
        elif self.dataset == 'cifar100':
            return 100
        elif self.dataset == 'emnist_full':
            return 62
        elif self.dataset == 'adult':
            return 2

    def _decide_input_feature_size(self):
        if 'cifar' in self.dataset:
            return 32 * 32 * 3
        elif 'mnist' in self.dataset:
            return 28 * 28
        elif self.dataset == 'adult':
            return 14
        else:
            raise NotImplementedError

    def forward(self, x):
        out = x.view(x.size(0), -1) + self.noise
        for i in range(1, self.num_layers + 1):
            out = getattr(self, 'layer{}'.format(i))(out)
        out = self.fc(out)
        return out


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, dataset, net_depth, widen_factor, drop_rate):
        super(WideResNet, self).__init__()
        self.dataset = dataset
        assert (net_depth - 4) % 6 == 0
        num_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        num_blocks = (net_depth - 4) // 6
        block = BasicBlock
        self.num_classes = self._decide_num_classes()
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(num_blocks, num_channels[0], num_channels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(num_blocks, num_channels[1], num_channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(num_blocks, num_channels[2], num_channels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.num_channels = num_channels[3]
        self.fc = nn.Linear(num_channels[3], self.num_classes)
        self._weight_initialization()

    def _decide_num_classes(self):
        if self.dataset == 'cifar10' or self.dataset == 'svhn':
            return 10
        elif self.dataset == 'cifar100':
            return 100
        elif 'imagenet' in self.dataset:
            return 1000

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
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
        out = out.view(-1, self.num_channels)
        return self.fc(out)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicLayer,
     lambda: ([], {'num_channels': 4, 'growth_rate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearMAFL,
     lambda: ([], {'in_features': 4, 'middle_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_planes': 4, 'out_planes': 4, 'block': _mock_layer, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'num_channels': 4, 'num_out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_MLOPTPSU_FedTorch(_paritybench_base):
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

