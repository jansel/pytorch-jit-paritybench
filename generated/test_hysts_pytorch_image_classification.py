import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
pytorch_image_classification = _module
collators = _module
cutmix = _module
mixup = _module
ricap = _module
config = _module
config_node = _module
defaults = _module
datasets = _module
dataloader = _module
datasets = _module
losses = _module
cutmix = _module
dual_cutout = _module
label_smoothing = _module
mixup = _module
ricap = _module
models = _module
cifar = _module
densenet = _module
pyramidnet = _module
resnet = _module
resnet_preact = _module
resnext = _module
se_resnet_preact = _module
shake_shake = _module
vgg = _module
wrn = _module
functions = _module
shake_shake_function = _module
imagenet = _module
densenet = _module
pyramidnet = _module
resnet = _module
resnet_preact = _module
resnext = _module
vgg = _module
initializer = _module
optim = _module
adabound = _module
lars = _module
scheduler = _module
combined_scheduler = _module
components = _module
multistep_scheduler = _module
sgdr = _module
transforms = _module
cutout = _module
random_erasing = _module
transforms = _module
utils = _module
diff_config = _module
dist = _module
env_info = _module
logger = _module
metric_logger = _module
metrics = _module
op_count = _module
tensorboard = _module
utils = _module
extract_images = _module
extract_scalars = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import torch


import torch.nn.functional as F


from typing import Callable


from typing import List


from typing import Tuple


from typing import Union


import torch.distributed as dist


from torch.utils.data import DataLoader


import torchvision


from torch.utils.data import Dataset


import torch.nn as nn


from torch.autograd import Function


import math


from torch.utils.tensorboard import SummaryWriter


import random


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, remove_first_relu, add_last_bn, preact=False):
        super().__init__()
        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if add_last_bn:
            self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))

    def forward(self, x):
        if self._preact:
            x = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(x)
        else:
            y = self.bn1(x)
            if not self._remove_first_relu:
                y = F.relu(y, inplace=True)
            y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        if self._add_last_bn:
            y = self.bn3(y)
        y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, stage_index, base_channels, cardinality):
        super().__init__()
        bottleneck_channels = cardinality * base_channels * 2 ** stage_index
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class TransitionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training, inplace=False)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def create_initializer(mode: str) ->Callable:
    if mode in ['kaiming_fan_out', 'kaiming_fan_in']:
        mode = mode[8:]

        def initializer(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, mode=mode, nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data, mode=mode, nonlinearity='relu')
                nn.init.zeros_(module.bias.data)
    else:
        raise ValueError()
    return initializer


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.vgg
        self.use_bn = model_config.use_bn
        n_channels = model_config.n_channels
        n_layers = model_config.n_layers
        self.stage1 = self._make_stage(config.dataset.n_channels, n_channels[0], n_layers[0])
        self.stage2 = self._make_stage(n_channels[0], n_channels[1], n_layers[1])
        self.stage3 = self._make_stage(n_channels[1], n_channels[2], n_layers[2])
        self.stage4 = self._make_stage(n_channels[2], n_channels[3], n_layers[3])
        self.stage5 = self._make_stage(n_channels[3], n_channels[4], n_layers[4])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.dataset.image_size, config.dataset.image_size), dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0]
        self.fc1 = nn.Linear(self.feature_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks):
        stage = nn.Sequential()
        for index in range(n_blocks):
            if index == 0:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            stage.add_module(f'conv{index}', conv)
            if self.use_bn:
                stage.add_module(f'bn{index}', nn.BatchNorm2d(out_channels))
            stage.add_module('relu', nn.ReLU(inplace=True))
        stage.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))
        return stage

    def _forward_conv(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x), inplace=True), training=self.training)
        x = F.dropout(F.relu(self.fc2(x), inplace=True), training=self.training)
        x = self.fc3(x)
        return x


class SELayer(nn.Module):

    def __init__(self, in_channels, reduction):
        super().__init__()
        mid_channels = in_channels // reduction
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        n_batches, n_channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, output_size=1).view(n_batches, n_channels)
        y = F.relu(self.fc1(y), inplace=True)
        y = F.sigmoid(self.fc2(y)).view(n_batches, n_channels, 1, 1)
        return x * y


class ResidualPath(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.bn2(self.conv2(x))
        return x


class SkipConnection(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        x = F.relu(x, inplace=False)
        y1 = F.avg_pool2d(x, kernel_size=1, stride=self.stride, padding=0)
        y1 = self.conv1(y1)
        y2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        y2 = F.avg_pool2d(y2, kernel_size=1, stride=self.stride, padding=0)
        y2 = self.conv2(y2)
        z = torch.cat([y1, y2], dim=1)
        z = self.bn(z)
        return z


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'remove_first_relu': 4, 'add_last_bn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'stage_index': 4, 'base_channels': 4, 'cardinality': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualPath,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SELayer,
     lambda: ([], {'in_channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SkipConnection,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransitionBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_hysts_pytorch_image_classification(_paritybench_base):
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

