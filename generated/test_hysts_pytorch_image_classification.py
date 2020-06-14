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
utils = _module
diff_config = _module
dist = _module
env_info = _module
logger = _module
metric_logger = _module
metrics = _module
op_count = _module
tensorboard = _module
extract_images = _module
extract_scalars = _module
train = _module

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


import time


import numpy as np


import torch


import torch.nn.functional as F


from typing import Callable


from typing import Tuple


import torch.nn as nn


from typing import List


import torch.distributed as dist


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        return torch.cat([x, y], dim=1)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        bottleneck_channels = out_channels * 4
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        return torch.cat([x, y], dim=1)


class TransitionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training,
                inplace=False)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def create_initializer(mode: str) ->Callable:
    if mode in ['kaiming_fan_out', 'kaiming_fan_in']:
        mode = mode[8:]

        def initializer(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, mode=mode,
                    nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data, mode=mode,
                    nonlinearity='relu')
                nn.init.zeros_(module.bias.data)
    else:
        raise ValueError()
    return initializer


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.densenet
        depth = model_config.depth
        block_type = model_config.block_type
        self.growth_rate = model_config.growth_rate
        self.drop_rate = model_config.drop_rate
        self.compression_rate = model_config.compression_rate
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 4) // 3
            assert n_blocks_per_stage * 3 + 4 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 4) // 6
            assert n_blocks_per_stage * 6 + 4 == depth
        in_channels = [2 * self.growth_rate]
        for index in range(3):
            denseblock_out_channels = int(in_channels[-1] + 
                n_blocks_per_stage * self.growth_rate)
            if index < 2:
                transitionblock_out_channels = int(denseblock_out_channels *
                    self.compression_rate)
            else:
                transitionblock_out_channels = denseblock_out_channels
            in_channels.append(transitionblock_out_channels)
        self.conv = nn.Conv2d(config.dataset.n_channels, in_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(in_channels[0], n_blocks_per_stage,
            block, True)
        self.stage2 = self._make_stage(in_channels[1], n_blocks_per_stage,
            block, True)
        self.stage3 = self._make_stage(in_channels[2], n_blocks_per_stage,
            block, False)
        self.bn = nn.BatchNorm2d(in_channels[3])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, n_blocks, block, add_transition_block):
        stage = nn.Sequential()
        for index in range(n_blocks):
            stage.add_module(f'block{index + 1}', block(in_channels + index *
                self.growth_rate, self.growth_rate, self.drop_rate))
        if add_transition_block:
            in_channels = int(in_channels + n_blocks * self.growth_rate)
            out_channels = int(in_channels * self.compression_rate)
            stage.add_module('transition', TransitionBlock(in_channels,
                out_channels, self.drop_rate))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.bn1(x)
        y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = self.bn3(y)
        if y.size(1) != x.size(1):
            y += F.pad(self.shortcut(x), (0, 0, 0, 0, 0, y.size(1) - x.size
                (1)), 'constant', 0)
        else:
            y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        bottleneck_channels = out_channels // self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.bn1(x)
        y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)
        y = self.bn4(y)
        if y.size(1) != x.size(1):
            y += F.pad(self.shortcut(x), (0, 0, 0, 0, 0, y.size(1) - x.size
                (1)), 'constant', 0)
        else:
            y += self.shortcut(x)
        return y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.pyramidnet
        depth = model_config.depth
        initial_channels = model_config.initial_channels
        block_type = model_config.block_type
        alpha = model_config.alpha
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth
        n_channels = [initial_channels]
        for _ in range(n_blocks_per_stage * 3):
            num = n_channels[-1] + alpha / (n_blocks_per_stage * 3)
            n_channels.append(num)
        n_channels = [(int(np.round(c)) * block.expansion) for c in n_channels]
        n_channels[0] //= block.expansion
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        self.stage1 = self._make_stage(n_channels[:n_blocks_per_stage + 1],
            n_blocks_per_stage, block, stride=1)
        self.stage2 = self._make_stage(n_channels[n_blocks_per_stage:
            n_blocks_per_stage * 2 + 1], n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(n_channels[n_blocks_per_stage * 2:],
            n_blocks_per_stage, block, stride=2)
        self.bn2 = nn.BatchNorm2d(n_channels[-1])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, n_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, block(n_channels[index],
                    n_channels[index + 1], stride=stride))
            else:
                stage.add_module(block_name, block(n_channels[index],
                    n_channels[index + 1], stride=1))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn2(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        bottleneck_channels = out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.resnet
        depth = model_config.depth
        initial_channels = model_config.initial_channels
        block_type = model_config.block_type
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth
        n_channels = [initial_channels, initial_channels * 2 * block.
            expansion, initial_channels * 4 * block.expansion]
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(initial_channels)
        self.stage1 = self._make_stage(n_channels[0], n_channels[0],
            n_blocks_per_stage, block, stride=1)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, block, stride=2)
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, remove_first_relu,
        add_last_bn, preact=False):
        super().__init__()
        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        if add_last_bn:
            self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))

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

    def __init__(self, in_channels, out_channels, stride, remove_first_relu,
        add_last_bn, preact=False):
        super().__init__()
        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact
        bottleneck_channels = out_channels // self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        if add_last_bn:
            self.bn4 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))

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
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)
        if self._add_last_bn:
            y = self.bn4(y)
        y += self.shortcut(x)
        return y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.resnet_preact
        initial_channels = model_config.initial_channels
        self._remove_first_relu = model_config.remove_first_relu
        self._add_last_bn = model_config.add_last_bn
        block_type = model_config.block_type
        depth = model_config.depth
        preact_stage = model_config.preact_stage
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth
        n_channels = [initial_channels, initial_channels * 2 * block.
            expansion, initial_channels * 4 * block.expansion]
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(n_channels[0], n_channels[0],
            n_blocks_per_stage, block, stride=1, preact=preact_stage[0])
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, block, stride=2, preact=preact_stage[1])
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, block, stride=2, preact=preact_stage[2])
        self.bn = nn.BatchNorm2d(n_channels[2])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, block,
        stride, preact):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride, remove_first_relu=self.
                    _remove_first_relu, add_last_bn=self._add_last_bn,
                    preact=preact))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1, remove_first_relu=self.
                    _remove_first_relu, add_last_bn=self._add_last_bn,
                    preact=False))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, stage_index,
        base_channels, cardinality):
        super().__init__()
        bottleneck_channels = cardinality * base_channels * 2 ** stage_index
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.resnext
        depth = model_config.depth
        initial_channels = model_config.initial_channels
        self.base_channels = model_config.base_channels
        self.cardinality = model_config.cardinality
        n_blocks_per_stage = (depth - 2) // 9
        assert n_blocks_per_stage * 9 + 2 == depth
        block = BottleneckBlock
        n_channels = [initial_channels, initial_channels * block.expansion,
            initial_channels * 2 * block.expansion, initial_channels * 4 *
            block.expansion]
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.stage1 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, 0, stride=1)
        self.stage2 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, 1, stride=2)
        self.stage3 = self._make_stage(n_channels[2], n_channels[3],
            n_blocks_per_stage, 2, stride=2)
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, stage_index,
        stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, BottleneckBlock(in_channels,
                    out_channels, stride, stage_index, self.base_channels,
                    self.cardinality))
            else:
                stage.add_module(block_name, BottleneckBlock(out_channels,
                    out_channels, 1, stage_index, self.base_channels, self.
                    cardinality))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, remove_first_relu,
        add_last_bn, se_reduction, preact=False):
        super().__init__()
        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        if add_last_bn:
            self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels, se_reduction)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))

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
        y = self.se(y)
        y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, remove_first_relu,
        add_last_bn, se_reduction, preact=False):
        super().__init__()
        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact
        bottleneck_channels = out_channels // self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        if add_last_bn:
            self.bn4 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels, se_reduction)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))

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
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)
        if self._add_last_bn:
            y = self.bn4(y)
        y = self.se(y)
        y += self.shortcut(x)
        return y


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out',
            nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight.data)
        nn.init.zeros_(module.biasd.data)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out',
            nonlinearity='relu')
        nn.init.zeros_(module.biasd.data)


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        initial_channels = config.model.se_resnet_preact.initial_channels
        block_type = config.model.se_resnet_preact.block_type
        depth = config.model.se_resnet_preact.depth
        self._remove_first_relu = (config.model.se_resnet_preact.
            remove_first_relu)
        self._add_last_bn = config.model.se_resnet_preact.add_last_bn
        preact_stage = config.model.se_resnet_preact.preact_stage
        se_reduction = config.model.se_resnet_preact.se_reduction
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth
        n_channels = [initial_channels, initial_channels * 2 * block.
            expansion, initial_channels * 4 * block.expansion]
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(n_channels[0], n_channels[0],
            n_blocks_per_stage, block, stride=1, se_reduction=se_reduction,
            preact=preact_stage[0])
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, block, stride=2, se_reduction=se_reduction,
            preact=preact_stage[1])
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, block, stride=2, se_reduction=se_reduction,
            preact=preact_stage[2])
        self.bn = nn.BatchNorm2d(n_channels[2])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block,
        stride, se_reduction, preact):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride, remove_first_relu=self.
                    _remove_first_relu, add_last_bn=self._add_last_bn,
                    se_reduction=se_reduction, preact=preact))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1, remove_first_relu=self.
                    _remove_first_relu, add_last_bn=self._add_last_bn,
                    se_reduction=se_reduction, preact=False))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualPath(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.bn2(self.conv2(x))
        return x


class SkipConnection(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=
            1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=
            1, stride=1, padding=0, bias=False)
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


def get_alpha_beta(batch_size, shake_config, device):
    forward_shake, backward_shake, shake_image = shake_config
    if forward_shake and not shake_image:
        alpha = torch.rand(1)
    elif forward_shake and shake_image:
        alpha = torch.rand(batch_size).view(batch_size, 1, 1, 1)
    else:
        alpha = torch.FloatTensor([0.5])
    if backward_shake and not shake_image:
        beta = torch.rand(1)
    elif backward_shake and shake_image:
        beta = torch.rand(batch_size).view(batch_size, 1, 1, 1)
    else:
        beta = torch.FloatTensor([0.5])
    alpha = alpha.to(device)
    beta = beta.to(device)
    return alpha, beta


class ShakeFunction(Function):

    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        ctx.save_for_backward(x1, x2, alpha, beta)
        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, alpha, beta = ctx.saved_variables
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None
        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)
        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_function = ShakeFunction.apply


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shake_config):
        super().__init__()
        self.shake_config = shake_config
        self.residual_path1 = ResidualPath(in_channels, out_channels, stride)
        self.residual_path2 = ResidualPath(in_channels, out_channels, stride)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('skip', SkipConnection(in_channels,
                out_channels, stride))

    def forward(self, x):
        x1 = self.residual_path1(x)
        x2 = self.residual_path2(x)
        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = False, False, False
        alpha, beta = get_alpha_beta(x.size(0), shake_config, x.device)
        y = shake_function(x1, x2, alpha, beta)
        return self.shortcut(x) + y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.shake_shake
        depth = model_config.depth
        initial_channels = model_config.initial_channels
        self.shake_config = [model_config.shake_forward, model_config.
            shake_backward, model_config.shake_image]
        block = BasicBlock
        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth
        n_channels = [initial_channels, initial_channels * 2, 
            initial_channels * 4]
        self.conv = nn.Conv2d(config.dataset.n_channels, 16, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, n_channels[0],
            n_blocks_per_stage, block, stride=1)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, block, stride=2)
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride, shake_config=self.
                    shake_config))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1, shake_config=self.shake_config))
        return stage

    def _forward_conv(self, x):
        x = self.bn(self.conv(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.vgg
        self.use_bn = model_config.use_bn
        n_channels = model_config.n_channels
        n_layers = model_config.n_layers
        self.stage1 = self._make_stage(config.dataset.n_channels,
            n_channels[0], n_layers[0])
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_layers[1])
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_layers[2])
        self.stage4 = self._make_stage(n_channels[2], n_channels[3],
            n_layers[3])
        self.stage5 = self._make_stage(n_channels[3], n_channels[4],
            n_layers[4])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks):
        stage = nn.Sequential()
        for index in range(n_blocks):
            if index == 0:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=1, padding=1)
            else:
                conv = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                    stride=1, padding=1)
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
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self._preactivate_both = in_channels != out_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))

    def forward(self, x):
        if self._preactivate_both:
            x = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(x)
        else:
            y = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(y)
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.wrn
        depth = model_config.depth
        initial_channels = model_config.initial_channels
        widening_factor = model_config.widening_factor
        drop_rate = model_config.drop_rate
        block = BasicBlock
        n_blocks_per_stage = (depth - 4) // 6
        assert n_blocks_per_stage * 6 + 4 == depth
        n_channels = [initial_channels, initial_channels * widening_factor,
            initial_channels * 2 * widening_factor, initial_channels * 4 *
            widening_factor]
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, block, stride=1, drop_rate=drop_rate)
        self.stage2 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, block, stride=2, drop_rate=drop_rate)
        self.stage3 = self._make_stage(n_channels[2], n_channels[3],
            n_blocks_per_stage, block, stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(n_channels[3])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, block,
        stride, drop_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride, drop_rate=drop_rate))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1, drop_rate=drop_rate))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        return torch.cat([x, y], dim=1)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        bottleneck_channels = out_channels * 4
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training,
                inplace=False)
        return torch.cat([x, y], dim=1)


class TransitionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training,
                inplace=False)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.densenet
        block_type = model_config.block_type
        n_blocks = model_config.n_blocks
        self.growth_rate = model_config.growth_rate
        self.drop_rate = model_config.drop_rate
        self.compression_rate = model_config.compression_rate
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock
        in_channels = [2 * self.growth_rate]
        for index in range(4):
            denseblock_out_channels = int(in_channels[-1] + n_blocks[index] *
                self.growth_rate)
            if index < 3:
                transitionblock_out_channels = int(denseblock_out_channels *
                    self.compression_rate)
            else:
                transitionblock_out_channels = denseblock_out_channels
            in_channels.append(transitionblock_out_channels)
        self.conv = nn.Conv2d(config.dataset.n_channels, in_channels[0],
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(in_channels[0])
        self.stage1 = self._make_stage(in_channels[0], n_blocks[0], block, True
            )
        self.stage2 = self._make_stage(in_channels[1], n_blocks[1], block, True
            )
        self.stage3 = self._make_stage(in_channels[2], n_blocks[2], block, True
            )
        self.stage4 = self._make_stage(in_channels[3], n_blocks[3], block, 
            False)
        self.bn_last = nn.BatchNorm2d(in_channels[4])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, n_blocks, block, add_transition_block):
        stage = nn.Sequential()
        for index in range(n_blocks):
            stage.add_module(f'block{index + 1}', block(in_channels + index *
                self.growth_rate, self.growth_rate, self.drop_rate))
        if add_transition_block:
            in_channels = int(in_channels + n_blocks * self.growth_rate)
            out_channels = int(in_channels * self.compression_rate)
            stage.add_module('transition', TransitionBlock(in_channels,
                out_channels, self.drop_rate))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bn_last(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.bn1(x)
        y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = self.bn3(y)
        if y.size(1) != x.size(1):
            y += F.pad(self.shortcut(x), (0, 0, 0, 0, 0, y.size(1) - x.size
                (1)), 'constant', 0)
        else:
            y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        bottleneck_channels = out_channels // self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.bn1(x)
        y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)
        y = self.bn4(y)
        if y.size(1) != x.size(1):
            y += F.pad(self.shortcut(x), (0, 0, 0, 0, 0, y.size(1) - x.size
                (1)), 'constant', 0)
        else:
            y += self.shortcut(x)
        return y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.pyramidnet
        initial_channels = model_config.initial_channels
        block_type = model_config.block_type
        n_blocks = model_config.n_blocks
        alpha = model_config.alpha
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock
        n_channels = [initial_channels]
        depth = sum(n_blocks)
        rate = alpha / depth
        for _ in range(depth):
            num = n_channels[-1] + rate
            n_channels.append(num)
        n_channels = [(int(np.round(c)) * block.expansion) for c in n_channels]
        n_channels[0] //= block.expansion
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])
        accs = [n_blocks[0]]
        for i in range(1, 4):
            accs.append(accs[-1] + n_blocks[i])
        self.stage1 = self._make_stage(n_channels[:accs[0] + 1], n_blocks[0
            ], block, stride=1)
        self.stage2 = self._make_stage(n_channels[accs[0]:accs[1] + 1],
            n_blocks[1], block, stride=2)
        self.stage3 = self._make_stage(n_channels[accs[1]:accs[2] + 1],
            n_blocks[2], block, stride=2)
        self.stage4 = self._make_stage(n_channels[accs[2]:accs[3] + 1],
            n_blocks[3], block, stride=2)
        self.bn_last = nn.BatchNorm2d(n_channels[-1])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, n_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, block(n_channels[index],
                    n_channels[index + 1], stride=stride))
            else:
                stage.add_module(block_name, block(n_channels[index],
                    n_channels[index + 1], stride=1))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn_last(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        bottleneck_channels = out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.resnet
        initial_channels = model_config.initial_channels
        block_type = model_config.block_type
        n_blocks = model_config.n_blocks
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock
        n_channels = [initial_channels, initial_channels * 2 * block.
            expansion, initial_channels * 4 * block.expansion, 
            initial_channels * 8 * block.expansion]
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(initial_channels)
        self.stage1 = self._make_stage(n_channels[0], n_channels[0],
            n_blocks[0], block, stride=1)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks[1], block, stride=2)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks[2], block, stride=2)
        self.stage4 = self._make_stage(n_channels[2], n_channels[3],
            n_blocks[3], block, stride=2)
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, remove_first_relu,
        add_last_bn, preact=False):
        super().__init__()
        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        if add_last_bn:
            self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))

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

    def __init__(self, in_channels, out_channels, stride, remove_first_relu,
        add_last_bn, preact=False):
        super().__init__()
        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact
        bottleneck_channels = out_channels // self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        if add_last_bn:
            self.bn4 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))

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
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)
        if self._add_last_bn:
            y = self.bn4(y)
        y += self.shortcut(x)
        return y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.resnet_preact
        initial_channels = model_config.initial_channels
        self._remove_first_relu = model_config.remove_first_relu
        self._add_last_bn = model_config.add_last_bn
        block_type = model_config.block_type
        n_blocks = model_config.n_blocks
        preact_stage = model_config.preact_stage
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock
        n_channels = [initial_channels, initial_channels * 2 * block.
            expansion, initial_channels * 4 * block.expansion, 
            initial_channels * 8 * block.expansion]
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.stage1 = self._make_stage(n_channels[0], n_channels[0],
            n_blocks[0], block, stride=1, preact=preact_stage[0])
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks[1], block, stride=2, preact=preact_stage[1])
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks[2], block, stride=2, preact=preact_stage[2])
        self.stage4 = self._make_stage(n_channels[2], n_channels[3],
            n_blocks[3], block, stride=2, preact=preact_stage[3])
        self.bn_last = nn.BatchNorm2d(n_channels[3])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, block,
        stride, preact):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride, remove_first_relu=self.
                    _remove_first_relu, add_last_bn=self._add_last_bn,
                    preact=preact))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1, remove_first_relu=self.
                    _remove_first_relu, add_last_bn=self._add_last_bn,
                    preact=False))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn_last(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, stage_index,
        base_channels, cardinality):
        super().__init__()
        bottleneck_channels = cardinality * base_channels * 2 ** stage_index
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.resnext
        initial_channels = model_config.initial_channels
        n_blocks = model_config.n_blocks
        self.base_channels = model_config.base_channels
        self.cardinality = model_config.cardinality
        block = BottleneckBlock
        n_channels = [initial_channels, initial_channels * block.expansion,
            initial_channels * 2 * block.expansion, initial_channels * 4 *
            block.expansion, initial_channels * 8 * block.expansion]
        self.conv = nn.Conv2d(config.dataset.n_channels, n_channels[0],
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.stage1 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks[0], 0, stride=1)
        self.stage2 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks[1], 1, stride=2)
        self.stage3 = self._make_stage(n_channels[2], n_channels[3],
            n_blocks[2], 2, stride=2)
        self.stage4 = self._make_stage(n_channels[3], n_channels[4],
            n_blocks[3], 3, stride=2)
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, stage_index,
        stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(block_name, BottleneckBlock(in_channels,
                    out_channels, stride, stage_index, self.base_channels,
                    self.cardinality))
            else:
                stage.add_module(block_name, BottleneckBlock(out_channels,
                    out_channels, 1, stage_index, self.base_channels, self.
                    cardinality))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config.model.vgg
        self.use_bn = model_config.use_bn
        n_channels = model_config.n_channels
        n_layers = model_config.n_layers
        self.stage1 = self._make_stage(config.dataset.n_channels,
            n_channels[0], n_layers[0])
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_layers[1])
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_layers[2])
        self.stage4 = self._make_stage(n_channels[2], n_channels[3],
            n_layers[3])
        self.stage5 = self._make_stage(n_channels[3], n_channels[4],
            n_layers[4])
        with torch.no_grad():
            dummy_data = torch.zeros((1, config.dataset.n_channels, config.
                dataset.image_size, config.dataset.image_size), dtype=torch
                .float32)
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0
                ]
        self.fc1 = nn.Linear(self.feature_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, config.dataset.n_classes)
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks):
        stage = nn.Sequential()
        for index in range(n_blocks):
            if index == 0:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=1, padding=1)
            else:
                conv = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                    stride=1, padding=1)
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
        x = F.dropout(F.relu(self.fc1(x), inplace=True), training=self.training
            )
        x = F.dropout(F.relu(self.fc2(x), inplace=True), training=self.training
            )
        x = self.fc3(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hysts_pytorch_image_classification(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1, 'remove_first_relu': 4, 'add_last_bn': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BottleneckBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1, 'stage_index': 4, 'base_channels': 4, 'cardinality': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ResidualPath(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(SELayer(*[], **{'in_channels': 4, 'reduction': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(SkipConnection(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(TransitionBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'drop_rate': 0.5}), [torch.rand([4, 4, 4, 4])], {})

