import sys
_module = sys.modules[__name__]
del sys
epoch_based_runtime = _module
iter_based_runtime = _module
adam_100k_iter = _module
sgd_100epoch = _module
sgd_100k_iter = _module
sgd_200epoch = _module
base_coco = _module
base_voc = _module
few_shot_coco = _module
few_shot_voc = _module
default_runtime = _module
faster_rcnn_r50_caffe_c4 = _module
faster_rcnn_r50_caffe_fpn = _module
schedule = _module
fsce_r101_fpn = _module
fsce_r101_fpn_contrastive_loss = _module
fsdetview_r101_c4 = _module
fsdetview_r50_c4 = _module
mpsr_r101_fpn = _module
tfa_r101_fpn = _module
demo_attention_rpn_detector_inference = _module
demo_metric_classifier_1shot_inference = _module
conf = _module
stat = _module
mmfewshot = _module
classification = _module
apis = _module
inference = _module
test = _module
train = _module
core = _module
evaluation = _module
eval_hooks = _module
datasets = _module
base = _module
builder = _module
cub = _module
dataset_wrappers = _module
mini_imagenet = _module
pipelines = _module
loading = _module
tiered_imagenet = _module
utils = _module
models = _module
backbones = _module
conv4 = _module
resnet12 = _module
utils = _module
wrn = _module
classifiers = _module
base = _module
base_finetune = _module
base_metric = _module
baseline = _module
baseline_plus = _module
maml = _module
matching_net = _module
meta_baseline = _module
neg_margin = _module
proto_net = _module
relation_net = _module
heads = _module
base_head = _module
cosine_distance_head = _module
linear_head = _module
matching_head = _module
meta_baseline_head = _module
neg_margin_head = _module
prototype_head = _module
relation_head = _module
losses = _module
mse_loss = _module
nll_loss = _module
maml_module = _module
meta_test_parallel = _module
detection = _module
inference = _module
test = _module
train = _module
eval_hooks = _module
mean_ap = _module
custom_hook = _module
builder = _module
coco = _module
dataloader_wrappers = _module
formatting = _module
transforms = _module
voc = _module
resnet_with_meta_conv = _module
dense_heads = _module
attention_rpn_head = _module
two_branch_rpn_head = _module
detectors = _module
attention_rpn_detector = _module
fsce = _module
fsdetview = _module
meta_rcnn = _module
mpsr = _module
query_support_detector = _module
tfa = _module
supervised_contrastive_loss = _module
roi_heads = _module
bbox_heads = _module
contrastive_bbox_head = _module
cosine_sim_bbox_head = _module
meta_bbox_head = _module
multi_relation_bbox_head = _module
two_branch_bbox_head = _module
contrastive_roi_head = _module
fsdetview_roi_head = _module
meta_rcnn_roi_head = _module
multi_relation_roi_head = _module
shared_heads = _module
meta_rcnn_res_layer = _module
two_branch_roi_head = _module
aggregation_layer = _module
collate = _module
collect_env = _module
compat_config = _module
dist_utils = _module
infinite_sampler = _module
local_seed = _module
logger = _module
runner = _module
version = _module
setup = _module
test_classification_dataloader_builder = _module
test_classification_dataset_wrappers = _module
test_classification_datasets = _module
test_conv4 = _module
test_resnet12 = _module
test_wrn = _module
test_classification_losses = _module
test_classification_model_utils = _module
test_base_finetune = _module
test_base_metric = _module
test_maml = _module
test_cosine_distance_head = _module
test_linear_head = _module
test_matching_head = _module
test_meta_baseline_head = _module
test_negmargin_head = _module
test_prototype_head = _module
test_relation_head = _module
test_meta_test_eval_hook = _module
test_meta_test_parallel = _module
test_nway_kshot_dataloader = _module
test_two_branch_dataloader = _module
test_few_shot_base_dataset = _module
test_few_shot_coco_dataset = _module
test_few_shot_voc_dataset = _module
test_nway_kshot_dataset = _module
test_query_aware_dataset = _module
test_two_branch_dataset = _module
test_detection_formatting = _module
test_detection_transforms = _module
test_resnet_with_meta_conv = _module
test_attention_rpn_head = _module
test_two_branch_rpn_head = _module
test_detection_losses = _module
test_detection_model_frozen = _module
test_detection_utils = _module
test_attention_rpn_detector = _module
test_fine_tune_based_detector = _module
test_meta_rcnn_detector = _module
test_mpsr_detector = _module
test_detection_bbox_head = _module
test_detection_roi_heads = _module
test_detection_shared_head = _module
test_conpat_config = _module
test_samplers = _module
test = _module
train = _module
unzip_tiered_imagenet = _module
initialize_bbox_head = _module
visualize_saved_dataset = _module
test = _module
train = _module
print_config = _module

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


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import numpy as np


import torch


import torch.nn as nn


import copy


from torch import nn


from torch.utils.data import DataLoader


import warnings


from torch.utils.data import Dataset


from abc import ABCMeta


from abc import abstractmethod


from typing import Mapping


from typing import Sequence


from functools import partial


from torch import Tensor


import torch.nn.functional as F


from torch.distributions import Bernoulli


from collections import OrderedDict


import torch.distributed as dist


from torch.nn.utils.weight_norm import WeightNorm


import math


import time


from typing import Iterable


from torch.nn.modules.batchnorm import _BatchNorm


from torch.utils.data import Sampler


from typing import Iterator


from collections.abc import Mapping


from collections.abc import Sequence


from torch.utils.data.dataloader import default_collate


import itertools


from torch.utils.data.sampler import Sampler


import logging


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, is_pooling: bool=True, padding: int=1) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = [nn.Conv2d(in_channels, out_channels, 3, padding=padding), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        if is_pooling:
            layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) ->Tensor:
        out = self.layers(x)
        return out


class ConvNet(nn.Module):
    """Simple ConvNet.

    Args:
        depth (int): The number of `ConvBlock`.
        pooling_blocks (Sequence[int]): Indicate which block to use
            2x2 max pooling.
        padding_blocks (Sequence[int]): Indicate which block to use
            conv layer with padding.
        flatten (bool): Whether to flatten features from (N, C, H, W)
            to (N, C*H*W). Default: True.
    """

    def __init__(self, depth: int, pooling_blocks: Sequence[int], padding_blocks: Sequence[int], flatten: bool=True) ->None:
        super().__init__()
        layers = []
        for i in range(depth):
            in_channels = 3 if i == 0 else 64
            out_channels = 64
            layers.append(ConvBlock(in_channels, out_channels, is_pooling=i in pooling_blocks, padding=1 if i in padding_blocks else 0))
        self.flatten = flatten
        self.layers = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x: Tensor) ->Tensor:
        out = self.layers(x)
        if self.flatten:
            out = out.view(out.size(0), -1)
        return out

    def init_weights(self) ->None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Conv4(ConvNet):

    def __init__(self, depth: int=4, pooling_blocks: Sequence[int]=(0, 1, 2, 3), padding_blocks: Sequence[int]=(0, 1, 2, 3), flatten: bool=True) ->None:
        super().__init__(depth=depth, pooling_blocks=pooling_blocks, padding_blocks=padding_blocks, flatten=flatten)


class Conv4NoPool(ConvNet):
    """Used for RelationNet."""

    def __init__(self, depth: int=4, pooling_blocks: Sequence[int]=(0, 1), padding_blocks: Sequence[int]=(2, 3), flatten: bool=False) ->None:
        super().__init__(depth=depth, pooling_blocks=pooling_blocks, padding_blocks=padding_blocks, flatten=flatten)


class DropBlock(nn.Module):

    def __init__(self, block_size: int) ->None:
        super().__init__()
        self.block_size = block_size

    def forward(self, x: Tensor, gamma: float) ->Tensor:
        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)))
            mask = mask
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()
            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask: Tensor) ->Tensor:
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)
        non_zero_idxes = mask.nonzero()
        nr_blocks = non_zero_idxes.shape[0]
        offsets = torch.stack([torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), torch.arange(self.block_size).repeat(self.block_size)]).t()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).long(), offsets.long()), 1)
        offsets = offsets
        if nr_blocks > 0:
            non_zero_idxes = non_zero_idxes.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()
            block_idxes = non_zero_idxes + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxes[:, 0], block_idxes[:, 1], block_idxes[:, 2], block_idxes[:, 3]] = 1.0
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
        block_mask = 1 - padded_mask
        return block_mask


def conv3x3(in_channels: int, out_channels: int, stride: int=1) ->nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int=1, downsample: Optional[nn.Module]=None, drop_rate: float=0.0, drop_block: bool=False, block_size: int=1) ->None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.drop_rate = drop_rate
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x: Tensor) ->Tensor:
        num_batches_tracked = int(self.norm1.num_batches_tracked.cpu().data)
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.max_pool(out)
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
    """ResNet12.

    Args:
        block (nn.Module): Block to build layers. Default: :class:`BasicBlock`.
        with_avgpool (bool): Whether to average pool the features.
            Default: True.
        pool_size (tuple(int,int)): The output shape of average pooling layer.
            Default: (1, 1).
        flatten (bool): Whether to flatten features from (N, C, H, W)
            to (N, C*H*W). Default: True.
        drop_rate (float): Dropout rate. Default: 0.0.
        drop_block_size (int): Size of drop block. Default: 5.
    """

    def __init__(self, block: nn.Module=BasicBlock, with_avgpool: bool=True, pool_size: Tuple[int, int]=(1, 1), flatten: bool=True, drop_rate: float=0.0, drop_block_size: int=5) ->None:
        self.in_channels = 3
        super().__init__()
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=drop_block_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=drop_block_size)
        self.with_avgpool = with_avgpool
        if with_avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d(pool_size)
        self.flatten = flatten
        self.num_batches_tracked = 0

    def _make_layer(self, block: nn.Module, out_channels: int, stride: int=1, drop_rate: float=0.0, drop_block: bool=False, block_size: int=1) ->nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels * block.expansion))
        layers = [block(self.in_channels, out_channels, stride, downsample, drop_rate, drop_block, block_size)]
        self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def init_weights(self) ->None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) ->Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.with_avgpool:
            x = self.avgpool(x)
        if self.flatten:
            x = x.view(x.size(0), -1)
        return x


class WRNBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int, drop_rate: float=0.0) ->None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(p=drop_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True))

    def forward(self, x: Tensor) ->Tensor:
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        if self.drop_rate > 0.0:
            out = self.dropout(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """WideResNet.

    Args:
        depth (int): The number of layers.
        widen_factor (int): The widen factor of channels. Default: 1.
        stride (int): Stride of first layer. Default: 1.
        drop_rate (float): Dropout rate. Default: 0.0.
        with_avgpool (bool): Whether to average pool the features.
            Default: True.
        flatten (bool): Whether to flatten features from (N, C, H, W)
            to (N, C*H*W). Default: True.
        pool_size (tuple(int,int)): The output shape of average pooling layer.
            Default: (1, 1).
    """

    def __init__(self, depth: int, widen_factor: int=1, stride: int=1, drop_rate: float=0.0, flatten: bool=True, with_avgpool: bool=True, pool_size: Tuple[int, int]=(1, 1)) ->None:
        super().__init__()
        num_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        num_layers = (depth - 4) / 6
        block = WRNBlock
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(num_layers, num_channels[0], num_channels[1], block, stride, drop_rate)
        self.layer2 = self._make_layer(num_layers, num_channels[1], num_channels[2], block, 2, drop_rate)
        self.layer3 = self._make_layer(num_layers, num_channels[2], num_channels[3], block, 2, drop_rate)
        self.norm1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.num_channels = num_channels[3]
        self.flatten = flatten
        self.with_avgpool = with_avgpool
        if self.with_avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d(pool_size)

    @staticmethod
    def _make_layer(num_layers: Union[int, float], in_channels: int, out_channels: int, block: nn.Module, stride: int, drop_rate: float) ->nn.Sequential:
        layers = []
        for i in range(int(num_layers)):
            layers.append(block(in_channels if i == 0 else out_channels, out_channels, stride if i == 0 else 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) ->Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.with_avgpool:
            x = self.avgpool(x)
        if self.flatten:
            x = x.view(x.size(0), -1)
        return x

    def init_weights(self) ->None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class WRN28x10(WideResNet):

    def __init__(self, depth: int=28, widen_factor: int=10, stride: int=1, drop_rate: float=0.5, flatten: bool=True, with_avgpool: bool=True, pool_size: Tuple[int, int]=(1, 1)) ->None:
        super().__init__(depth=depth, widen_factor=widen_factor, stride=stride, drop_rate=drop_rate, flatten=flatten, with_avgpool=with_avgpool, pool_size=pool_size)


class LinearWithFastWeight(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool=True) ->None:
        super().__init__(in_features, out_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x: Tensor) ->Tensor:
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super().forward(x)
        return out


class Conv2dWithFastWeight(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple], stride: Union[int, Tuple]=1, padding: Union[int, Tuple, str]=0, bias: bool=True) ->None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x: Tensor) ->Tensor:
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super().forward(x)
        elif self.weight.fast is not None and self.bias.fast is not None:
            out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
        else:
            out = super().forward(x)
        return out


class BatchNorm2dWithFastWeight(nn.BatchNorm2d):

    def __init__(self, num_features: int) ->None:
        super().__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x: Tensor) ->Tensor:
        running_mean = torch.zeros(x.data.size()[1])
        running_var = torch.ones(x.data.size()[1])
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True, momentum=1)
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out


class MetaTestParallel(nn.Module):
    """The MetaTestParallel module that supports DataContainer.

    Note that each task is tested on a single GPU. Thus the data and model
    on different GPU should be independent. :obj:`MMDistributedDataParallel`
    always automatically synchronizes the grad in different GPUs when doing
    the loss backward, which can not meet the requirements. Thus we simply
    copy the module and wrap it with an :obj:`MetaTestParallel`, which will
    send data to the device model.

    MetaTestParallel has two main differences with PyTorch DataParallel:

        - It supports a custom type :class:`DataContainer` which allows
          more flexible control of input data during both GPU and CPU
          inference.
        - It implement three more APIs ``before_meta_test()``,
          ``before_forward_support()`` and ``before_forward_query()``.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """

    def __init__(self, module: nn.Module, dim: int=0) ->None:
        super().__init__()
        self.dim = dim
        self.module = module
        self.device = self.module.device
        if self.device == 'cpu':
            self.device_id = [-1]
        else:
            self.device_id = [self.module.get_device()]

    def forward(self, *inputs, **kwargs):
        """Override the original forward function.

        The main difference lies in the CPU inference where the data in
        :class:`DataContainers` will still be gathered.
        """
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = (),
            kwargs = {},
        return self.module(*inputs[0], **kwargs[0])

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def before_meta_test(self, *inputs, **kwargs) ->None:
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = (),
            kwargs = {},
        return self.module.before_meta_test(*inputs[0], **kwargs[0])

    def before_forward_support(self, *inputs, **kwargs) ->None:
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = (),
            kwargs = {},
        return self.module.before_forward_support(*inputs[0], **kwargs[0])

    def before_forward_query(self, *inputs, **kwargs) ->None:
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = (),
            kwargs = {},
        return self.module.before_forward_query(*inputs[0], **kwargs[0])


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.meta_test_cfg = None
        self.conv = nn.Conv2d(3, 3, 3)
        self.register_buffer('device_indicator', torch.empty(0))

    def forward(self, img, img_metas, test_mode=False, **kwargs):
        return img

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)

    def device(self) ->torch.device:
        return self.device_indicator.device

    def get_device(self):
        return self.device_indicator.get_device()

    def before_meta_test(self, meta_test_cfg, **kwargs):
        pass

    def before_forward_support(self, **kwargs):
        pass

    def before_forward_query(self, **kwargs):
        pass


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BatchNorm2dWithFastWeight,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2dWithFastWeight,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Conv4NoPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNet,
     lambda: ([], {'depth': 1, 'pooling_blocks': [4, 4], 'padding_blocks': [4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (DropBlock,
     lambda: ([], {'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ExampleModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearWithFastWeight,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNet12,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (WRN28x10,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (WRNBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_open_mmlab_mmfewshot(_paritybench_base):
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

