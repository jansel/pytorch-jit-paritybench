import sys
_module = sys.modules[__name__]
del sys
test_bert_emebddings = _module
test_MHSA = _module
test_add_extra_info = _module
test_qadd = _module
conf = _module
download = _module
main = _module
model = _module
main = _module
model = _module
dataset = _module
evaluate = _module
main = _module
models = _module
yolov3 = _module
yolov4 = _module
utils = _module
main = _module
evaluation = _module
main = _module
main = _module
model = _module
main = _module
utils = _module
main = _module
mmdet3d_plugin = _module
backbones = _module
vovnet = _module
vovnetcp = _module
detectors = _module
qbevdepth = _module
necks = _module
view_transformer_wo_dcnv2 = _module
analyze_logs = _module
benchmark = _module
get_flops = _module
vis = _module
create_data = _module
data_converter = _module
create_gt_database = _module
indoor_converter = _module
kitti_converter = _module
kitti_data_utils = _module
lyft_converter = _module
lyft_data_fixer = _module
nuimage_converter = _module
nuscenes_converter = _module
prepare_nuscenes_for_bevdet4d = _module
s3dis_data_utils = _module
scannet_data_utils = _module
sunrgbd_data_utils = _module
waymo_converter = _module
browse_dataset = _module
fuse_conv_bn = _module
print_config = _module
visualize_results = _module
convert_h3dnet_checkpoints = _module
convert_votenet_checkpoints = _module
publish_model = _module
regnet2mmdet = _module
qat_test = _module
qat_train = _module
test = _module
train = _module
vovnet = _module
vovnetcp = _module
qbevdet = _module
benchmark = _module
bevpool_unit_test = _module
get_flops = _module
create_gt_database = _module
fuse_conv_bn = _module
convert_h3dnet_checkpoints = _module
convert_votenet_checkpoints = _module
publish_model = _module
regnet2mmdet = _module
qat_test = _module
qat_train = _module
test = _module
train = _module
main = _module
model = _module
main = _module
model = _module
main = _module
model = _module
main = _module
model = _module
setup = _module
quantization = _module
common = _module
converters = _module
fuse_operations = _module
disable_unnecessary_quant = _module
fuse_bn = _module
lists = _module
prune = _module
simplifiers = _module
getattr_to_shape = _module
remove_identity = _module
unbind_getitem_to_subtensor = _module
bitpartite_graph_matching = _module
disjoint_set_union = _module
dominator_tree = _module
subgraph_matching = _module
subgraph_matching_node = _module
subgraph_matching_replace_pattern = _module
subgraph_matching_utils = _module
modules = _module
activations = _module
base = _module
conv = _module
embedding = _module
linear = _module
math = _module
matmul = _module
normalization = _module
pool = _module
python_builtins = _module
resize = _module
shape = _module
torchvision_ops = _module
unary = _module
observers = _module
aciq = _module
base = _module
kl_histogram = _module
minmax = _module
moving_average = _module
mse = _module
percentile = _module
quant_config = _module
quant_model = _module
quant_tracer = _module
quantizers = _module
base = _module
dorefa = _module
lsq = _module
lsq_plus = _module
pact = _module
quant_descriptor = _module
quant_tensor = _module
uniform = _module
tools = _module
calibration = _module
errors_profiler = _module
graph_wrapper = _module
tensor_wrapper = _module
utils = _module
sparse = _module
modules = _module
base = _module
conv = _module
linear = _module
normalization = _module
sparse_config = _module
sparse_model = _module
sparsers = _module
base = _module
l1norm = _module
common = _module
yaml_utils = _module

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


import random


import copy


import time


import numpy as np


import pandas as pd


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import torch.backends.cudnn as cudnn


from torch.utils.data import TensorDataset


from torch.utils.data import random_split


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


import warnings


from enum import Enum


import torch.nn.parallel


import torch.distributed as dist


import torch.optim


from torch.optim.lr_scheduler import StepLR


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torch.nn.functional as F


from types import SimpleNamespace


from torch.utils.data import Dataset


from torchvision import transforms as tv_transforms


from collections import OrderedDict


from torchvision.ops import boxes as box_ops


from torchvision.ops import nms


import torchvision.models as models


from torch.optim.lr_scheduler import CosineAnnealingLR


from collections import defaultdict


from collections import deque


from torch.nn.modules.batchnorm import _BatchNorm


import torch.utils.checkpoint as cp


from torch import nn as nn


import torch.fx as torch_fx


import collections


from functools import partial


from torch.optim import AdamW


from typing import Callable


from torch.nn.utils.fusion import fuse_conv_bn_eval


from torch.nn.utils.fusion import fuse_linear_bn_eval


import torch.fx as fx


from typing import Set


from typing import Dict


import torch.fx


from torchvision.ops.stochastic_depth import stochastic_depth


import math


from torch import nn


from torch.quantization.observer import _ObserverBase


from scipy import stats


from torch.fx import GraphModule


from torch.fx import Tracer


from typing import Any


import abc


from torch.nn import functional as F


import torch.distributed as ddp


from abc import ABC


class ConvAdd(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x_left = self.conv1(x)
        x_right = self.conv2(x)
        out = torch.add(x_left, x_right)
        return out


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.seq_length = 0

    def forward(self, input_ids, token_type_ids, past_key_values_length=0):
        position_ids = self.position_ids[:, past_key_values_length:self.seq_length + past_key_values_length]
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertForSequenceClassification(nn.Module):

    def __init__(self, traced_model, config=None):
        super(BertForSequenceClassification, self).__init__()
        self.bert = traced_model
        self.config = config
        self.num_labels = config.num_labels
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, zero_init_residual=False, groups=1, width_per_group=16, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(64, self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
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

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def conv_bn_lrelu(ni: int, nf: int, ks: int=3, stride: int=1) ->nn.Sequential:
    """Create a seuence Conv2d->BatchNorm2d->ReLu layer."""
    return nn.Sequential(OrderedDict([('conv', nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks // 2)), ('bn', nn.BatchNorm2d(nf)), ('relu', nn.LeakyReLU(negative_slope=0.1, inplace=True))]))


class ResLayer(nn.Module):
    """Resnet style layer with `ni` inputs."""

    def __init__(self, ni: int):
        super(ResLayer, self).__init__()
        self.layer1 = conv_bn_lrelu(ni, ni // 2, ks=1)
        self.layer2 = conv_bn_lrelu(ni // 2, ni, ks=3)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class Darknet(nn.Module):

    def make_group_layer(self, ch_in: int, num_blocks: int, stride: int=1):
        """starts with conv layer - `ch_in` channels in - then has `num_blocks` `ResLayer`"""
        return [conv_bn_lrelu(ch_in, ch_in * 2, stride=stride)] + [ResLayer(ch_in * 2) for i in range(num_blocks)]

    def __init__(self, depth=53, ch_in=3, nf=32):
        """
        depth (int): depth of darknet used in model, usually use [21, 53] for this param
        ch_in (int): input channels, for example, ch_in of RGB image is 3
        nf (int): number of filters output in stem.
        out_features (List[str]): desired output layer name.
        num_classes (int): For ImageNet, num_classes is 1000. If None, no linear layer will be
            added.
        """
        super(Darknet, self).__init__()
        self.stem = conv_bn_lrelu(ch_in, nf, ks=3, stride=1)
        current_stride = 1
        """create darknet with `nf` and `num_blocks` layers"""
        self.stages_and_names = []
        num_blocks = [1, 2, 8, 8, 4]
        for i, nb in enumerate(num_blocks):
            stage = nn.Sequential(*self.make_group_layer(nf, nb, stride=2))
            name = 'dark' + str(i + 1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            current_stride *= 2
            nf *= 2

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        outputs.append(x)
        for stage, _ in self.stages_and_names:
            x = stage(x)
            outputs.append(x)
        return outputs[-3], outputs[-2], outputs[-1]


class YOLOv3(nn.Module):

    def __init__(self, pretrain_path=None, num_classes=80, num_anchors=3):
        super(YOLOv3, self).__init__()
        self.backbone = Darknet()
        if pretrain_path:
            None
            state_dict = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)
        in_channels = [256, 512, 1024]
        out_filter_0 = (1 + 4 + num_classes) * num_anchors
        self.out0 = self._make_embedding(in_channels[-1], [512, 1024], out_filter_0)
        self.out1_cbl = conv_bn_lrelu(512, 256, 1)
        self.out1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        out_filter_1 = (1 + 4 + num_classes) * num_anchors
        self.out1 = self._make_embedding(in_channels[-2] + 256, [256, 512], out_filter_1)
        self.out2_cbl = conv_bn_lrelu(256, 128, 1)
        self.out2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        out_filter_2 = (1 + 4 + num_classes) * num_anchors
        self.out2 = self._make_embedding(in_channels[-3] + 128, [128, 256], out_filter_2)

    def _make_embedding(self, in_filters, filters_list, out_filter):
        m = nn.ModuleList([conv_bn_lrelu(in_filters, filters_list[0], 1), conv_bn_lrelu(filters_list[0], filters_list[1], 3), conv_bn_lrelu(filters_list[1], filters_list[0], 1), conv_bn_lrelu(filters_list[0], filters_list[1], 3), conv_bn_lrelu(filters_list[1], filters_list[0], 1), conv_bn_lrelu(filters_list[0], filters_list[1], 3)])
        m.add_module('conv_out', nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True))
        return m

    def forward(self, x):

        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        x2, x1, x0 = self.backbone(x)
        out0, out0_branch = _branch(self.out0, x0)
        x1_in = self.out1_cbl(out0_branch)
        x1_in = self.out1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.out1, x1_in)
        x2_in = self.out2_cbl(out1_branch)
        x2_in = self.out2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.out2, x2_in)
        outputs = [out0, out1, out2]
        return outputs


class Conv_Bn_Activation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=bias))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == 'mish':
            self.conv.append(nn.Mish())
        elif activation == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == 'leaky':
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'linear':
            pass
        else:
            raise NotImplementedError

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')
        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = x6 + x4
        x7 = self.conv7(x6)
        x7 = torch.cat([x7, x3], 1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        self.resblock = ResBlock(ch=64, nblocks=2)
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], 1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], 1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], 1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], 1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], 1)
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        up = self.upsample1(x7)
        x8 = self.conv8(downsample4)
        x8 = torch.cat([x8, up], 1)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        up = self.upsample2(x14)
        x15 = self.conv15(downsample3)
        x15 = torch.cat([x15, up], 1)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):

    def __init__(self, output_ch, n_classes):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)
        x3 = self.conv3(input1)
        x3 = torch.cat([x3, input2], 1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x8)
        x11 = torch.cat([x11, input3], 1)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        return x18, x10, x2


class Yolov4(nn.Module):

    def __init__(self, n_classes=80):
        super().__init__()
        output_ch = (4 + 1 + n_classes) * 3
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        self.neck = Neck()
        self.head = Yolov4Head(output_ch, n_classes)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        x20, x13, x6 = self.neck(d5, d4, d3)
        output = self.head(x20, x13, x6)
        return output


class eSEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


def dw_conv3x3(in_channels, out_channels, module_name, postfix, stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [('{}_{}/dw_conv3x3'.format(module_name, postfix), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=out_channels, bias=False)), ('{}_{}/pw_conv1x1'.format(module_name, postfix), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)), ('{}_{}/pw_norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)), ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True))]


class _OSA_module(nn.Module):

    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False, with_cp=True):
        super(_OSA_module, self).__init__()
        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.use_checkpoint = with_cp
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(OrderedDict(conv1x1(in_channel, stage_ch, '{}_reduction'.format(module_name), '0')))
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))
        self.ese = eSEModule(concat_ch)

    def _forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)
        if self.identity:
            xt = xt + identity_feat
        return xt

    def forward(self, x):
        if self.use_checkpoint and self.training:
            xt = cp.checkpoint(self._forward, x)
        else:
            xt = self._forward(x)
        return xt


class _OSA_stage(nn.Sequential):

    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False, depthwise=False):
        super(_OSA_stage, self).__init__()
        if not stage_num == 2:
            self.add_module('Pooling', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        if block_per_stage != 1:
            SE = False
        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:
                SE = False
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name, _OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=True, depthwise=depthwise))


class SELikeModule(nn.Module):

    def __init__(self, in_channel=512, feat_channel=256, intrinsic_channel=33):
        super(SELikeModule, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, feat_channel, kernel_size=1, padding=0)
        self.fc = nn.Sequential(nn.BatchNorm1d(intrinsic_channel), nn.Linear(intrinsic_channel, feat_channel), nn.Sigmoid())

    def forward(self, x, cam_params):
        x = self.input_conv(x)
        b, c, _, _ = x.shape
        y = self.fc(cam_params).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ViewTransformerLSSBEVDepthForward(nn.Module):

    def __init__(self, model):
        super(ViewTransformerLSSBEVDepthForward, self).__init__()
        self.model = model

    def forward(self, input, img_feat, depth_digit):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, C, H, W = x.shape
        depth_prob = self.model.get_depth_dist(depth_digit)
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.model.numC_Trans, self.model.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)
        if self.model.accelerate:
            bev_feat = self.model.voxel_pooling_accelerated(rots, trans, intrins, post_rots, post_trans, volume)
        else:
            geom = self.model.get_geometry(rots, trans, intrins, post_rots, post_trans)
            bev_feat = self.model.voxel_pooling(geom, volume)
        return bev_feat


class BEVDepthTraced(nn.Module):

    def __init__(self, model):
        super(BEVDepthTraced, self).__init__()
        _model = copy.deepcopy(model)
        self.img_backbone = _model.img_backbone
        self.img_neck = _model.img_neck
        self.img_view_transformer_featnet = _model.img_view_transformer.featnet
        self.se = SELikeModule(_model.img_view_transformer.se)
        self.extra_depthnet = _model.img_view_transformer.extra_depthnet
        self.dcn = _model.img_view_transformer.dcn
        self.depthnet = _model.img_view_transformer.depthnet
        _model.img_view_transformer.featnet = nn.Identity()
        _model.img_view_transformer.se = nn.Identity()
        _model.img_view_transformer.extra_depthnet = nn.Identity()
        _model.img_view_transformer.dcn = nn.Identity()
        _model.img_view_transformer.depthnet = nn.Identity()
        self.img_view_transformer = ViewTransformerLSSBEVDepthForward(_model.img_view_transformer)
        self.bev_encoder_backbone = _model.img_bev_encoder_backbone
        self.bev_encoder_neck = _model.img_bev_encoder_neck
        self.head = _model.pts_bbox_head
        self.head_shared_conv = _model.pts_bbox_head.shared_conv
        self.head_task_heads = _model.pts_bbox_head.task_heads
        self.img_view_transformer_featnet_quant = nn.Identity()
        self.img_view_transformer_featnet_quant.remove = False
        self.img_view_transformer_depth_quant = nn.Identity()
        self.img_view_transformer_depth_quant.remove = False
        self.loss = _model.pts_bbox_head.loss
        self.get_bboxes = _model.pts_bbox_head.get_bboxes

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        x = self.img_neck(x)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def image_view_transformer_encoder(self, input):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, oldC, H, W = x.shape
        x = x.view(B * N, oldC, H, W)
        img_feat = self.img_view_transformer_featnet(x)
        img_feat = self.img_view_transformer_featnet_quant(img_feat)
        depth_feat = x
        cam_params = torch.cat([intrins.reshape(B * N, -1), post_rots.reshape(B * N, -1), post_trans.reshape(B * N, -1), rots.reshape(B * N, -1), trans.reshape(B * N, -1)], dim=1)
        depth_feat = self.se(depth_feat, cam_params)
        depth_feat = self.extra_depthnet(depth_feat)[0]
        depth_feat = self.dcn(depth_feat)
        depth_digit = self.depthnet(depth_feat)
        depth_digit = self.img_view_transformer_depth_quant(depth_digit)
        return img_feat, depth_digit

    def bev_encoder(self, x):
        x = self.bev_encoder_backbone(x)
        x = self.bev_encoder_neck(x)
        return x

    def forward_single(self, x):
        x = self.head_shared_conv(x)
        task_heads = self.head_task_heads
        ret_dicts = []
        ret_dicts.append(task_heads[0](x))
        ret_dicts.append(task_heads[1](x))
        ret_dicts.append(task_heads[2](x))
        ret_dicts.append(task_heads[3](x))
        ret_dicts.append(task_heads[4](x))
        ret_dicts.append(task_heads[5](x))
        return ret_dicts

    def forward_pts_head(self, feats):
        return multi_apply(self.forward_single, feats)

    def enable_traced(self):
        self.traced = True

    def forward(self, img_inputs, img_metas=None, rescale=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        img, rots, trans, intrins, post_rots, post_trans, depth_gt = img_inputs
        x = self.image_encoder(img)
        img_feat, depth_digit = self.image_view_transformer_encoder([x, rots, trans, intrins, post_rots, post_trans])
        x = self.img_view_transformer([x, rots, trans, intrins, post_rots, post_trans], img_feat, depth_digit)
        x = self.bev_encoder(x)
        x = self.forward_pts_head([x])
        return x


class BEVDetTraced(nn.Module):

    def __init__(self, model):
        super(BEVDetTraced, self).__init__()
        _model = copy.deepcopy(model)
        self.img_backbone = _model.img_backbone
        self.img_neck = _model.img_neck
        self.img_view_transformer_depthnet = _model.img_view_transformer.depthnet
        _model.img_view_transformer.depthnet = nn.Identity()
        self.img_view_transformer = _model.img_view_transformer
        self.bev_encoder_backbone = _model.img_bev_encoder_backbone
        self.bev_encoder_neck = _model.img_bev_encoder_neck
        self.head = _model.pts_bbox_head
        self.head_shared_conv = _model.pts_bbox_head.shared_conv
        self.head_task_heads = _model.pts_bbox_head.task_heads
        self.img_view_transformer_quant = nn.Identity()
        self.img_view_transformer_quant.remove = False
        self.loss = _model.pts_bbox_head.loss
        self.get_bboxes = _model.pts_bbox_head.get_bboxes

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        x = self.img_neck(x)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def image_view_transformer_encoder(self, x):
        B, num_cams, oldC, H, W = x.shape
        x = x.view(B * num_cams, oldC, H, W)
        x = self.img_view_transformer_depthnet(x)
        x = self.img_view_transformer_quant(x)
        x = x.view(B, num_cams, -1, H, W)
        return x

    def bev_encoder(self, x):
        x = self.bev_encoder_backbone(x)
        x = self.bev_encoder_neck(x)
        return x

    def forward_single(self, x):
        x = self.head_shared_conv(x)
        task_heads = self.head_task_heads
        ret_dicts = []
        ret_dicts.append(task_heads[0](x))
        ret_dicts.append(task_heads[1](x))
        ret_dicts.append(task_heads[2](x))
        ret_dicts.append(task_heads[3](x))
        ret_dicts.append(task_heads[4](x))
        ret_dicts.append(task_heads[5](x))
        return ret_dicts

    def forward_pts_head(self, feats):
        return multi_apply(self.forward_single, feats)

    def enable_traced(self):
        self.traced = True

    def forward(self, img_inputs, img_metas, rescale=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        img, rots, trans, intrins, post_rots, post_trans = img_inputs
        x = self.image_encoder(img)
        x = self.image_view_transformer_encoder(x)
        x = self.img_view_transformer([x, rots, trans, intrins, post_rots, post_trans])
        x = self.bev_encoder(x)
        x = self.forward_pts_head([x])
        return x


class QuantTarget(Enum):
    WEIGHT = 0
    FEATURE = 1


class Backend(Enum):
    VIRTUAL = 0
    ONNXRUNTIME = 1
    TENSORRT = 2


def get_backend(backend):
    target_backend = ['onnxruntime', 'tensorrt', 'virtual']
    if backend == 'virtual':
        return Backend.VIRTUAL
    if backend == 'onnxruntime':
        return Backend.ONNXRUNTIME
    if backend == 'tensorrt':
        return Backend.TENSORRT
    raise TypeError('only support backend in {}, not {}'.format(target_backend, backend))


def update_config(config, key, value):
    config.defrost()
    keys = key.split('.')

    def _set_config_attr(cfg, keys, value):
        if len(keys) > 1:
            cfg = getattr(cfg, keys[0].upper())
            _set_config_attr(cfg, keys[1:], value)
        else:
            setattr(cfg, keys[0].upper(), value)
    _set_config_attr(config, keys, value)
    config.freeze()
    return config


class QuantOpr(nn.Module):
    """QuantOpr是torch算子的量化版本。
    它提供可配置的 ``input_quantizer`` 和 ``weight_quantizer`` ,
    可根据需要启用。启用后,将转出QDQ格式的onnx模型,便于tensorRT运行。

    Attributes:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        weight_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            参数量化器。仅在该算子存在 ``weight`` 属性时可启用。
    """

    def __init__(self):
        super(QuantOpr, self).__init__()
        self.weight = None
        self.input_quantizer = None
        self.weight_quantizer = None

    def forward(self, x_in: torch.Tensor):
        """在考虑量化前提下，描述算子前向传播。

        基类不包含算子的实现。请在子类中添加。
        """
        raise NotImplementedError('no found a forward in {}'.format(self.__class__.__name__))

    def build_quantizer(self, config):
        """根据config配置 ``input_quantizer`` 和 ``weight_quantizer`` 。"""
        _backend = get_backend(config.BACKEND)
        if self.weight is not None:
            update_config(config.W, 'TARGET', (QuantTarget.WEIGHT,))
            self.weight_quantizer = build_quantizer(cfg=config.W)
            self.weight_quantizer.set_backend(_backend)
        update_config(config.A, 'TARGET', (QuantTarget.FEATURE,))
        self.input_quantizer = build_quantizer(cfg=config.A)
        self.input_quantizer.set_backend(_backend)

    def set_quant(self, w_quant: bool=False, a_quant: bool=False):
        """开关本算子的 ``input_quantizer`` 和 ``weight_quantizer`` 。

        .. Note::

            注意 ``input_quantizer`` 和 ``weight_quantizer`` 同时被设置。
            如果只设置其中一个,另一个将被默认设置为关闭。
        """
        if self.weight_quantizer:
            if w_quant and not self.weight_quantizer.fake_fused:
                self.weight_quantizer.enable_quant()
            else:
                self.weight_quantizer.disable_quant()
        if self.input_quantizer:
            if a_quant and not self.input_quantizer.fake_fused:
                self.input_quantizer.enable_quant()
            else:
                self.input_quantizer.disable_quant()

    def __repr__(self):
        info = self._repr_info
        if self.weight_quantizer and self.weight_quantizer.is_enable:
            info += '\n\tweight_quantizer: {}'.format(self.weight_quantizer.__repr__())
        if self.input_quantizer and self.input_quantizer.is_enable:
            info += '\n\tinput_quantizer: {}'.format(self.input_quantizer.__repr__())
        return info


class MultipleInputsQuantOpr(nn.Module):
    """MultipleInputsQuantOpr是torch算子的多输入量化版本。
    它不会提供 ``input_quantizer`` 和 ``weight_quantizer`` ,
    而是在build_quantizer时对每个输入插入一个独立 ``QIdentity`` 算子，在算子中包含 ``input_quantizer`` 。
    请注意本算子自身不做量化。
    """

    def __init__(self):
        super(MultipleInputsQuantOpr, self).__init__()
        self.input_quantizer_generated = False
        self.apply_input_quant = True

    def prepare_input_quantizer(self, node, model):
        if self.input_quantizer_generated:
            return
        input_nodes_cache = list(node.all_input_nodes)
        for idx, input_node in enumerate(input_nodes_cache):
            new_module_name = node.name + '_identity{}'.format(idx)
            new_module = QIdentity()
            model.add_module(new_module_name, new_module)
            with model.graph.inserting_before(node):
                identity_node = model.graph.create_node(op='call_module', target=new_module_name, args=(input_node,), kwargs={}, name=new_module_name)
            node.replace_input_with(input_node, identity_node)
        self.input_quantizer_generated = True


QMODULE_MAP = {}


def register_qmodule(sources: [nn.Module, str, ...]):

    def real_register(qmodule):
        for src in sources:
            QMODULE_MAP[src] = qmodule
        return qmodule
    return real_register


class QConv2d(QuantOpr):
    """量化卷积层,拥有 ``input_quantizer`` 和 ``weight_quantizer`` 。

    是QuantOpr的子类。

    Attributes:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        weight_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            参数量化器。
        fwd_kwargs (Dict[str, any]): 运行 ``torch.nn.Conv2d`` forward需要的参数。
        weight (torch.nn.Parameter): 卷积层的weight,引用自原Module。
        bias (torch.nn.Parameter): 卷积层的bias,引用自原Module。
    """

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Conv2d)
        super().__init__()
        self.cfg = config
        self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding, dilation=org_module.dilation, groups=org_module.groups)
        self.weight = org_module.weight
        self.bias = org_module.bias
        self._repr_info = 'Q' + org_module.__repr__()

    def forward(self, x_in: torch.Tensor):
        """卷积层的前向传播,但加入了input和weight量化。"""
        x_in = self.input_quantizer(x_in)
        weight = self.weight_quantizer(self.weight)
        out = F.conv2d(x_in, weight, self.bias, **self.fwd_kwargs)
        return out


class QConvTranspose2d(QuantOpr):

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.ConvTranspose2d)
        super().__init__()
        self.cfg = config
        self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding, output_padding=org_module.output_padding, dilation=org_module.dilation, groups=org_module.groups)
        self.weight = org_module.weight
        self.bias = org_module.bias
        self._repr_info = 'Q' + org_module.__repr__()

    def forward(self, x_in: torch.Tensor):
        """卷积层的前向传播,但加入了input和weight量化。"""
        x_in = self.input_quantizer(x_in)
        weight = self.weight_quantizer(self.weight)
        out = F.conv_transpose2d(x_in, weight, self.bias, **self.fwd_kwargs)
        return out


class QEmbedding(QuantOpr):
    """量化嵌入层, 仅有 ``weight_quantizer``, 默认由于输入是index值, 即不量化输入.

    是QuantOpr的子类。

    Attributes:
        weight_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
        weight (torch.nn.Parameter): embedding的weight,引用自原Module。
    """

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Embedding)
        super().__init__()
        self.cfg = config
        self.weight = org_module.weight
        self.padding_idx = org_module.padding_idx
        self.max_norm = org_module.max_norm
        self.norm_type = org_module.norm_type
        self.scale_grad_by_freq = org_module.scale_grad_by_freq
        self.sparse = org_module.sparse
        self._repr_info = 'Q' + org_module.__repr__()

    def build_quantizer(self, config):
        QuantOpr.build_quantizer(self, config)
        self.input_quantizer.set_fake_fused()

    def forward(self, x_in, *args, **kwargs):
        weight = self.weight_quantizer(self.weight)
        return F.embedding(x_in, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class QLinear(QuantOpr):
    """量化全连接层,拥有 ``input_quantizer`` 和 ``weight_quantizer`` 。

    是QuantOpr的子类。

    Attributes:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        weight_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            参数量化器。
        weight (torch.nn.Parameter): 卷积层的weight,引用自原Module。
        bias (torch.nn.Parameter): 卷积层的bias,引用自原Module。
    """

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Linear)
        super().__init__()
        self.cfg = config
        self.weight = org_module.weight
        self.bias = org_module.bias
        self._repr_info = 'Q' + org_module.__repr__()

    def forward(self, x_in: torch.Tensor):
        """全连接层的前向传播,但加入了input和weight量化。"""
        x_in = self.input_quantizer(x_in)
        weight = self.weight_quantizer(self.weight)
        out = F.linear(x_in, weight, self.bias)
        return out


class QAdd(MultipleInputsQuantOpr):

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = 'QAdd'
        self.apply_input_quant = config.A.QADD.ENABLE_QUANT

    def prepare_input_quantizer(self, node, model):
        if not self.apply_input_quant:
            return
        super(QAdd, self).prepare_input_quantizer(node, model)

    def forward(self, x_left, x_right):
        out = x_left + x_right
        return out


class QSubtract(nn.Module):

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = 'QSubtract '

    def forward(self, x_left, x_right):
        out = x_left - x_right
        return out


class QMul(nn.Module):

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = 'QMul'

    def forward(self, x_left, x_right):
        out = x_left * x_right
        return out


class QDivide(nn.Module):

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = 'QDivide '

    def forward(self, x_left, x_right):
        out = x_left / x_right
        return out


class QFloorDiv(nn.Module):

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = 'QFloorDiv '

    def forward(self, x_left, x_right):
        out = x_left // x_right
        return out


class QMean(QuantOpr):

    def __init__(self, org_module=None, config=None):
        super(QMean, self).__init__()
        self._repr_info = 'QMean'
        if isinstance(org_module, torch.fx.Node):
            self.dim = org_module.args[1]
            self.keepdim = org_module.kwargs['keepdim']
        else:
            raise NotImplementedError

    def forward(self, x_in, *args, **kwargs):
        x_in = self.input_quantizer(x_in)
        out = torch.mean(x_in, dim=self.dim, keepdim=self.keepdim)
        return out


class MatMul(MultipleInputsQuantOpr):
    """量化矩阵乘法，但算子本身不包含量化 。

    量化输入在build_quantizer中处理, 通过在输入上增加QIdentity层来解决。
    """

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = 'QMatmul '

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        out = torch.matmul(x_left, x_right)
        return out


class QBatchNorm2d(QuantOpr):
    """量化BN层。由于默认为BN会跟在conv/linear前或后, 所以可以被fused, 故不执行量化。

    是QuantOpr的子类。
    """

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self.module = org_module
        self._repr_info = 'QBatchNorm2d '

    def forward(self, x_in):
        """BN层的前向传播,不做量化。"""
        out = self.module(x_in)
        return out


class QBatchNorm1d(nn.Module):
    """未量化的BN1d层。"""

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self.module = org_module
        self._repr_info = 'QBatchNorm1d '

    def forward(self, x_in):
        """BN层的前向传播,不做量化。"""
        out = self.module(x_in)
        return out


class QLayerNorm(QuantOpr):

    def __init__(self, org_module=None, config=None):
        super(QLayerNorm, self).__init__()
        self.module = org_module
        self._repr_info = 'QLayerNorm '

    def forward(self, x_in):
        x_in = self.input_quantizer(x_in)
        out = self.module(x_in)
        return out


class MaxPool2d(nn.Module):
    """MaxPool层。认为maxpool不改变运算前后值域范围,所以不做量化。
    Attributes:
        fwd_kwargs (Dict[str, any]): 运行 ``torch.nn.functional.max_pool2d`` 需要的参数。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        self.fwd_kwargs = dict(kernel_size=org_module.kernel_size, stride=org_module.stride, padding=org_module.padding, dilation=org_module.dilation, ceil_mode=org_module.ceil_mode)

    def forward(self, x_in):
        """MaxPool层的前向传播,不做量化。"""
        return F.max_pool2d(x_in, **self.fwd_kwargs)


class QAdaptiveAvgPool2d(QuantOpr):
    """量化AvgPool层,拥有 ``input_quantizer`` 。

    是QuantOpr的子类。

    Attributes:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        output_size (any):
            同 ``torch.nn.AdaptiveAvgPool2d`` 。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        if isinstance(org_module, nn.Module):
            self.output_size = org_module.output_size
        else:
            self.output_size = org_module.args[1]
        self._repr_info = 'Q' + org_module.__repr__()

    def forward(self, x_in, *args):
        """AvgPool层的前向传播,但加入了input量化。"""
        x_in = self.input_quantizer(x_in)
        out = F.adaptive_avg_pool2d(x_in, self.output_size)
        return out


class QGetAttr(nn.Module):

    def __init__(self, org_module=None, config=None):
        super(QGetAttr, self).__init__()
        assert isinstance(org_module, torch.fx.Node)
        self.target_attr = org_module.args[1]
        if self.target_attr != 'shape':
            self.output = getattr(org_module.args[0], org_module.args[1])
        self._repr_info = 'QGetAttr '

    def forward(self, x_in, *args):
        if self.target_attr == 'shape':
            return x_in.shape()
        else:
            return self.output


class QGetItem(nn.Module):

    def __init__(self, org_module=None, config=None):
        super(QGetItem, self).__init__()
        self.target_item = org_module.args[1]
        self._repr_info = 'QGetItem '

    def forward(self, x_in, *args):
        return x_in[self.target_item]


class QEqual(nn.Module):

    def __init__(self, org_module=None, config=None):
        super(QEqual, self).__init__()
        self._repr_info = 'QEqual '

    def forward(self, x_left, x_right):
        return x_left == x_right


class QUpsample(QuantOpr):

    def __init__(self, org_module=None, config=None):
        super(QUpsample, self).__init__()
        self.scale_factor = org_module.scale_factor
        self.mode = org_module.mode
        self._repr_info = 'QUpsample, mode: {} '.format(self.mode)

    def build_quantizer(self, config):
        """
        force the bit of resize oprs is 8bit
        """
        QuantOpr.build_quantizer(self, config)
        if self.mode == 'nearest':
            self.input_quantizer.set_fake_fused()
        else:
            self.input_quantizer.set_bit(bit=8)

    def forward(self, x_in, *args):
        x_in = self.input_quantizer(x_in)
        out = F.interpolate(x_in, scale_factor=self.scale_factor, mode=self.mode)
        return out


class QInterpolate(QuantOpr):

    def __init__(self, org_module=None, config=None):
        super(QInterpolate, self).__init__()
        if isinstance(org_module, nn.Module):
            raise NotImplementedError
        else:
            self.mode = org_module.kwargs['mode']
        self._repr_info = 'QInterpolate, mode: {} '.format(self.mode)

    def build_quantizer(self, config):
        """
        force the bit of resize oprs is 8bit
        """
        QuantOpr.build_quantizer(self, config)
        if self.mode == 'nearest':
            self.input_quantizer.set_fake_fused()
        else:
            self.input_quantizer.set_bit(bit=8)

    def forward(self, x_in, *args, **kwargs):
        x_in = self.input_quantizer(x_in)
        out = F.interpolate(x_in, *args, **kwargs)
        return out


class Flatten(nn.Module):
    """量化Flatten层。认为flatten不改变运算前后值域范围,所以不做量化。

    是QuantOpr的子类。

    Attributes:
        start_dim (any): 同 ``torch.nn.Flatten`` 。
        end_dim (any): 同 ``torch.nn.Flatten`` 。
    """

    def __init__(self, org_module=None, config=None):
        super(Flatten, self).__init__()
        if isinstance(org_module, torch.fx.Node):
            start_dim = org_module.args[1]
            end_dim = org_module.args[2] if len(org_module.args) == 3 else -1
            self.start_dim = start_dim
            self.end_dim = end_dim
        else:
            self.start_dim = org_module.start_dim
            self.end_dim = org_module.end_dim

    def forward(self, x_in, *args):
        """Flatten层的前向传播,不做量化。"""
        out = torch.flatten(x_in, self.start_dim, self.end_dim)
        return out


class Size(nn.Module):

    def __init__(self, org_module=None, config=None):
        super(Size, self).__init__()
        if isinstance(org_module, torch.fx.Node):
            if 'dim' in org_module.kwargs:
                self.dim = org_module.kwargs['dim']
            elif len(org_module.args) == 2:
                self.dim = org_module.args[1]
            else:
                self.dim = None
        else:
            self.dim = None

    def forward(self, x, *args, **kwargs):
        out = x.size(*args, **kwargs)
        return out


class Reshape(nn.Module):

    def __init__(self, org_module=None, config=None):
        super(Reshape, self).__init__()

    def forward(self, x_in, *args):
        if isinstance(args, tuple) and len(args) != 1:
            args = args,
        return torch.reshape(x_in, *args)


class Concat(nn.Module):

    def __init__(self, org_module=None, config=None):
        super().__init__()
        if 'dim' in org_module.kwargs:
            self.dim = org_module.kwargs['dim']
        elif len(org_module.args) == 2:
            self.dim = org_module.args[1]
        else:
            self.dim = None
        self._repr_info = 'Concat'

    def forward(self, x_in, *args, **kwargs):
        out = torch.cat(x_in, dim=self.dim)
        return out


class Expand(nn.Module):

    def __init__(self, org_module=None, config=None):
        super(Expand, self).__init__()
        self.sizes = []
        self.input_places = []
        for idx, sz in enumerate(org_module.args[1:]):
            if isinstance(sz, torch.fx.Node):
                self.sizes.append(None)
                self.input_places.append(idx)
            else:
                assert isinstance(sz, int)
                self.sizes.append(sz)

    def forward(self, x_in, *args):
        sz = self.sizes.copy()
        for idx, place in enumerate(self.input_places):
            sz[place] = args[idx]
        out = x_in.expand(sz)
        return out


class Expand_as(nn.Module):

    def __init__(self, org_module=None, config=None):
        super(Expand_as, self).__init__()

    def forward(self, x_in, *args):
        out = x_in.expand_as(*args)
        return out


class Transpose(nn.Module):

    def __init__(self, org_module=None, config=None):
        super(Transpose, self).__init__()
        self.dim0 = org_module.args[1]
        self.dim1 = org_module.args[2]

    def forward(self, x_in, *args):
        out = torch.transpose(x_in, dim0=self.dim0, dim1=self.dim1)
        return out


class Permute(nn.Module):

    def __init__(self, org_module=None, config=None):
        super(Permute, self).__init__()
        self.dims = org_module.args[1:]

    def forward(self, x_in, *args):
        out = torch.permute(x_in, dims=self.dims)
        return out


class StochasticDepth(nn.Module):

    def __init__(self, org_module, config=None):
        super().__init__()

    def forward(self, x_in, *args, **kwargs):
        out = stochastic_depth(x_in, *args, **kwargs)
        return out


class Dropout(nn.Module):

    def __init__(self, org_module, config=None):
        super().__init__()
        self.inplace = org_module.inplace
        self.p = org_module.p

    def forward(self, x_in):
        return F.dropout(x_in, self.p, training=self.training, inplace=self.inplace)


class QIdentity(QuantOpr):

    def __init__(self, org_module=None, config=None):
        super(QIdentity, self).__init__()
        self._repr_info = 'QIdentity'

    def forward(self, x_in):
        x_in = self.input_quantizer(x_in)
        return x_in


class QSoftmax(QuantOpr):

    def __init__(self, org_module=None, config=None):
        super().__init__()
        assert isinstance(org_module, torch.fx.Node)
        if 'dim' in org_module.kwargs:
            self.dim = org_module.kwargs['dim']
        else:
            self.dim = org_module.args[1]
        self._repr_info = 'QSoftmax '

    def forward(self, x_in, *args, **kwargs):
        if 'dim' in kwargs:
            assert self.dim == kwargs['dim'], 'parameter mismatch in softmax'
        else:
            assert self.dim == args[0], 'parameter mismatch in softmax'
        x_in = self.input_quantizer(x_in)
        out = F.softmax(x_in, dim=self.dim)
        return out


class Clone(nn.Module):
    """clone can be useful in quantization-aware training"""

    def __init__(self, org_module=None, config=None):
        super().__init__()

    def forward(self, x):
        return x.clone()


class Contiguous(nn.Module):

    def __init__(self, org_module=None, config=None):
        super().__init__()

    def forward(self, x):
        return x.contiguous()


class Granularity(Enum):
    LAYERWISE = 0
    CHANNELWISE = 1


OBSERVERS_MAP = {}


def register_observer(observer):
    OBSERVERS_MAP[observer.TYPE.lower()] = observer
    return observer


class QTracer(Tracer):

    def __init__(self, skipped_module_names: List[str]):
        super().__init__()
        self.skipped_module_names = skipped_module_names

    def _probe(self, module_name, patterns):
        for p in patterns:
            if fnmatch(module_name, p):
                return True
        return False

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) ->bool:
        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential) or self._probe(module_qualified_name, self.skipped_module_names)


def ort_fake_quant(x_f, scale, zero_point, qdesc):
    assert x_f.device == scale.device == zero_point.device, 'input, scale and zero_point of quantizer must be on same device!'
    qmin, qmax = qdesc.qrange
    if torch.cuda.is_available() and 'cuda' in x_f.device.type:
        if x_f.dtype == torch.float16:
            x_f = x_f.float()
        if qdesc.is_perchannel:
            x_dq = fake_quant_kernel.quant_perchannel_forward(x_f.contiguous(), scale.contiguous(), zero_point.contiguous(), qmin, qmax, qdesc.ch_axis, 0)
        else:
            x_dq = fake_quant_kernel.quant_pertensor_forward(x_f.contiguous(), scale, zero_point, qmin, qmax, 0)
    else:
        zp = zero_point.round()
        x_q = torch.clamp((x_f / scale).round() + zp, qmin, qmax)
        x_dq = (x_q - zp) * scale
    return x_dq


def trt_fake_quant(x_f, scale, zero_point, qdesc):
    assert x_f.device == scale.device == zero_point.device, 'input, scale and zero_point of quantizer must be on same device!'
    assert abs(zero_point).sum() == 0, 'tensorrt only support symmetric quant, but zp={}'.format(zero_point)
    qmin, qmax = qdesc.qrange
    if torch.cuda.is_available() and 'cuda' in x_f.device.type:
        if x_f.dtype == torch.float16:
            x_f = x_f.float()
        if qdesc.is_perchannel:
            x_dq = fake_quant_kernel.quant_perchannel_forward(x_f.contiguous(), scale.contiguous(), zero_point.contiguous(), qmin, qmax, qdesc.ch_axis, 0)
        else:
            x_dq = fake_quant_kernel.quant_pertensor_forward(x_f.contiguous(), scale, zero_point, qmin, qmax, 0)
    else:
        x_q = torch.clamp((x_f / scale).round(), qmin, qmax)
        x_dq = x_q * scale
    return x_dq


fake_quant_factory = {Backend.VIRTUAL: ort_fake_quant, Backend.ONNXRUNTIME: ort_fake_quant, Backend.TENSORRT: trt_fake_quant}


class STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, zero_point, qdesc, backend):
        x_fq = fake_quant_factory[backend](x, scale, zero_point, qdesc)
        ctx.save_for_backward(x, scale, zero_point)
        ctx.qdesc = qdesc
        return x_fq

    @staticmethod
    def backward(ctx, gout):
        x, scale, zero_point = ctx.saved_tensors
        qdesc = ctx.qdesc
        qmin, qmax = qdesc.qmin, qdesc.qmax
        if torch.cuda.is_available():
            if x.dtype == torch.float16:
                x = x.float()
            if qdesc.is_perchannel:
                gx, gs, gzp = fake_quant_kernel.quant_perchannel_backward(x.contiguous(), scale.contiguous(), zero_point.float().contiguous(), gout.contiguous(), qmin, qmax, qdesc.ch_axis, 0)
            else:
                gx, gs, gzp = fake_quant_kernel.quant_pertensor_backward(x.contiguous(), scale, zero_point.float(), gout.contiguous(), qmin, qmax, 0)
            gs = gs if scale.requires_grad else None
            gzp = gzp if zero_point.requires_grad else None
        else:
            raise NotImplementedError('We recommended that use cuda to speedup when training')
        return gx, gs, gzp, None, None


QUANTIZERS_MAP = {}


def register_quantizer(quantizer):
    QUANTIZERS_MAP[quantizer.TYPE.lower()] = quantizer
    return quantizer


class PruneGraph(object):
    """网络剪枝，去掉和输出无关的算子。"""

    def __init__(self):
        pass

    def apply(self, m: torch.fx.GraphModule):
        """运行剪枝。

        Args:
            m (torch.fx.GraphModule): 需要剪枝的模型。

        Returns:
            torch.fx.GraphModule: 剪枝后的新模型。
        """
        node_dict = {i.name: i for i in m.graph.nodes}
        q = []
        q_names = set()
        for node in m.graph.nodes:
            if node.op == 'output':
                q_names.add(node.name)
                q.append(node.name)
        pos = 0
        while pos < len(q):
            node_name = q[pos]
            pos += 1
            node = node_dict[node_name]
            for input_node in node.all_input_nodes:
                if isinstance(input_node, torch.fx.Node):
                    if input_node.name not in q_names:
                        q_names.add(input_node.name)
                        q.append(input_node.name)
        delete_nodes = [i for i in m.graph.nodes if i.name not in q_names]
        for delete_node in reversed(delete_nodes):
            m.graph.erase_node(delete_node)
        m.recompile()
        return fx.GraphModule(m, m.graph)


def fx_symbolic_trace(model):
    if not getattr(model, 'graph', None):
        model = torch.fx.symbolic_trace(model)
    return model


def fuse_operations(model: torch.fx.GraphModule, config, custom_lists=None):
    model = fx_symbolic_trace(model)
    model = PruneGraph().apply(model)
    cur_list = custom_lists if custom_lists else default_lists
    for task in cur_list:
        if getattr(config, task.upper(), True):
            module = importlib.import_module('.{}'.format(task), package=__package__)
            if getattr(module, 'ReplacePatterns', None):
                classes = module.ReplacePatterns
                for cls in classes:
                    cls.apply(model)
            else:
                func = module.ReplacePattern
                func().apply(model)
            model = PruneGraph().apply(model)
    return model


lists = ['fuse_bn', 'disable_unnecessary_quant']


def simplify(model: torch.fx.GraphModule):
    model = fx_symbolic_trace(model)
    model = PruneGraph().apply(model)
    for task in lists:
        module = importlib.import_module('.{}'.format(task), package=__package__).ReplacePattern
        module().apply(model)
        model = PruneGraph().apply(model)
    return model


class SparseOpr(nn.Module, ABC):

    def __init__(self):
        super(SparseOpr, self).__init__()
        self._repr_info = 'base'

    def forward(self, x_in: torch.Tensor):
        raise NotImplementedError('no found a forward in {}'.format(self.__class__.__name__))

    def build_mask(self, pre_mask=None):
        raise NotImplementedError('no found a calc_mask in {}'.format(self.__class__.__name__))

    def build_sparser(self, config):
        self.sparser = build_sparser(config, opr=self._repr_info)


SMODULE_MAP = {}


def register_smodule(sources: [nn.Module, str, ...]):

    def real_register(smodule):
        for src in sources:
            SMODULE_MAP[src] = smodule
        return smodule
    return real_register


class SConv2d(SparseOpr):

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Conv2d)
        super().__init__()
        self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding, dilation=org_module.dilation, groups=org_module.groups)
        self.weight = org_module.weight
        self.bias = org_module.bias
        w_mask = torch.ones_like(self.weight)
        b_mask = torch.ones_like(self.bias) if self.bias is not None else None
        self.register_buffer('w_mask', w_mask)
        self.register_buffer('b_mask', b_mask)
        self._repr_info = 'S' + org_module.__repr__()

    def calc_mask(self, pre_mask=None):
        self.w_mask = self.sparser.calc_mask(self.weight)
        if self.sparser.type == 'structed':
            mask = self.w_mask[:, 0, 0, 0]
            if self.bias is not None:
                self.b_mask.data.copy_(mask.data)
            if self.sparser.strategy == 'l1norm':
                return mask
        return None

    def forward(self, x_in: torch.Tensor):
        weight = self.weight * self.w_mask
        bias = self.bias * self.b_mask if self.bias is not None else self.bias
        out = F.conv2d(x_in, weight, bias, **self.fwd_kwargs)
        return out


class SLinear(SparseOpr):

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Linear)
        super().__init__()
        self.weight = org_module.weight
        self.bias = org_module.bias
        w_mask = torch.ones_like(self.weight)
        b_mask = torch.ones_like(self.bias) if self.bias is not None else None
        self.register_buffer('w_mask', w_mask)
        self.register_buffer('b_mask', b_mask)
        self._repr_info = 'S' + org_module.__repr__()

    def calc_mask(self, pre_mask=None):
        self.w_mask = self.sparser.calc_mask(self.weight)
        if self.sparser.type == 'structed':
            mask = self.w_mask[:, 0]
            if self.bias is not None:
                self.b_mask.data.copy_(mask.data)
        return None

    def forward(self, x_in: torch.Tensor):
        weight = self.weight * self.w_mask
        bias = self.bias * self.b_mask if self.bias is not None else self.bias
        out = F.linear(x_in, weight, bias)
        return out


class SBatchNorm2d(SparseOpr):

    def __init__(self, org_module, config=None):
        super().__init__()
        self.module = org_module
        self.mask = torch.ones([1, self.module.num_features, 1, 1])
        self._repr_info = 'S' + org_module.__repr__()

    def calc_mask(self, pre_mask=None):
        if self.sparser.type == 'structed' and self.sparser.strategy == 'l1norm' and pre_mask is not None:
            pre_mask = pre_mask.reshape(self.mask.shape)
            self.mask.data.copy_(pre_mask.data)
        return None

    def forward(self, x_in):
        out = self.module(x_in) * self.mask
        return out


class SparseModel(nn.Module):

    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model = fx.symbolic_trace(model)
        self._run_simplifiers()
        self._convert2sparsemodule()
        self._build_sparser()

    def _convert2sparsemodule(self):
        """
        将网络中部分node转成对应的sparse_module
        """
        named_modules = dict(self.model.named_modules(remove_duplicate=False))
        traced = self.model
        traced.graph.print_tabular()
        snodes = []
        for n in traced.graph.nodes:
            if not isinstance(n, fx.Node) or n in snodes:
                continue
            elif n.op == 'call_module':
                assert n.target in named_modules, 'no found {} in model'.format(n.target)
                if type(named_modules[n.target]) in SMODULE_MAP:
                    org_module = named_modules[n.target]
                    new_module = SMODULE_MAP[type(org_module)](org_module)
                else:
                    new_module = named_modules[n.target]
            elif n.op in ['call_function', 'call_method', 'placeholder', 'get_attr', 'output']:
                continue
            with traced.graph.inserting_after(n):
                traced.add_module(n.name, new_module)
                new_node = traced.graph.call_module(n.name, n.args, n.kwargs)
                snodes.append(new_node)
                n.replace_all_uses_with(new_node)
                traced.graph.erase_node(n)
        traced.recompile()
        self.model = fx.GraphModule(traced, traced.graph)

    def _build_sparser(self):
        """
        递归对每个SparseModule建立sparser
        """
        for n, m in self.model.named_modules():
            if isinstance(m, SparseOpr):
                _config = self.config.clone()
                m.build_sparser(_config)

    def disable_sparse_before_add(self):
        named_modules = dict(self.model.named_modules())
        add_nodes = [n for n in self.model.graph.nodes if n.op == 'call_function' and n.target in [operator.add, torch.add]]
        for add_node in add_nodes:
            add_inputs = [a for a in add_node.args if isinstance(a, torch.fx.Node)]
            while len(add_inputs) > 0:
                n = add_inputs.pop()
                if n.op == 'call_module' and n.target in named_modules:
                    m = named_modules[n.target]
                else:
                    m = None
                if hasattr(m, 'sparser') and m.sparser:
                    m.sparser.set_ratio(0.0)
                if not isinstance(m, SConv2d):
                    n_list = [a for a in n.args if isinstance(a, torch.fx.Node)]
                    add_inputs.extend(n_list)

    def calc_params(self):
        pre_mask = None
        for node in self.model.graph.nodes:
            if node.op == 'call_module':
                module = getattr(self.model, node.target, None)
                if isinstance(module, SparseOpr) and getattr(module, 'sparser', None):
                    pre_mask = module.calc_mask(pre_mask)

    def _run_simplifiers(self):
        self.model = simplify(self.model)

    def prepare_calibration(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def export_onnx(self, dummy_data, name, input_names=None, output_names=None, dynamic_axes=None, opset_version=13, verbose=False, extra_info=False):
        self.eval()
        torch.onnx.export(self.model.cpu(), dummy_data.cpu(), name, opset_version=opset_version, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, verbose=verbose)


SPARSERS_MAP = {}


def register_sparser(sparser):
    SPARSERS_MAP[sparser.STRATEGY.lower()] = sparser
    return sparser


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Clone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Contiguous,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvAdd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Darknet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DownSample1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (DownSample2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (DownSample3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
    (DownSample4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (DownSample5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (MatMul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (QDivide,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (QEqual,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (QFloorDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (QMul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (QSubtract,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResLayer,
     lambda: ([], {'ni': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Size,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (YOLOv3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Yolov4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (eSEModule,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_megvii_research_Sparsebit(_paritybench_base):
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

