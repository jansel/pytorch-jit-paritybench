import sys
_module = sys.modules[__name__]
del sys
datasets = _module
base = _module
base_dataset = _module
builder = _module
epickitchen100 = _module
epickitchen100_feature = _module
hmdb51 = _module
imagenet = _module
kinetics400 = _module
kinetics700 = _module
ssv2 = _module
ucf101 = _module
utils = _module
collate_functions = _module
mixup = _module
preprocess_ssv2 = _module
random_erasing = _module
transformations = _module
models = _module
backbone = _module
base_blocks = _module
builder = _module
models = _module
slowfast = _module
transformer = _module
module_zoo = _module
branches = _module
csn_branch = _module
non_local = _module
r2d3d_branch = _module
r2plus1d_branch = _module
s3dg_branch = _module
slowfast_branch = _module
tada_branch = _module
heads = _module
bmn_head = _module
mosi_head = _module
slowfast_head = _module
transformer_head = _module
ops = _module
tadaconv = _module
stems = _module
downsample_stem = _module
embedding_stem = _module
r2plus1d_stem = _module
init_helper = _module
lars = _module
localization_losses = _module
losses = _module
lr_policy = _module
model_ema = _module
optimizer = _module
params = _module
run = _module
submission_test = _module
test = _module
test_epic_localization = _module
train = _module
sslgenerators = _module
mosi_generator = _module
bboxes_1d = _module
bucket = _module
checkpoint = _module
config = _module
distributed = _module
eval_epic_detection = _module
eval_tal = _module
launcher = _module
logging = _module
meters = _module
metrics = _module
misc = _module
registry = _module
sampler = _module
tal_tools = _module
tensor = _module
timer = _module
val_dist_sampler = _module

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


import random


import torch


import torchvision


import torch.nn as nn


import torch.utils.data


import torch.utils.dlpack as dlpack


import re


import abc


import time


import numpy as np


from torchvision.transforms import Compose


import itertools


from torch.utils.data._utils.collate import default_collate


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


import torchvision.transforms._transforms_video as transforms


import torch.nn.functional as F


import math


import torchvision.transforms._functional_video as F


from torchvision.transforms import Lambda


import numbers


from torchvision.utils import make_grid


from torchvision.utils import save_image


from torch import nn


from torch import einsum


from torch.nn.modules.utils import _triple


from collections import OrderedDict


import warnings


from torch.nn.init import _calculate_fan_in_and_fan_out


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.optim import *


from copy import deepcopy


import copy


from torch.hub import tqdm


from torch.hub import load_state_dict_from_url as load_url


import functools


import logging


import torch.distributed as dist


from collections import defaultdict


from collections import deque


from torch.utils.data.sampler import Sampler


import pandas as pd


class Registry(object):
    """
    The Registry class provides a registry for all things
    To initialize:
        REGISTRY = Registry()
    
    To register a tracker:
        @REGISTRY.register()
        class Model():
            ...
    """

    def __init__(self, table_name=''):
        """
        Initializes the registry.
        Args:
            table_name (str): specifies the name of the registry
        """
        self._entry_map = {}
        self.table_name = table_name

    def _register(self, name, entry):
        """
        Registers the instance.
        Args:
            name (str): name of the entry
            entry ():   instance of the entry, could be any type
        """
        assert type(name) is str
        assert name not in self._entry_map.keys(), '{} {} already registered.'.format(self.table_name, name)
        self._entry_map[name] = entry

    def register(self):
        """
        Wrapper function for registering a module.
        """

        def reg(obj):
            name = obj.__name__
            self._register(name, obj)
            return obj
        return reg

    def get(self, name):
        """
        Returns the instance specified by the name. 
        Args:
            name (str): name of the specified instance.
        """
        if name not in self._entry_map.keys():
            return None
        obj = self._entry_map.get(name)
        return obj

    def get_all_registered(self):
        """
        Prints all registered class. 
        """
        return self._entry_map.keys()


BACKBONE_REGISTRY = Registry('Backbone')


BRANCH_REGISTRY = Registry('Branch')


def update_3d_conv_params(cfg, conv, idx):
    """
    Automatically decodes parameters for 3D convolution blocks according to the config and its index in the model.
    Args: 
        cfg (Config):       Config object that contains model parameters such as channel dimensions, whether to downsampling or not, etc.
        conv (BaseBranch):  Branch whose parameters needs to be specified. 
        idx (list):         List containing the index of the current block. ([stage_id, block_id])
    """
    stage_id, block_id = idx
    conv.stage_id = stage_id
    conv.block_id = block_id
    if block_id == 0:
        conv.dim_in = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id - 1]
        if hasattr(cfg.VIDEO.BACKBONE, 'ADD_FUSION_CHANNEL') and cfg.VIDEO.BACKBONE.ADD_FUSION_CHANNEL:
            conv.dim_in = conv.dim_in * cfg.VIDEO.BACKBONE.SLOWFAST.CONV_CHANNEL_RATIO // cfg.VIDEO.BACKBONE.SLOWFAST.BETA + conv.dim_in
        conv.downsampling = cfg.VIDEO.BACKBONE.DOWNSAMPLING[stage_id]
        conv.downsampling_temporal = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[stage_id]
    else:
        conv.downsampling = False
        conv.dim_in = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
    conv.num_filters = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
    conv.bn_mmt = cfg.BN.MOMENTUM
    conv.bn_eps = cfg.BN.EPS
    conv.kernel_size = cfg.VIDEO.BACKBONE.KERNEL_SIZE[stage_id]
    conv.expansion_ratio = cfg.VIDEO.BACKBONE.EXPANSION_RATIO if hasattr(cfg.VIDEO.BACKBONE, 'EXPANSION_RATIO') else None
    if conv.downsampling:
        if conv.downsampling_temporal:
            conv.stride = [2, 2, 2]
        else:
            conv.stride = [1, 2, 2]
    else:
        conv.stride = [1, 1, 1]
    if isinstance(cfg.VIDEO.BACKBONE.DEPTH, str):
        conv.transformation = 'bottleneck'
    elif cfg.VIDEO.BACKBONE.DEPTH <= 34:
        conv.transformation = 'simple_block'
    else:
        conv.transformation = 'bottleneck'
    num_downsampling_spatial = sum(cfg.VIDEO.BACKBONE.DOWNSAMPLING[:stage_id + (block_id > 0)])
    if 'DownSample' in cfg.VIDEO.BACKBONE.STEM.NAME:
        num_downsampling_spatial += 1
    num_downsampling_temporal = sum(cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[:stage_id + (block_id > 0)])
    conv.h = cfg.DATA.TRAIN_CROP_SIZE // 2 ** num_downsampling_spatial + cfg.DATA.TRAIN_CROP_SIZE // 2 ** (num_downsampling_spatial - 1) % 2
    conv.w = conv.h
    conv.t = cfg.DATA.NUM_INPUT_FRAMES // 2 ** num_downsampling_temporal


class Base3DBlock(nn.Module):
    """
    Constructs a base 3D block, composed of a shortcut and a conv branch.
    """

    def __init__(self, cfg, block_idx):
        """
        Args: 
            cfg         (Config): global config object. 
            block_idx   (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(Base3DBlock, self).__init__()
        self.cfg = cfg
        update_3d_conv_params(cfg, self, block_idx)
        self._construct_block(cfg=cfg, block_idx=block_idx)

    def _construct_block(self, cfg, block_idx):
        if self.dim_in != self.num_filters or self.downsampling:
            self.short_cut = nn.Conv3d(self.dim_in, self.num_filters, kernel_size=1, stride=self.stride, padding=0, bias=False)
            self.short_cut_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.conv_branch = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, block_idx)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        short_cut = x
        if hasattr(self, 'short_cut'):
            short_cut = self.short_cut_bn(self.short_cut(short_cut))
        x = self.relu(short_cut + self.conv_branch(x))
        return x


class Base3DResStage(nn.Module):
    """
    ResNet Stage containing several blocks.
    """

    def __init__(self, cfg, num_blocks, stage_idx):
        """
        Args:
            num_blocks (int): number of blocks contained in this res-stage.
            stage_idx  (int): the stage index of this res-stage.
        """
        super(Base3DResStage, self).__init__()
        self.cfg = cfg
        self.num_blocks = num_blocks
        self._construct_stage(cfg=cfg, stage_idx=stage_idx)

    def _construct_stage(self, cfg, stage_idx):
        res_block = Base3DBlock(cfg=cfg, block_idx=[stage_idx, 0])
        self.add_module('res_{}'.format(1), res_block)
        for i in range(self.num_blocks - 1):
            res_block = Base3DBlock(cfg=cfg, block_idx=[stage_idx, i + 1])
            self.add_module('res_{}'.format(i + 2), res_block)
        if cfg.VIDEO.BACKBONE.NONLOCAL.ENABLE and stage_idx + 1 in cfg.VIDEO.BACKBONE.NONLOCAL.STAGES:
            non_local = BRANCH_REGISTRY.get('NonLocal')(cfg=cfg, block_idx=[stage_idx, i + 2])
            self.add_module('nonlocal', non_local)

    def forward(self, x):
        for i in range(self.num_blocks):
            res_block = getattr(self, 'res_{}'.format(i + 1))
            x = res_block(x)
        if hasattr(self, 'nonlocal'):
            non_local = getattr(self, 'nonlocal')
            x = non_local(x)
        return x


STEM_REGISTRY = Registry('Stem')


def c2_msra_fill(module: nn.Module) ->None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def _init_convnet_weights(model, fc_init_std=0.01, zero_init_final_bn=True):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if hasattr(m, 'skip_init'):
            continue
        if isinstance(m, nn.Conv3d) and not hasattr(m, 'linear'):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d):
            if hasattr(m, 'transform_final_bn') and m.transform_final_bn and zero_init_final_bn:
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            if m.weight is not None:
                m.weight.data.fill_(batchnorm_weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear) or hasattr(m, 'linear'):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()


_n_conv_resnet = {(10): (1, 1, 1, 1), (16): (2, 2, 2, 1), (18): (2, 2, 2, 2), (26): (2, 2, 2, 2), (34): (3, 4, 6, 3), (50): (3, 4, 6, 3), (101): (3, 4, 23, 3), (152): (3, 8, 36, 3)}


class ResNet3D(nn.Module):
    """
    Meta architecture for 3D ResNet based models. 
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(ResNet3D, self).__init__()
        self._construct_backbone(cfg)

    def _construct_backbone(self, cfg):
        self.conv1 = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg=cfg)
        n1, n2, n3, n4 = _n_conv_resnet[cfg.VIDEO.BACKBONE.DEPTH]
        self.conv2 = Base3DResStage(cfg=cfg, num_blocks=n1, stage_idx=1)
        self.conv3 = Base3DResStage(cfg=cfg, num_blocks=n2, stage_idx=2)
        self.conv4 = Base3DResStage(cfg=cfg, num_blocks=n3, stage_idx=3)
        self.conv5 = Base3DResStage(cfg=cfg, num_blocks=n4, stage_idx=4)
        if cfg.VIDEO.BACKBONE.INITIALIZATION == 'kaiming':
            _init_convnet_weights(self)

    def forward(self, x):
        if type(x) is list:
            x = x[0]
        elif isinstance(x, dict):
            x = x['video']
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class BaseModule(nn.Module):
    """
    Constructs base module that contains basic visualization function and corresponding hooks.
    Note: The visualization function has only tested in the single GPU scenario.
        By default, the visualization is disabled.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseModule, self).__init__()
        self.cfg = cfg
        self.id = 0
        if self.cfg.VISUALIZATION.ENABLE and self.cfg.VISUALIZATION.FEATURE_MAPS.ENABLE:
            self.base_output_dir = self.cfg.VISUALIZATION.FEATURE_MAPS.BASE_OUTPUT_DIR
            self.register_forward_hook(self.visualize_features)

    def visualize_features(self, module, input, output_x):
        """
        Visualizes and saves the normalized output features for the module.
        """
        b, c, t, h, w = output_x.shape
        xmin, xmax = output_x.min(1).values.unsqueeze(1), output_x.max(1).values.unsqueeze(1)
        x_vis = ((output_x.detach() - xmin) / (xmax - xmin)).permute(0, 1, 3, 2, 4).reshape(b, c * h, t * w).detach().cpu().numpy()
        if hasattr(self, 'stage_id'):
            stage_id = self.stage_id
            block_id = self.block_id
        else:
            stage_id = 0
            block_id = 0
        for i in range(b):
            if not os.path.exists(f'{self.base_output_dir}/{self.cfg.VISUALIZATION.NAME}/im_{self.id + i}/'):
                os.makedirs(f'{self.base_output_dir}/{self.cfg.VISUALIZATION.NAME}/im_{self.id + i}/')
        self.id += b


class InceptionBaseConv3D(BaseModule):
    """
    Constructs basic inception 3D conv.
    Modified from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """

    def __init__(self, cfg, in_planes, out_planes, kernel_size, stride, padding=0):
        super(InceptionBaseConv3D, self).__init__(cfg)
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv.weight.data.normal_(mean=0, std=0.01)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SelfGating(nn.Module):

    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G"""
        spatiotemporal_average = torch.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


class InceptionBlock3D(nn.Module):
    """
    Element constructing the S3D/S3DG.
    See models/base/backbone.py L99-186.

    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """

    def __init__(self, cfg, in_planes, out_planes):
        super(InceptionBlock3D, self).__init__()
        _gating = cfg.VIDEO.BACKBONE.BRANCH.GATING
        assert len(out_planes) == 6
        assert isinstance(out_planes, list)
        [num_out_0_0a, num_out_1_0a, num_out_1_0b, num_out_2_0a, num_out_2_0b, num_out_3_0b] = out_planes
        self.branch0 = nn.Sequential(InceptionBaseConv3D(cfg, in_planes, num_out_0_0a, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(InceptionBaseConv3D(cfg, in_planes, num_out_1_0a, kernel_size=1, stride=1), BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, num_out_1_0a, num_out_1_0b, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(InceptionBaseConv3D(cfg, in_planes, num_out_2_0a, kernel_size=1, stride=1), BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, num_out_2_0a, num_out_2_0b, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), InceptionBaseConv3D(cfg, in_planes, num_out_3_0b, kernel_size=1, stride=1))
        self.out_channels = sum([num_out_0_0a, num_out_1_0b, num_out_2_0b, num_out_3_0b])
        self.gating = _gating
        if _gating:
            self.gating_b0 = SelfGating(num_out_0_0a)
            self.gating_b1 = SelfGating(num_out_1_0b)
            self.gating_b2 = SelfGating(num_out_2_0b)
            self.gating_b3 = SelfGating(num_out_3_0b)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if self.gating:
            x0 = self.gating_b0(x0)
            x1 = self.gating_b1(x1)
            x2 = self.gating_b2(x2)
            x3 = self.gating_b3(x3)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Inception3D(nn.Module):
    """
    Backbone architecture for I3D/S3DG. 
    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Inception3D, self).__init__()
        _input_channel = cfg.DATA.NUM_INPUT_CHANNELS
        self._construct_backbone(cfg, _input_channel)

    def _construct_backbone(self, cfg, input_channel):
        self.Conv_1a = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg, input_channel, 64, kernel_size=7, stride=2, padding=3)
        self.block1 = nn.Sequential(self.Conv_1a)
        self.MaxPool_2a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Conv_2b = InceptionBaseConv3D(cfg, 64, 64, kernel_size=1, stride=1)
        self.Conv_2c = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, 64, 192, kernel_size=3, stride=1, padding=1)
        self.block2 = nn.Sequential(self.MaxPool_2a, self.Conv_2b, self.Conv_2c)
        self.MaxPool_3a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Mixed_3b = InceptionBlock3D(cfg, in_planes=192, out_planes=[64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionBlock3D(cfg, in_planes=256, out_planes=[128, 128, 192, 32, 96, 64])
        self.block3 = nn.Sequential(self.MaxPool_3a, self.Mixed_3b, self.Mixed_3c)
        self.MaxPool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = InceptionBlock3D(cfg, in_planes=480, out_planes=[192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionBlock3D(cfg, in_planes=512, out_planes=[160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionBlock3D(cfg, in_planes=512, out_planes=[128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionBlock3D(cfg, in_planes=512, out_planes=[112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionBlock3D(cfg, in_planes=528, out_planes=[256, 160, 320, 32, 128, 128])
        self.block4 = nn.Sequential(self.MaxPool_4a, self.Mixed_4b, self.Mixed_4c, self.Mixed_4d, self.Mixed_4e, self.Mixed_4f)
        self.MaxPool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = InceptionBlock3D(cfg, in_planes=832, out_planes=[256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionBlock3D(cfg, in_planes=832, out_planes=[384, 192, 384, 48, 128, 128])
        self.block5 = nn.Sequential(self.MaxPool_5a, self.Mixed_5b, self.Mixed_5c)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['video']
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class SimpleLocalizationConv(nn.Module):
    """
    Backbone architecture for temporal action localization, which only contains three simple convs.
    """

    def __init__(self, cfg):
        super(SimpleLocalizationConv, self).__init__()
        _input_channel = cfg.DATA.NUM_INPUT_CHANNELS
        self.hidden_dim_1d = cfg.VIDEO.DIM1D
        self.layer_num = cfg.VIDEO.BACKBONE_LAYER
        self.groups_num = cfg.VIDEO.BACKBONE_GROUPS_NUM
        self._construct_backbone(cfg, _input_channel)

    def _construct_backbone(self, cfg, input_channel):
        self.conv_list = [nn.Conv1d(input_channel, self.hidden_dim_1d, kernel_size=3, padding=1, groups=self.groups_num), nn.ReLU(inplace=True)]
        assert self.layer_num >= 1
        for ln in range(self.layer_num - 1):
            self.conv_list.append(nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=self.groups_num))
            self.conv_list.append(nn.ReLU(inplace=True))
        self.conv_layer = nn.Sequential(*self.conv_list)

    def forward(self, x):
        x['video'] = self.conv_layer(x['video'])
        return x


class BaseBranch(BaseModule):
    """
    Constructs the base convolution branch for ResNet based approaches.
    """

    def __init__(self, cfg, block_idx, construct_branch=True):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
            construct_branch (bool):   whether or not to automatically construct the branch.
                In the cases that the branch is not automatically contructed, e.g., some extra
                parameters need to be specified before branch construction, the branch could be
                constructed by "self._construct_branch" function.
        """
        super(BaseBranch, self).__init__(cfg)
        self.cfg = cfg
        update_3d_conv_params(cfg, self, block_idx)
        if construct_branch:
            self._construct_branch()

    def _construct_branch(self):
        if self.transformation == 'simple_block':
            self._construct_simple_block()
        elif self.transformation == 'bottleneck':
            self._construct_bottleneck()

    @abc.abstractmethod
    def _construct_simple_block(self):
        return

    @abc.abstractmethod
    def _construct_bottleneck(self):
        return

    @abc.abstractmethod
    def forward(self, x):
        return


class Base2DStem(BaseModule):
    """
    Constructs basic ResNet 2D Stem.
    A single 2D convolution is performed in the base 2D stem.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base2DStem, self).__init__(cfg)
        self.cfg = cfg
        _downsampling = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        _downsampling_temporal = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        if _downsampling:
            _stride = [1, 2, 2]
        else:
            _stride = [1, 1, 1]
        self._construct_block(cfg=cfg, dim_in=cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS, num_filters=cfg.VIDEO.BACKBONE.NUM_FILTERS[0], kernel_sz=cfg.VIDEO.BACKBONE.KERNEL_SIZE[0], stride=_stride, bn_eps=cfg.BN.EPS, bn_mmt=cfg.BN.MOMENTUM)

    def _construct_block(self, cfg, dim_in, num_filters, kernel_sz, stride, bn_eps=1e-05, bn_mmt=0.1):
        self.a = nn.Conv3d(dim_in, num_filters, kernel_size=[1, kernel_sz[1], kernel_sz[2]], stride=[1, stride[1], stride[2]], padding=[0, kernel_sz[1] // 2, kernel_sz[2] // 2], bias=False)
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x


class Base3DStem(BaseModule):
    """
    Constructs basic ResNet 3D Stem.
    A single 3D convolution is performed in the base 3D stem.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base3DStem, self).__init__(cfg)
        self.cfg = cfg
        _downsampling = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        _downsampling_temporal = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        if _downsampling:
            if _downsampling_temporal:
                _stride = [2, 2, 2]
            else:
                _stride = [1, 2, 2]
        else:
            _stride = [1, 1, 1]
        self._construct_block(cfg=cfg, dim_in=cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS, num_filters=cfg.VIDEO.BACKBONE.NUM_FILTERS[0], kernel_sz=cfg.VIDEO.BACKBONE.KERNEL_SIZE[0], stride=_stride, bn_eps=cfg.BN.EPS, bn_mmt=cfg.BN.MOMENTUM)

    def _construct_block(self, cfg, dim_in, num_filters, kernel_sz, stride, bn_eps=1e-05, bn_mmt=0.1):
        self.a = nn.Conv3d(dim_in, num_filters, kernel_size=kernel_sz, stride=stride, padding=[kernel_sz[0] // 2, kernel_sz[1] // 2, kernel_sz[2] // 2], bias=False)
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x


HEAD_REGISTRY = Registry('Head')


class BaseHead(nn.Module):
    """
    Constructs base head.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseHead, self).__init__()
        self.cfg = cfg
        dim = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        num_classes = cfg.VIDEO.HEAD.NUM_CLASSES
        dropout_rate = cfg.VIDEO.HEAD.DROPOUT_RATE
        activation_func = cfg.VIDEO.HEAD.ACTIVATION
        self._construct_head(dim, num_classes, dropout_rate, activation_func)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(dim, num_classes, bias=True)
        if activation_func == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(activation_func))

    def forward(self, x):
        """
        Returns:
            x (Tensor): classification predictions.
            logits (Tensor): global average pooled features.
        """
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            out = self.dropout(x)
        else:
            out = x
        out = self.out(out)
        if not self.training:
            out = self.activation(out)
        out = out.view(out.shape[0], -1)
        return out, x.view(x.shape[0], -1)


class BaseHeadx2(BaseHead):
    """
    Constructs two base heads in parallel.
    This is specifically for EPIC-KITCHENS dataset, where 'noun' and 'verb' class are predicted.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseHeadx2, self).__init__(cfg)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        if activation_func == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_func == 'identity':
            self.activation = nn.Identity()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(activation_func))

    def forward(self, x):
        """
        Returns:
            x (dict): dictionary of classification predictions,
                with keys "verb_class" and "noun_class" indicating
                the predictions on the verb and noun.
            logits (Tensor): global average pooled features.
        """
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            out1 = self.dropout(x)
            out2 = out1
        else:
            out1 = x
            out2 = x
        out1 = self.linear1(out1)
        out2 = self.linear2(out2)
        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        return {'verb_class': out1, 'noun_class': out2}, x


def drop_path(x, drop_prob: float=0.0, training: bool=False):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class BaseVideoModel(nn.Module):
    """
    Standard video model.
    The model is divided into the backbone and the head, where the backbone
    extracts features and the head performs classification.

    The backbones can be defined in model/base/backbone.py or anywhere else
    as long as the backbone is registered by the BACKBONE_REGISTRY.
    The heads can be defined in model/module_zoo/heads/ or anywhere else
    as long as the head is registered by the HEAD_REGISTRY.

    The registries automatically finds the registered modules and construct 
    the base video model.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseVideoModel, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def train(self, mode=True):
        """Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        super(BaseVideoModel, self).train(mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.cfg.BN.FREEZE:
                module.train(False)
        return self


MODEL_REGISTRY = Registry('Model')


class MoSINet(BaseVideoModel):

    def __init__(self, cfg):
        super(MoSINet, self).__init__(cfg)

    def forward(self, x):
        if isinstance(x, dict):
            x_data = x['video']
        else:
            x_data = x
        b, n, c, t, h, w = x_data.shape
        x_data = x_data.reshape(b * n, c, t, h, w)
        res, logits = super(MoSINet, self).forward(x_data)
        pred = {}
        if isinstance(res, dict):
            for k, v in res.items():
                pred[k] = v
        else:
            pred['move_joint'] = res
        return pred, logits


class FuseFastToSlow(nn.Module):

    def __init__(self, cfg, stage_idx, mode):
        super(FuseFastToSlow, self).__init__()
        self.mode = mode
        if mode == 'slowfast':
            slow_cfg, fast_cfg = cfg
            dim_in = fast_cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_idx]
            dim_out = fast_cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_idx] * fast_cfg.VIDEO.BACKBONE.SLOWFAST.CONV_CHANNEL_RATIO
            kernel_size = [fast_cfg.VIDEO.BACKBONE.SLOWFAST.KERNEL_SIZE, 1, 1]
            stride = [fast_cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA, 1, 1]
            padding = [fast_cfg.VIDEO.BACKBONE.SLOWFAST.KERNEL_SIZE // 2, 0, 0]
            bias = fast_cfg.VIDEO.BACKBONE.SLOWFAST.FUSION_CONV_BIAS
            self.conv_fast_to_slow = nn.Conv3d(dim_in, dim_out, kernel_size, stride, padding, bias=bias)
            if fast_cfg.VIDEO.BACKBONE.SLOWFAST.FUSION_BN:
                self.bn = nn.BatchNorm3d(dim_out, eps=fast_cfg.BN.EPS, momentum=fast_cfg.BN.MOMENTUM)
            if fast_cfg.VIDEO.BACKBONE.SLOWFAST.FUSION_RELU:
                self.relu = nn.ReLU(inplace=True)

    def forward(self, x_slow, x_fast):
        if self.mode == 'slowfast':
            fuse = self.conv_fast_to_slow(x_fast)
            if hasattr(self, 'bn'):
                fuse = self.bn(fuse)
            if hasattr(self, 'relu'):
                fuse = self.relu(fuse)
            return torch.cat((x_slow, fuse), 1), x_fast
        else:
            return x_slow, x_fast


class Slowfast(nn.Module):
    """
    Constructs SlowFast model.
    
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."

    Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py.
    """

    def __init__(self, cfg):
        super(Slowfast, self).__init__()
        self.mode = cfg.VIDEO.BACKBONE.SLOWFAST.MODE
        if self.mode == 'slowfast':
            self.slow_enable = True
            self.fast_enable = True
        elif self.mode == 'slowonly':
            self.slow_enable = True
            self.fast_enable = False
        elif self.mode == 'fastonly':
            self.slow_enable = False
            self.fast_enable = True
        self._construct_backbone(cfg)

    def _construct_slowfast_cfg(self, cfg):
        cfgs = []
        for i in range(2):
            pseudo_cfg = cfg.deep_copy()
            pseudo_cfg.VIDEO.BACKBONE.KERNEL_SIZE = pseudo_cfg.VIDEO.BACKBONE.KERNEL_SIZE[i]
            pseudo_cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK = pseudo_cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[i]
            if i == 1:
                pseudo_cfg.VIDEO.BACKBONE.ADD_FUSION_CHANNEL = False
                for idx, k in enumerate(pseudo_cfg.VIDEO.BACKBONE.NUM_FILTERS):
                    pseudo_cfg.VIDEO.BACKBONE.NUM_FILTERS[idx] = k // cfg.VIDEO.BACKBONE.SLOWFAST.BETA
            else:
                pseudo_cfg.VIDEO.BACKBONE.ADD_FUSION_CHANNEL = self.fast_enable
            cfgs.append(pseudo_cfg)
        return cfgs

    def _construct_slowfast_module(self, cfgs, module, **kwargs):
        modules = []
        for idx, cfg in enumerate(cfgs):
            if idx == 0 and self.slow_enable == True or idx == 1 and self.fast_enable == True:
                modules.append(module(cfg, **kwargs))
            else:
                modules.append(nn.Identity)
        return modules

    def _construct_backbone(self, cfg):
        cfgs = self._construct_slowfast_cfg(cfg)
        self.slow_conv1, self.fast_conv1 = self._construct_slowfast_module(cfgs, STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME))
        self.slowfast_fusion1 = FuseFastToSlow(cfgs, stage_idx=0, mode=self.mode)
        n1, n2, n3, n4 = _n_conv_resnet[cfg.VIDEO.BACKBONE.DEPTH]
        self.slow_conv2, self.fast_conv2 = self._construct_slowfast_module(cfgs, Base3DResStage, num_blocks=n1, stage_idx=1)
        self.slowfast_fusion2 = FuseFastToSlow(cfgs, stage_idx=1, mode=self.mode)
        self.slow_conv3, self.fast_conv3 = self._construct_slowfast_module(cfgs, Base3DResStage, num_blocks=n2, stage_idx=2)
        self.slowfast_fusion3 = FuseFastToSlow(cfgs, stage_idx=2, mode=self.mode)
        self.slow_conv4, self.fast_conv4 = self._construct_slowfast_module(cfgs, Base3DResStage, num_blocks=n3, stage_idx=3)
        self.slowfast_fusion4 = FuseFastToSlow(cfgs, stage_idx=3, mode=self.mode)
        self.slow_conv5, self.fast_conv5 = self._construct_slowfast_module(cfgs, Base3DResStage, num_blocks=n4, stage_idx=4)
        _init_convnet_weights(self)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['video']
        assert isinstance(x, list), 'Input to SlowFast should be lists'
        x_slow = x[0]
        x_fast = x[1]
        x_slow, x_fast = self.slow_conv1(x_slow), self.fast_conv1(x_fast)
        x_slow, x_fast = self.slowfast_fusion1(x_slow, x_fast)
        x_slow, x_fast = self.slow_conv2(x_slow), self.fast_conv2(x_fast)
        x_slow, x_fast = self.slowfast_fusion2(x_slow, x_fast)
        x_slow, x_fast = self.slow_conv3(x_slow), self.fast_conv3(x_fast)
        x_slow, x_fast = self.slowfast_fusion3(x_slow, x_fast)
        x_slow, x_fast = self.slow_conv4(x_slow), self.fast_conv4(x_fast)
        x_slow, x_fast = self.slowfast_fusion4(x_slow, x_fast)
        x_slow, x_fast = self.slow_conv5(x_slow), self.fast_conv5(x_fast)
        return x_slow, x_fast


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, ff_dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(ff_dropout), nn.Linear(dim * mult, dim), nn.Dropout(ff_dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Self-attention module. 
    Currently supports both full self-attention on all the input tokens,
    or only-spatial/only-temporal self-attention. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer.
    and 
    Gedas Bertasius, Heng Wang, Lorenzo Torresani.
    Is Space-Time Attention All You Need for Video Understanding?

    Modified from 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, dim, num_heads=12, attn_dropout=0.0, ff_dropout=0.0, einops_from=None, einops_to=None, **einops_dims):
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.ff_dropout = nn.Dropout(ff_dropout)
        if einops_from is not None and einops_to is not None:
            self.partial = True
            self.einops_from = einops_from
            self.einops_to = einops_to
            self.einops_dims = einops_dims
        else:
            self.partial = False

    def forward(self, x):
        if self.partial:
            return self.forward_partial(x, self.einops_from, self.einops_to, **self.einops_dims)
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

    def forward_partial(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q *= self.scale
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))
        cls_attn = (cls_q @ k.transpose(1, 2)).softmax(-1)
        cls_attn = self.attn_dropout(cls_attn)
        cls_out = cls_attn @ v
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))
        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)
        attn = (q_ @ k_.transpose(1, 2)).softmax(-1)
        attn = self.attn_dropout(attn)
        x = attn @ v_
        x = rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)
        x = torch.cat((cls_out, x), dim=1)
        x = rearrange(x, '(b h) n d -> b n (h d)', h=h)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x


class BaseTransformerLayer(nn.Module):

    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()
        dim = cfg.VIDEO.BACKBONE.NUM_FEATURES if cfg is not None else 768
        num_heads = cfg.VIDEO.BACKBONE.NUM_HEADS if cfg is not None else 1
        attn_dropout = cfg.VIDEO.BACKBONE.ATTN_DROPOUT if cfg is not None else 0.1
        ff_dropout = cfg.VIDEO.BACKBONE.FF_DROPOUT if cfg is not None else 0.1
        mlp_mult = cfg.VIDEO.BACKBONE.MLP_MULT if cfg is not None else 4
        drop_path = drop_path_rate
        self.norm = nn.LayerNorm(dim, eps=1e-06)
        self.attn = Attention(dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-06)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


class TimesformerLayer(nn.Module):

    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()
        image_size = cfg.DATA.TRAIN_CROP_SIZE if cfg is not None else 224
        num_frames = cfg.DATA.NUM_INPUT_FRAMES if cfg is not None else 8
        dim = cfg.VIDEO.BACKBONE.NUM_FEATURES if cfg is not None else 768
        num_heads = cfg.VIDEO.BACKBONE.NUM_HEADS if cfg is not None else 1
        attn_dropout = cfg.VIDEO.BACKBONE.ATTN_DROPOUT if cfg is not None else 0.1
        ff_dropout = cfg.VIDEO.BACKBONE.FF_DROPOUT if cfg is not None else 0.1
        patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE if cfg is not None else 16
        drop_path = drop_path_rate
        num_patches = (image_size // patch_size) ** 2
        self.norm_temporal = nn.LayerNorm(dim, eps=1e-06)
        self.attn_temporal = Attention(dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout, einops_from='b (f n) d', einops_to='(b n) f d', n=num_patches)
        self.norm = nn.LayerNorm(dim, eps=1e-06)
        self.attn = Attention(dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout, einops_from='b (f n) d', einops_to='(b f) n d', f=num_frames)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-06)
        self.ffn = FeedForward(dim=dim, ff_dropout=ff_dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn_temporal(self.norm_temporal(x)))
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


class PatchEmbedStem(nn.Module):
    """ 
    Video to Patch Embedding.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        image_size = cfg.DATA.TRAIN_CROP_SIZE if cfg is not None else 224
        channels = cfg.DATA.NUM_INPUT_CHANNELS if cfg is not None else 3
        num_frames = cfg.DATA.NUM_INPUT_FRAMES if cfg is not None else 16
        patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE if cfg is not None else 16
        dim = cfg.VIDEO.BACKBONE.NUM_FEATURES if cfg is not None else 768
        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=dim, kernel_size=[1, patch_size, patch_size], stride=[1, patch_size, patch_size])

    def forward(self, x):
        b, c, t, h, w, p = *x.shape, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _init_transformer_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class Transformer(nn.Module):

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        num_frames = cfg.DATA.NUM_INPUT_FRAMES if cfg is not None else 8
        image_size = cfg.DATA.TRAIN_CROP_SIZE if cfg is not None else 224
        num_features = cfg.VIDEO.BACKBONE.NUM_FEATURES if cfg is not None else 768
        patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE if cfg is not None else 16
        depth = cfg.VIDEO.BACKBONE.DEPTH if cfg is not None else 12
        drop_path = cfg.VIDEO.BACKBONE.DROP_PATH if cfg is not None else 16
        if hasattr(cfg.VIDEO.BACKBONE, 'TUBELET_SIZE'):
            tubelet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE if cfg is not None else 2
        else:
            tubelet_size = 1
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_per_frame = (image_size // patch_size) ** 2
        num_patches = num_frames * num_patches_per_frame // tubelet_size
        self.stem = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)
        self.pos_embd = nn.Parameter(torch.zeros(1, num_patches + 1, num_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.layers = nn.Sequential(*[BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, drop_path_rate=dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(num_features, eps=1e-06)
        trunc_normal_(self.pos_embd, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['video']
        x = self.stem(x)
        cls_token = self.cls_token.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embd
        x = self.layers(x)
        x = self.norm(x)
        return x[:, 0]


class FactorizedTransformer(nn.Module):
    """
    The factorized transformer. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer. 
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        num_frames = cfg.DATA.NUM_INPUT_FRAMES if cfg is not None else 8
        image_size = cfg.DATA.TRAIN_CROP_SIZE if cfg is not None else 224
        num_features = cfg.VIDEO.BACKBONE.NUM_FEATURES if cfg is not None else 768
        patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE if cfg is not None else 16
        depth = cfg.VIDEO.BACKBONE.DEPTH if cfg is not None else 12
        depth_temp = cfg.VIDEO.BACKBONE.DEPTH_TEMP if cfg is not None else 4
        drop_path = cfg.VIDEO.BACKBONE.DROP_PATH if cfg is not None else 16
        if hasattr(cfg.VIDEO.BACKBONE, 'TUBELET_SIZE'):
            tubelet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE if cfg is not None else 2
        else:
            tubelet_size = 1
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size
        self.stem = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)
        self.pos_embd = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.temp_embd = nn.Parameter(torch.zeros(1, num_frames // tubelet_size + 1, num_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))
        self.cls_token_out = nn.Parameter(torch.randn(1, 1, num_features))
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth + depth_temp)]
        self.layers = nn.Sequential(*[BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, drop_path_rate=dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(num_features, eps=1e-06)
        self.layers_temporal = nn.Sequential(*[BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, drop_path_rate=dpr[i + depth]) for i in range(depth_temp)])
        self.norm_out = nn.LayerNorm(num_features, eps=1e-06)
        trunc_normal_(self.pos_embd, std=0.02)
        trunc_normal_(self.temp_embd, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_token_out, std=0.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['video']
        h, w = x.shape[-2:]
        actual_num_patches_per_frame = h // self.patch_size * (w // self.patch_size)
        x = self.stem(x)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training
            x = rearrange(x, 'b (t n) c -> (b t) n c', n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, 'b (t n) c -> (b t) n c', n=self.num_patches_per_frame)
        cls_token = self.cls_token.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token, x), dim=1)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            actual_num_pathces_per_side = int(math.sqrt(actual_num_patches_per_frame))
            if not hasattr(self, 'new_pos_embd') or self.new_pos_embd.shape[1] != actual_num_pathces_per_side ** 2 + 1:
                cls_pos_embd = self.pos_embd[:, 0, :].unsqueeze(1)
                pos_embd = self.pos_embd[:, 1:, :]
                num_patches_per_side = int(math.sqrt(self.num_patches_per_frame))
                pos_embd = pos_embd.reshape(1, num_patches_per_side, num_patches_per_side, -1).permute(0, 3, 1, 2)
                pos_embd = torch.nn.functional.interpolate(pos_embd, size=(actual_num_pathces_per_side, actual_num_pathces_per_side), mode='bilinear').permute(0, 2, 3, 1).reshape(1, actual_num_pathces_per_side ** 2, -1)
                self.new_pos_embd = torch.cat((cls_pos_embd, pos_embd), dim=1)
            x += self.new_pos_embd
        else:
            x += self.pos_embd
        x = self.layers(x)
        x = self.norm(x)[:, 0]
        x = rearrange(x, '(b t) c -> b t c', t=self.num_patches // self.num_patches_per_frame)
        cls_token_out = self.cls_token_out.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token_out, x), dim=1)
        x += self.temp_embd
        x = self.layers_temporal(x)
        x = self.norm_out(x)
        return x[:, 0]


class CSNBranch(BaseBranch):
    """
    The ir-CSN branch.
    
    See Du Tran et al.
    Video Classification with Channel-Separated Convolutional Networks.
    """

    def __init__(self, cfg, block_idx):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(CSNBranch, self).__init__(cfg, block_idx)

    def _construct_bottleneck(self):
        self.a = nn.Conv3d(in_channels=self.dim_in, out_channels=self.num_filters // self.expansion_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters // self.expansion_ratio, kernel_size=self.kernel_size, stride=self.stride, padding=[self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False, groups=self.num_filters // self.expansion_ratio)
        self.b_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)
        self.c = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

    def forward(self, x):
        if self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)
            x = self.c(x)
            x = self.c_bn(x)
            return x


class NonLocal(BaseBranch):
    """
    Non-local block.
    
    See Xiaolong Wang et al.
    Non-local Neural Networks.
    """

    def __init__(self, cfg, block_idx):
        super(NonLocal, self).__init__(cfg, block_idx)
        self.dim_middle = self.dim_in // 2
        self.qconv = nn.Conv3d(self.dim_in, self.dim_middle, kernel_size=1, stride=1, padding=0)
        self.kconv = nn.Conv3d(self.dim_in, self.dim_middle, kernel_size=1, stride=1, padding=0)
        self.vconv = nn.Conv3d(self.dim_in, self.dim_middle, kernel_size=1, stride=1, padding=0)
        self.out_conv = nn.Conv3d(self.dim_middle, self.num_filters, kernel_size=1, stride=1, padding=0)
        self.out_bn = nn.BatchNorm3d(self.num_filters, eps=1e-05, momentum=self.bn_mmt)

    def forward(self, x):
        n, c, t, h, w = x.shape
        query = self.qconv(x).view(n, self.dim_middle, -1)
        key = self.kconv(x).view(n, self.dim_middle, -1)
        value = self.vconv(x).view(n, self.dim_middle, -1)
        attn = torch.einsum('nct,ncp->ntp', (query, key))
        attn = attn * self.dim_middle ** -0.5
        attn = F.softmax(attn, dim=2)
        out = torch.einsum('ntg,ncg->nct', (attn, value))
        out = out.view(n, self.dim_middle, t, h, w)
        out = self.out_conv(out)
        out = self.out_bn(out)
        return x + out


class R2D3DBranch(BaseBranch):
    """
    The R2D3D Branch. 

    Essentially the MCx model in 
    Du Tran et al.
    A Closer Look at Spatiotemporal Convoluitions for Action Recognition.

    The model is used in DPC, MemDPC for self-supervised video 
    representation learning.
    """

    def __init__(self, cfg, block_idx):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(R2D3DBranch, self).__init__(cfg, block_idx)

    def _construct_simple_block(self):
        self.a = nn.Conv3d(in_channels=self.dim_in, out_channels=self.num_filters, kernel_size=self.kernel_size, stride=self.stride, padding=[self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=self.kernel_size, stride=1, padding=[self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.b_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

    def _construct_bottleneck(self):
        self.a = nn.Conv3d(in_channels=self.dim_in, out_channels=self.num_filters // self.expansion_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters // self.expansion_ratio, kernel_size=self.kernel_size, stride=self.stride, padding=[self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.b_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)
        self.c = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

    def forward(self, x):
        if self.transformation == 'simple_block':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)
            x = self.b(x)
            x = self.b_bn(x)
            return x
        elif self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)
            x = self.c(x)
            x = self.c_bn(x)
            return x


class R2Plus1DBranch(BaseBranch):
    """
    The R(2+1)D Branch. 

    See Du Tran et al.
    A Closer Look at Spatiotemporal Convoluitions for Action Recognition.
    """

    def __init__(self, cfg, block_idx):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(R2Plus1DBranch, self).__init__(cfg, block_idx)

    def _construct_simple_block(self):
        mid_dim = int(math.floor(self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.dim_in * self.num_filters / (self.kernel_size[1] * self.kernel_size[2] * self.dim_in + self.kernel_size[0] * self.num_filters)))
        self.a1 = nn.Conv3d(in_channels=self.dim_in, out_channels=mid_dim, kernel_size=[1, self.kernel_size[1], self.kernel_size[2]], stride=[1, self.stride[1], self.stride[2]], padding=[0, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.a1_bn = nn.BatchNorm3d(mid_dim, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a1_relu = nn.ReLU(inplace=True)
        self.a2 = nn.Conv3d(in_channels=mid_dim, out_channels=self.num_filters, kernel_size=[self.kernel_size[0], 1, 1], stride=[self.stride[0], 1, 1], padding=[self.kernel_size[0] // 2, 0, 0], bias=False)
        self.a2_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a2_relu = nn.ReLU(inplace=True)
        mid_dim = int(math.floor(self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.num_filters * self.num_filters / (self.kernel_size[1] * self.kernel_size[2] * self.num_filters + self.kernel_size[0] * self.num_filters)))
        self.b1 = nn.Conv3d(in_channels=self.num_filters, out_channels=mid_dim, kernel_size=[1, self.kernel_size[1], self.kernel_size[2]], stride=1, padding=[0, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.b1_bn = nn.BatchNorm3d(mid_dim, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)
        self.b2 = nn.Conv3d(in_channels=mid_dim, out_channels=self.num_filters, kernel_size=[self.kernel_size[0], 1, 1], stride=1, padding=[self.kernel_size[0] // 2, 0, 0], bias=False)
        self.b2_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

    def _construct_bottleneck(self):
        self.a = nn.Conv3d(in_channels=self.dim_in, out_channels=self.num_filters // self.expansion_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
        self.b1 = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters // self.expansion_ratio, kernel_size=[1, self.kernel_size[1], self.kernel_size[2]], stride=[1, self.stride[1], self.stride[2]], padding=[0, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.b1_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)
        self.b2 = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters // self.expansion_ratio, kernel_size=[self.kernel_size[0], 1, 1], stride=[self.stride[0], 1, 1], padding=[self.kernel_size[0] // 2, 0, 0], bias=False)
        self.b2_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)
        self.c = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

    def forward(self, x):
        if self.transformation == 'simple_block':
            x = self.a1(x)
            x = self.a1_bn(x)
            x = self.a1_relu(x)
            x = self.a2(x)
            x = self.a2_bn(x)
            x = self.a2_relu(x)
            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)
            x = self.b2(x)
            x = self.b2_bn(x)
            return x
        elif self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)
            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)
            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)
            x = self.c(x)
            x = self.c_bn(x)
            return x


class STConv3d(nn.Module):
    """
    Element constructing the S3D/S3DG.
    See models/base/backbone.py L99-186.

    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """

    def __init__(self, cfg, in_planes, out_planes, kernel_size, stride, padding=0):
        super(STConv3d, self).__init__()
        if isinstance(stride, tuple):
            t_stride = stride[0]
            stride = stride[-1]
        else:
            t_stride = stride
        self.bn_mmt = cfg.BN.MOMENTUM
        self.bn_eps = cfg.BN.EPS
        self._construct_branch(cfg, in_planes, out_planes, kernel_size, stride, t_stride, padding)

    def _construct_branch(self, cfg, in_planes, out_planes, kernel_size, stride, t_stride, padding=0):
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, padding, padding), bias=False)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size, 1, 1), stride=(t_stride, 1, 1), padding=(padding, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes, eps=self.bn_eps, momentum=self.bn_mmt)
        self.bn2 = nn.BatchNorm3d(out_planes, eps=self.bn_eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.conv1.weight.data.normal_(mean=0, std=0.01)
        self.conv2.weight.data.normal_(mean=0, std=0.01)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class SlowfastBranch(BaseBranch):
    """
    Constructs SlowFast conv branch.

    See Christoph Feichtenhofer et al.
    SlowFast Networks for Video Recognition.
    """

    def __init__(self, cfg, block_idx):
        super(SlowfastBranch, self).__init__(cfg, block_idx)

    def _construct_simple_block(self):
        self.a = nn.Conv3d(in_channels=self.dim_in, out_channels=self.num_filters, kernel_size=self.kernel_size, stride=self.stride, padding=[self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=self.kernel_size, stride=1, padding=[self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.b_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_bn.transform_final_bn = True

    def _construct_bottleneck(self):
        self.a = nn.Conv3d(in_channels=self.dim_in, out_channels=self.num_filters // self.expansion_ratio, kernel_size=[3, 1, 1] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 1, stride=1, padding=[1, 0, 0] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 0, bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters // self.expansion_ratio, kernel_size=self.kernel_size, stride=self.stride, padding=[self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.b_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)
        self.c = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        if self.transformation == 'simple_block':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)
            x = self.b(x)
            x = self.b_bn(x)
            return x
        elif self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)
            x = self.c(x)
            x = self.c_bn(x)
            return x


class RouteFuncMLP(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in, ratio, kernels, bn_eps=1e-05, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=1, padding=0)
        self.a = nn.Conv3d(in_channels=c_in, out_channels=int(c_in // ratio), kernel_size=[kernels[0], 1, 1], padding=[kernels[0] // 2, 0, 0])
        self.bn = nn.BatchNorm3d(int(c_in // ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(in_channels=int(c_in // ratio), out_channels=c_in, kernel_size=[kernels[1], 1, 1], padding=[kernels[1] // 2, 0, 0], bias=False)
        self.b.skip_init = True
        self.b.weight.data.zero_()

    def forward(self, x):
        g = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(g))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x


class TAdaConv2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d 
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, cal_dim='cin'):
        super(TAdaConv2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv2d.
            padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d. 
            bias (bool): whether to use bias in TAdaConv2d.
        """
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ['cin', 'cout']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim
        self.weight = nn.Parameter(torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(1, -1, h, w)
        if self.cal_dim == 'cin':
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2) * self.weight).reshape(-1, c_in // self.groups, kh, kw)
        elif self.cal_dim == 'cout':
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(3) * self.weight).reshape(-1, c_in // self.groups, kh, kw)
        bias = None
        if self.bias is not None:
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:], dilation=self.dilation[1:], groups=self.groups * b * t)
        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0, 2, 1, 3, 4)
        return output

    def __repr__(self):
        return f'TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, ' + f'stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim="{self.cal_dim}")'


class TAdaConvBlockAvgPool(BaseBranch):
    """
    The TAdaConv branch with average pooling as the feature aggregation scheme.

    For details, see
    Ziyuan Huang, Shiwei Zhang, Liang Pan, Zhiwu Qing, Mingqian Tang, Ziwei Liu, and Marcelo H. Ang Jr.
    "TAda! Temporally-Adaptive Convolutions for Video Understanding."
    
    """

    def __init__(self, cfg, block_idx):
        super(TAdaConvBlockAvgPool, self).__init__(cfg, block_idx, construct_branch=False)
        self._construct_branch()

    def _construct_bottleneck(self):
        self.a = nn.Conv3d(in_channels=self.dim_in, out_channels=self.num_filters // self.expansion_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = TAdaConv2d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters // self.expansion_ratio, kernel_size=[1, self.kernel_size[1], self.kernel_size[2]], stride=[1, self.stride[1], self.stride[2]], padding=[0, self.kernel_size[1] // 2, self.kernel_size[2] // 2], bias=False)
        self.b_rf = RouteFuncMLP(c_in=self.num_filters // self.expansion_ratio, ratio=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_R, kernels=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_K)
        self.b_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_avgpool = nn.AvgPool3d(kernel_size=[self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[0], self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[1], self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[2]], stride=1, padding=[self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[0] // 2, self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[1] // 2, self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[2] // 2])
        self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_avgpool_bn.skip_init = True
        self.b_avgpool_bn.weight.data.zero_()
        self.b_avgpool_bn.bias.data.zero_()
        self.b_relu = nn.ReLU(inplace=True)
        self.c = nn.Conv3d(in_channels=self.num_filters // self.expansion_ratio, out_channels=self.num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

    def forward(self, x):
        if self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)
            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            x = self.b_relu(x)
            x = self.c(x)
            x = self.c_bn(x)
            return x


class BaseBMN(nn.Module):
    """
    Head for predicting boundary matching map.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): the global config object. 
        """
        super(BaseBMN, self).__init__()
        self.cfg = cfg
        self.tscale = cfg.DATA.TEMPORAL_SCALE
        self.dscale = cfg.DATA.DURATION_SCALE if cfg.DATA.DURATION_SCALE > 0 else cfg.DATA.TEMPORAL_SCALE
        self.num_sample = cfg.VIDEO.HEAD.NUM_SAMPLE
        self.num_sample_perbin = cfg.VIDEO.HEAD.NUM_SAMPLE_PERBIN
        self.hidden_dim_1d = cfg.VIDEO.DIM1D
        self.hidden_dim_2d = cfg.VIDEO.DIM2D
        self.hidden_dim_3d = cfg.VIDEO.DIM3D
        self.prop_boundary_ratio = cfg.VIDEO.HEAD.BOUNDARY_RATIO
        self._construct_head()

    def _construct_head(self):
        self.sample_mask = nn.Parameter(self.get_interp1d_mask(self.prop_boundary_ratio, self.num_sample), requires_grad=False)
        self.x_1d_s = nn.Sequential(nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4), nn.ReLU(inplace=True), nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1), nn.Sigmoid())
        self.x_1d_e = nn.Sequential(nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4), nn.ReLU(inplace=True), nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1), nn.Sigmoid())
        self.x_1d_p = nn.Sequential(nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.x_3d_p = nn.Sequential(nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1), stride=(self.num_sample, 1, 1)), nn.ReLU(inplace=True))
        self.x_2d_p = nn.Sequential(nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1), nn.Sigmoid())
        if self.cfg.VIDEO.HEAD.USE_BMN_REGRESSION:
            self.x_2d_r = nn.Sequential(nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1))
        if type(self.cfg.VIDEO.HEAD.NUM_CLASSES) is list:
            self.x_2d_verb = nn.Sequential(nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.cfg.VIDEO.HEAD.NUM_CLASSES[0], kernel_size=1))
            self.x_2d_noun = nn.Sequential(nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.cfg.VIDEO.HEAD.NUM_CLASSES[1], kernel_size=1))

    def forward(self, x):
        """
        Args:
            x (dict): {
                "video" (tensor): Features for sliding windows.
            }
        Returns:
            output (dict): {
                confidence_map: (tensor),
                 start_map: (tensor),
                 end_map: (tensor),
                 reg_map: (tensor),
                 verb_map: (tensor),
                 noun_map: (tensor)
            } 
        """
        base_feature = x['video']
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        mid_feature = self.x_1d_p(base_feature)
        mid_feature = self._boundary_matching_layer(mid_feature)
        mid_feature = self.x_3d_p(mid_feature).squeeze(2)
        confidence_map = self.x_2d_p(mid_feature)
        if self.cfg.VIDEO.HEAD.USE_BMN_REGRESSION:
            reg_map = self.x_2d_r(mid_feature)
        else:
            reg_map = {}
        if hasattr(self, 'x_2d_verb'):
            verb_map = self.x_2d_verb(mid_feature)
            noun_map = self.x_2d_noun(mid_feature)
        else:
            verb_map, noun_map = {}, {}
        output = {'confidence_map': confidence_map, 'start': start, 'end': end, 'reg_map': reg_map, 'verb_map': verb_map, 'noun_map': noun_map}
        return output, {}

    def _boundary_matching_layer(self, x):
        """
        Apply boundary mathcing operation for input feature
        Args:
            x (tensor): 1D feature for boundary mathcing operation.
        Returns:
            output (Tensor): matched features for proposals
        """
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0], input_size[1], self.num_sample, self.dscale, self.tscale)
        return out

    def get_interp1d_mask(self, prop_boundary_ratio, num_sample):
        """
        generate sample mask for each point in Boundary-Matching Map
        Args:
            prop_boundary_ratio (float): Boundary expand ratio.
            num_sample (int): The number of sample points for each proposal.
        Returns:
            output (Tensor): sample mask
        """
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.dscale):
                if start_index + duration_index < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(sample_xmin, sample_xmax, self.tscale, num_sample, self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        return torch.Tensor(mask_mat).view(self.tscale, -1)

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        """
        generate sample mask for a boundary-matching pair
        Args:
            seg_xmin (float): Start time of the proposal.
            seg_xmax (float): End time of the proposal.
            tscale (int): Temporal len for bmn.
            num_sample (int): The number of sample points for each proposal.
            num_sample_perbin (int): The number of sample points for each bin.
        Returns:
            output (Tensor): one sample mask
        """
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [(seg_xmin + plen_sample * ii) for ii in range(num_sample * num_sample_perbin)]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= tscale - 1 and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= tscale - 1 and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask


class MoSIHeadOnlyX(BaseHead):
    """
    Head for predicting MoSI categories.
    This head performs prediction on only x-axis.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): the global config object. 
        """
        self.num_classes = cfg.VIDEO.HEAD.NUM_CLASSES - 1 + 1 * (not cfg.PRETRAIN.ZERO_OUT)
        super(MoSIHeadOnlyX, self).__init__(cfg)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.out_x = nn.Linear(dim, self.num_classes, bias=True)
        if activation_func == 'softmax':
            self.activation = nn.Softmax(dim=4)
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(activation_func))

    def forward(self, x):
        """
        Returns:
            x (dict): dictionary with keys "move_x", indicating the category
                prediction on the x-axis.
            logits (Tensor): global average pooled features.
        """
        out = {}
        x = self.global_avg_pool(x)
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        out['move_x'] = self.out_x(x)
        if not self.training:
            out['move_x'] = self.activation(out['move_x'])
            out['move_x'] = out['move_x'].mean([1, 2, 3])
        out['move_x'] = out['move_x'].view(out['move_x'].shape[0], -1)
        return out, x.view(x.shape[0], -1)


class MoSIHeadOnlyY(BaseHead):
    """
    Head for predicting MoSI categories.
    This head performs prediction on only y-axis.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): the global config object. 
        """
        self.num_classes = cfg.VIDEO.HEAD.NUM_CLASSES - 1 + 1 * (not cfg.PRETRAIN.ZERO_OUT)
        super(MoSIHeadOnlyY, self).__init__(cfg)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.out_x = nn.Linear(dim, self.num_classes, bias=True)
        if activation_func == 'softmax':
            self.activation = nn.Softmax(dim=4)
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(activation_func))

    def forward(self, x):
        """
        Returns:
            x (dict): dictionary with keys "move_y", indicating the category
                prediction on the y-axis.
            logits (Tensor): global average pooled features.
        """
        out = {}
        x = self.global_avg_pool(x)
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        out['move_y'] = self.out_x(x)
        if not self.training:
            out['move_y'] = self.activation(out['move_y'])
            out['move_y'] = out['move_y'].mean([1, 2, 3])
        out['move_y'] = out['move_y'].view(out['move_y'].shape[0], -1)
        return out, x.view(x.shape[0], -1)


class MoSIHeadJoint(BaseHead):
    """
    Head for predicting MoSI categories.
    This head performs prediction jointly on both axes.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): the global config object. 
        """
        if cfg.PRETRAIN.DECOUPLE:
            self.num_classes = len(cfg.PRETRAIN.DATA_MODE) * (cfg.VIDEO.HEAD.NUM_CLASSES - 1) + 1 * (not cfg.PRETRAIN.ZERO_OUT)
        else:
            self.num_classes = cfg.VIDEO.HEAD.NUM_CLASSES ** len(cfg.PRETRAIN.DATA_MODE)
        super(MoSIHeadJoint, self).__init__(cfg)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.out_joint = nn.Linear(dim, self.num_classes, bias=True)
        if activation_func == 'softmax':
            self.activation = nn.Softmax(dim=4)
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(activation_func))

    def forward(self, x):
        """
        Returns:
            x (Tensor): joint prediction on both axes.
            logits (Tensor): global average pooled features.
        """
        x = self.global_avg_pool(x)
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        out = self.out_joint(x)
        if not self.training:
            out = self.activation(out)
            out = out.mean([1, 2, 3])
        out = out.view(out.shape[0], -1)
        return out, x.view(x.shape[0], -1)


class SlowFastHead(nn.Module):
    """
    Constructs head for the SlowFast Networks. 
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(SlowFastHead, self).__init__()
        self.cfg = cfg
        self.mode = cfg.VIDEO.BACKBONE.SLOWFAST.MODE
        dim = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        if self.mode == 'slowfast':
            dim = dim + dim // cfg.VIDEO.BACKBONE.SLOWFAST.BETA
        elif self.mode == 'fastonly':
            dim = dim // cfg.VIDEO.BACKBONE.SLOWFAST.BETA
        elif self.mode == 'slowonly':
            pass
        else:
            raise NotImplementedError('Mode {} not supported.'.format(self.mode))
        num_classes = cfg.VIDEO.HEAD.NUM_CLASSES
        dropout_rate = cfg.VIDEO.HEAD.DROPOUT_RATE
        activation_func = cfg.VIDEO.HEAD.ACTIVATION
        self._construct_head(dim, num_classes, dropout_rate, activation_func)
        _init_convnet_weights(self)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(dim, num_classes, bias=True)
        if activation_func == 'softmax':
            self.activation = nn.Softmax(dim=4)
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(activation_func))

    def forward(self, x):
        """
        Returns:
            x (Tensor): classification predictions.
            logits (Tensor): global average pooled features.
        """
        if self.mode == 'slowfast':
            x = torch.cat((self.global_avg_pool(x[0]), self.global_avg_pool(x[1])), dim=1)
        elif self.mode == 'slowonly':
            x = self.global_avg_pool(x[0])
        elif self.mode == 'fastonly':
            x = self.global_avg_pool(x[1])
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            out = self.dropout(x)
        else:
            out = x
        out = self.out(out)
        if not self.training:
            out = self.activation(out)
            out = out.mean([1, 2, 3])
        out = out.view(out.shape[0], -1)
        return out, x.view(x.shape[0], -1)


class SlowFastHeadx2(nn.Module):
    """
    SlowFast Head for EPIC-KITCHENS dataset.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(SlowFastHeadx2, self).__init__()
        self.cfg = cfg
        self.mode = cfg.VIDEO.BACKBONE.SLOWFAST.MODE
        dim = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        if self.mode == 'slowfast':
            dim = dim + dim // cfg.VIDEO.BACKBONE.SLOWFAST.BETA
        elif self.mode == 'fastonly':
            dim = dim // cfg.VIDEO.BACKBONE.SLOWFAST.BETA
        elif self.mode == 'slowonly':
            pass
        else:
            raise NotImplementedError('Mode {} not supported.'.format(self.mode))
        num_classes = cfg.VIDEO.HEAD.NUM_CLASSES
        dropout_rate = cfg.VIDEO.HEAD.DROPOUT_RATE
        activation_func = cfg.VIDEO.HEAD.ACTIVATION
        self._construct_head(dim, num_classes, dropout_rate, activation_func)
        _init_convnet_weights(self)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.out1 = nn.Linear(dim, num_classes[0], bias=True)
        self.out2 = nn.Linear(dim, num_classes[1], bias=True)
        if activation_func == 'softmax':
            self.activation = nn.Softmax(dim=4)
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(activation_func))

    def forward(self, x):
        """
        Returns:
            x (dict): dictionary of classification predictions,
                with keys "verb_class" and "noun_class" indicating
                the predictions on the verb and noun.
            logits (Tensor): global average pooled features.
        """
        if self.mode == 'slowfast':
            x = torch.cat((self.global_avg_pool(x[0]), self.global_avg_pool(x[1])), dim=1)
        elif self.mode == 'slowonly':
            x = self.global_avg_pool(x[0])
        elif self.mode == 'fastonly':
            x = self.global_avg_pool(x[1])
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        out1 = self.out1(x)
        out2 = self.out2(x)
        if not self.training:
            out1 = self.activation(out1)
            out1 = out1.mean([1, 2, 3])
            out2 = self.activation(out2)
            out2 = out2.mean([1, 2, 3])
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        return {'verb_class': out1, 'noun_class': out2}, x.view(x.shape[0], -1)


class TransformerHead(BaseHead):
    """
    Construct head for video vision transformers.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(TransformerHead, self).__init__(cfg)
        self.apply(_init_transformer_weights)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        if self.cfg.VIDEO.HEAD.PRE_LOGITS:
            self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(dim, dim)), ('act', nn.Tanh())]))
        self.linear = nn.Linear(dim, num_classes)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        if activation_func == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_func == 'identity':
            self.activation = nn.Identity()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(activation_func))

    def forward(self, x):
        """
        Returns:
            x (Tensor): classification predictions.
            logits (Tensor): global average pooled features.
        """
        if hasattr(self, 'dropout'):
            out = self.dropout(x)
        else:
            out = x
        if hasattr(self, 'pre_logits'):
            out = self.pre_logits(out)
        out = self.linear(out)
        if not self.training:
            out = self.activation(out)
        return out, x


class TransformerHeadx2(BaseHead):
    """
    The Transformer head for EPIC-KITCHENS dataset.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(TransformerHeadx2, self).__init__(cfg)
        self.apply(_init_transformer_weights)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        if self.cfg.VIDEO.HEAD.PRE_LOGITS:
            self.pre_logits1 = nn.Sequential(OrderedDict([('fc', nn.Linear(dim, dim)), ('act', nn.Tanh())]))
            self.pre_logits2 = nn.Sequential(OrderedDict([('fc', nn.Linear(dim, dim)), ('act', nn.Tanh())]))
        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        if activation_func == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_func == 'identity':
            self.activation = nn.Identity()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(activation_func))

    def forward(self, x):
        """
        Returns:
            x (dict): dictionary of classification predictions,
                with keys "verb_class" and "noun_class" indicating
                the predictions on the verb and noun.
            logits (Tensor): global average pooled features.
        """
        if hasattr(self, 'dropout'):
            out1 = self.dropout(x)
            out2 = self.dropout(x)
        else:
            out1 = x
            out2 = x
        if hasattr(self, 'pre_logits1'):
            out1 = self.pre_logits1(out1)
            out2 = self.pre_logits2(out2)
        out1 = self.linear1(out1)
        out2 = self.linear2(out2)
        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        return {'verb_class': out1, 'noun_class': out2}, x


class TAdaConv3d(nn.Module):
    """
    Performs temporally adaptive 3D convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, cal_dim='cin'):
        super(TAdaConv3d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv3d.
            padding (list): padding for the convolution in TAdaConv3d.
            dilation (list): dilation of the convolution in TAdaConv3d.
            groups (int): number of groups for TAdaConv3d. 
            bias (bool): whether to use bias in TAdaConv3d.
        """
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        assert stride[0] == 1
        assert dilation[0] == 1
        assert cal_dim in ['cin', 'cout']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim
        self.weight = nn.Parameter(torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _, _, c_out, c_in, kt, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, kt // 2, kt // 2), 'constant', 0).unfold(dimension=2, size=kt, step=1).permute(0, 2, 1, 5, 3, 4).reshape(1, -1, kt, h, w)
        if self.cal_dim == 'cin':
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2).unsqueeze(-1) * self.weight).reshape(-1, c_in // self.groups, kt, kh, kw)
        elif self.cal_dim == 'cout':
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(3).unsqueeze(-1) * self.weight).reshape(-1, c_in // self.groups, kt, kh, kw)
        bias = None
        if self.bias is not None:
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv3d(x, weight=weight, bias=bias, stride=[1] + list(self.stride[1:]), padding=[0] + list(self.padding[1:]), dilation=[1] + list(self.dilation[1:]), groups=self.groups * b * t)
        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0, 2, 1, 3, 4)
        return output

    def __repr__(self):
        return f'TAdaConv3d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, ' + f'stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim="{self.cal_dim}")'


class DownSampleStem(Base3DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """

    def __init__(self, cfg):
        super(DownSampleStem, self).__init__(cfg)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.maxpool(x)
        return x


class TubeletEmbeddingStem(nn.Module):
    """ 
    Video to Tubelet Embedding.
    """

    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        image_size = cfg.DATA.TRAIN_CROP_SIZE if cfg is not None else 224
        channels = cfg.DATA.NUM_INPUT_CHANNELS if cfg is not None else 3
        num_frames = cfg.DATA.NUM_INPUT_FRAMES if cfg is not None else 16
        patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE if cfg is not None else 16
        dim = cfg.VIDEO.BACKBONE.NUM_FEATURES if cfg is not None else 768
        tubelet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE if cfg is not None else 2
        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=dim, kernel_size=[tubelet_size, patch_size, patch_size], stride=[tubelet_size, patch_size, patch_size])

    def forward(self, x):
        b, c, t, h, w, p = *x.shape, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        return x


class R2Plus1DStem(Base3DStem):
    """
    R(2+1)D Stem.
    """

    def __init__(self, cfg):
        super(R2Plus1DStem, self).__init__(cfg)

    def _construct_block(self, cfg, dim_in, num_filters, kernel_sz, stride, bn_eps=1e-05, bn_mmt=0.1):
        mid_dim = int(math.floor(kernel_sz[0] * kernel_sz[1] * kernel_sz[2] * dim_in * num_filters / (kernel_sz[1] * kernel_sz[2] * dim_in + kernel_sz[0] * num_filters)))
        self.a1 = nn.Conv3d(in_channels=dim_in, out_channels=mid_dim, kernel_size=[1, kernel_sz[1], kernel_sz[2]], stride=[1, stride[1], stride[2]], padding=[0, kernel_sz[1] // 2, kernel_sz[2] // 2], bias=False)
        self.a1_bn = nn.BatchNorm3d(mid_dim, eps=bn_eps, momentum=bn_mmt)
        self.a1_relu = nn.ReLU(inplace=True)
        self.a2 = nn.Conv3d(in_channels=mid_dim, out_channels=num_filters, kernel_size=[kernel_sz[0], 1, 1], stride=[stride[0], 1, 1], padding=[kernel_sz[0] // 2, 0, 0], bias=False)
        self.a2_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a2_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.a1(x)
        x = self.a1_bn(x)
        x = self.a1_relu(x)
        x = self.a2(x)
        x = self.a2_bn(x)
        x = self.a2_relu(x)
        return x


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, reduction=None):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FuseFastToSlow,
     lambda: ([], {'cfg': _mock_config(), 'stage_idx': 4, 'mode': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SelfGating,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_alibaba_mmai_research_TAdaConv(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

