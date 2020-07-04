import sys
_module = sys.modules[__name__]
del sys
deeplabv3 = _module
deeplabv3plus = _module
fpn = _module
pspnet = _module
unet = _module
decode = _module
encode_voc12 = _module
encode_voc12_aug = _module
test = _module
trainval = _module
vedaseg = _module
assembler = _module
criteria = _module
builder = _module
registry = _module
seg_wrapper = _module
dataloaders = _module
steel = _module
datasets = _module
base = _module
coil = _module
dummy = _module
transforms = _module
transforms = _module
voc = _module
loggers = _module
lr_schedulers = _module
poly_lr = _module
models = _module
builder = _module
decoders = _module
bricks = _module
builder = _module
gfpn = _module
gfpn = _module
encoders = _module
backbones = _module
resnet = _module
builder = _module
enhance_modules = _module
aspp = _module
ppm = _module
heads = _module
head = _module
utils = _module
act = _module
builder = _module
conv_module = _module
norm = _module
upsample = _module
weight_init = _module
optims = _module
runner = _module
runner = _module
checkpoint = _module
common = _module
config = _module
metrics = _module
misc = _module
path = _module
registry = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch.nn.functional as F


import random


import numpy as np


import torch


import torchvision.transforms as tt


import math


import copy


import logging


from torchvision.models.resnet import model_urls


from functools import partial


from torch.nn.parameter import Parameter


import warnings


from collections.abc import Iterable


import time


from collections import OrderedDict


import torchvision


from torch.utils import model_zoo


import inspect


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


CRITERIA = Registry('criterion')


def obj_from_dict_module(info, parent=None, default_args=None):
    """Initialize an object from dict.
    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.
    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.
    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.
            format(type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def obj_from_dict_registry(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(obj_type,
                registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.
            format(type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)


def build_from_cfg(cfg, parent, default_args=None, src='registry'):
    if src == 'registry':
        return obj_from_dict_registry(cfg, parent, default_args)
    elif src == 'module':
        return obj_from_dict_module(cfg, parent, default_args)
    else:
        raise ValueError('Method %s is not supported' % src)


class CriterionWrapper(nn.Module):
    """LossWrapper

        Args:
    """

    def __init__(self, cfg):
        super().__init__()
        self.criterion = build_from_cfg(cfg, CRITERIA, src='registry')

    def forward(self, pred, target):
        pred = F.interpolate(pred, target.shape[2:])
        return self.criterion(pred, target)


BRICKS = Registry('brick')


UTILS = Registry('utils')


def build_module(cfg, default_args=None):
    util = build_from_cfg(cfg, UTILS, default_args)
    return util


@BRICKS.register_module
class JunctionBlock(nn.Module):
    """JunctionBlock

    Args:
    """

    def __init__(self, top_down, lateral, post, to_layer, fusion_method=None):
        super().__init__()
        self.from_layer = {}
        self.to_layer = to_layer
        top_down_ = copy.copy(top_down)
        lateral_ = copy.copy(lateral)
        self.fusion_method = fusion_method
        self.top_down_block = []
        if top_down_:
            self.from_layer['top_down'] = top_down_.pop('from_layer')
            if 'trans' in top_down_:
                self.top_down_block.append(build_module(top_down_['trans']))
            self.top_down_block.append(build_module(top_down_['upsample']))
        self.top_down_block = nn.Sequential(*self.top_down_block)
        if lateral_:
            self.from_layer['lateral'] = lateral_.pop('from_layer')
            if lateral_:
                self.lateral_block = build_module(lateral_)
            else:
                self.lateral_block = nn.Sequential()
        else:
            self.lateral_block = nn.Sequential()
        if post:
            self.post_block = build_module(post)
        else:
            self.post_block = nn.Sequential()

    def forward(self, top_down=None, lateral=None):
        if top_down is not None:
            top_down = self.top_down_block(top_down)
        if lateral is not None:
            lateral = self.lateral_block(lateral)
        if top_down is not None:
            if lateral is not None:
                assert self.fusion_method in ('concat', 'add')
                if self.fusion_method == 'concat':
                    feat = torch.cat([top_down, lateral], 1)
                elif self.fusion_method == 'add':
                    feat = top_down + lateral
            else:
                assert self.fusion_method is None
                feat = top_down
        else:
            assert self.fusion_method is None
            if lateral is not None:
                feat = lateral
            else:
                raise ValueError(
                    'There is neither top down feature nor lateral feature')
        feat = self.post_block(feat)
        return feat


@BRICKS.register_module
class FusionBlock(nn.Module):
    """FusionBlock

        Args:
    """

    def __init__(self, method, from_layers, feat_strides, in_channels_list,
        out_channels_list, upsample, conv_cfg=dict(type='Conv'), norm_cfg=
        dict(type='BN'), act_cfg=dict(type='Relu', inplace=True),
        common_stride=4):
        super().__init__()
        assert method in ('add', 'concat')
        self.method = method
        self.from_layers = from_layers
        assert len(in_channels_list) == len(out_channels_list)
        self.blocks = nn.ModuleList()
        for idx in range(len(from_layers)):
            in_channels = in_channels_list[idx]
            out_channels = out_channels_list[idx]
            from_layer = from_layers[idx]
            feat_stride = feat_strides[idx]
            ups_num = int(max(1, math.log2(feat_stride) - math.log2(
                common_stride)))
            head_ops = []
            for idx2 in range(ups_num):
                cur_in_channels = in_channels if idx2 == 0 else out_channels
                conv = ConvModule(cur_in_channels, out_channels,
                    kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=
                    norm_cfg, act_cfg=act_cfg)
                head_ops.append(conv)
                if int(feat_stride) != int(common_stride):
                    head_ops.append(build_module(upsample))
            self.blocks.append(nn.Sequential(*head_ops))

    def forward(self, feats):
        outs = []
        for idx, key in enumerate(self.from_layers):
            block = self.blocks[idx]
            feat = feats[key]
            out = block(feat)
            outs.append(out)
        if self.method == 'add':
            res = torch.stack(outs, 0).sum(0)
        else:
            res = torch.cat(outs, 1)
        return res


@BRICKS.register_module
class CollectBlock(nn.Module):
    """FusionBlock

        Args:
    """

    def __init__(self, from_layer, to_layer=None):
        super().__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer

    def forward(self, feats):
        if self.to_layer is None:
            return feats[self.from_layer]
        else:
            res[self.to_layer] = feats[self.from_layer]


DECODERS = Registry('decoder')


def build_brick(cfg, default_args=None):
    brick = build_from_cfg(cfg, BRICKS, default_args)
    return brick


def build_bricks(cfgs):
    bricks = nn.ModuleList()
    for brick_cfg in cfgs:
        bricks.append(build_brick(brick_cfg))
    return bricks


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0,
    distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode,
            nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity
            =nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            kaiming_init(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            constant_init(m, 1)


logger = logging.getLogger()


@DECODERS.register_module
class GFPN(nn.Module):
    """GFPN

    Args:
    """

    def __init__(self, neck, fusion=None):
        super().__init__()
        self.neck = build_bricks(neck)
        if fusion:
            self.fusion = build_brick(fusion)
        else:
            self.fusion = None
        logger.info('GFPN init weights')
        init_weights(self.modules())

    def forward(self, bottom_up):
        x = None
        feats = {}
        for ii, layer in enumerate(self.neck):
            top_down_from_layer = layer.from_layer.get('top_down')
            lateral_from_layer = layer.from_layer.get('lateral')
            if lateral_from_layer:
                ll = bottom_up[lateral_from_layer]
            else:
                ll = None
            if top_down_from_layer is None:
                td = None
            elif 'c' in top_down_from_layer:
                td = bottom_up[top_down_from_layer]
            elif 'p' in top_down_from_layer:
                td = feats[top_down_from_layer]
            else:
                raise ValueError('Key error')
            x = layer(td, ll)
            feats[layer.to_layer] = x
        if self.fusion:
            x = self.fusion(feats)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_layer, act_layer, stride=1,
        downsample=None, groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = act_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.relu2 = act_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer, act_layer, stride=1,
        downsample=None, groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = act_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = act_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = act_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


def build_act_layer(cfg, num_features, postfix='', layer_only=False):
    """ Build activate layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify activate layer type.
            layer args: args needed to instantiate a activate layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into act abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created act layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in act_cfg:
        raise KeyError('Unrecognized activate type {}'.format(layer_type))
    else:
        abbr, act_layer = act_cfg[layer_type]
        if act_layer is None:
            raise NotImplementedError
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    if layer_type != 'Tlu':
        layer = act_layer(**cfg_)
    else:
        layer = act_layer(num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    if layer_only:
        return layer
    else:
        return name, layer


def build_norm_layer(cfg, num_features, postfix='', layer_only=False):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    if layer_only:
        return layer
    return name, layer


class ResNetCls(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False, groups=1, width_per_group=64, replace_stride_with_dilation=
        None, multi_grid=None, norm_cfg=None, act_cfg=None):
        super(ResNetCls, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        self._norm_layer = partial(build_norm_layer, norm_cfg, layer_only=True)
        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)
        self._act_layer = partial(build_act_layer, act_cfg, layer_only=True)
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu1 = self._act_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2], multi_grid=multi_grid)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
        multi_grid=None):
        norm_layer = self._norm_layer
        act_layer = self._act_layer
        downsample = None
        previous_dilation = self.dilation
        if multi_grid is None:
            multi_grid = [(1) for _ in range(blocks)]
        else:
            assert len(multi_grid) == blocks
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, norm_layer, act_layer,
            stride, downsample, self.groups, self.base_width, 
            previous_dilation * multi_grid[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=
                norm_layer, act_layer=act_layer, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation *
                multi_grid[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation, norm_layer,
        act_layer):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
            dilation=dilation, bias=False), norm_layer(out_channels),
            act_layer(out_channels)]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_layer, act_layer):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.
            Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(
            out_channels), act_layer(out_channels))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


ENHANCE_MODULES = Registry('enhance_module')


@ENHANCE_MODULES.register_module
class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, atrous_rates, from_layer,
        to_layer, dropout=None, norm_cfg=None, act_cfg=None):
        super(ASPP, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        norm_layer = partial(build_norm_layer, norm_cfg, layer_only=True)
        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)
        act_layer = partial(build_act_layer, act_cfg, layer_only=True)
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels), act_layer(out_channels)))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1,
            norm_layer, act_layer))
        modules.append(ASPPConv(in_channels, out_channels, rate2,
            norm_layer, act_layer))
        modules.append(ASPPConv(in_channels, out_channels, rate3,
            norm_layer, act_layer))
        modules.append(ASPPPooling(in_channels, out_channels, norm_layer,
            act_layer))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels,
            out_channels, 1, bias=False), norm_layer(out_channels),
            act_layer(out_channels))
        self.with_dropout = dropout is not None
        if self.with_dropout:
            self.dropout = nn.Dropout(dropout)
        logger.info('ASPP init weights')
        init_weights(self.modules())

    def forward(self, feats):
        feats_ = feats.copy()
        x = feats_[self.from_layer]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        if self.with_dropout:
            res = self.dropout(res)
        feats_[self.to_layer] = res
        return feats_


@ENHANCE_MODULES.register_module
class PPM(nn.Module):

    def __init__(self, in_channels, out_channels, bins, from_layer,
        to_layer, norm_cfg=None, act_cfg=None):
        super(PPM, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)
        self.blocks = nn.ModuleList()
        for bin_ in bins:
            self.blocks.append(nn.Sequential(nn.AdaptiveAvgPool2d(bin_), nn
                .Conv2d(in_channels, out_channels, 1, bias=False),
                build_norm_layer(norm_cfg, out_channels, layer_only=True),
                build_act_layer(act_cfg, out_channels, layer_only=True)))
        logger.info('PPM init weights')
        init_weights(self.modules())

    def forward(self, feats):
        feats_ = feats.copy()
        x = feats_[self.from_layer]
        h, w = x.shape[2:]
        out = [x]
        for block in self.blocks:
            feat = F.interpolate(block(x), (h, w), mode='bilinear',
                align_corners=True)
            out.append(feat)
        out = torch.cat(out, 1)
        feats_[self.to_layer] = out
        return feats_


HEADS = Registry('head')


@HEADS.register_module
class Head(nn.Module):
    """Head

    Args:
    """

    def __init__(self, in_channels, out_channels, inter_channels=None,
        conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=dict(
        type='Relu', inplace=True), num_convs=0, upsample=None, dropouts=None):
        super().__init__()
        if num_convs > 0:
            layers = [ConvModules(in_channels, inter_channels, 3, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                num_convs=num_convs, dropouts=dropouts), nn.Conv2d(
                inter_channels, out_channels, 1)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 1)]
        if upsample:
            upsample_layer = build_module(upsample)
            layers.append(upsample_layer)
        self.block = nn.Sequential(*layers)
        logger.info('Head init weights')
        init_weights(self.modules())

    def forward(self, x):
        feat = self.block(x)
        return feat


class TLU(nn.Module):

    def __init__(self, num_features):
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(1, num_features, 1, 1),
            requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def forward(self, x):
        return torch.max(x, self.tau)

    def extra_repr(self):
        return '{num_features}'.format(**self.__dict__)


conv_cfg = {'Conv': nn.Conv2d}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer


@UTILS.register_module
class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (str or None): Config dict for activation layer.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias='auto', conv_cfg=dict(type=
        'Conv'), norm_cfg=None, act_cfg=dict(type='Relu', inplace=True),
        order=('conv', 'norm', 'act'), dropout=None):
        super(ConvModule, self).__init__()
        assert isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])
        self.with_norm = norm_cfg is not None
        self.with_act = act_cfg is not None
        self.with_dropout = dropout is not None
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias
        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        if self.with_act:
            if order.index('act') > order.index('conv'):
                act_channels = out_channels
            else:
                act_channels = in_channels
            self.act_name, act = build_act_layer(act_cfg, act_channels)
            self.add_module(self.act_name, act)
        if self.with_dropout:
            self.dropout = nn.Dropout2d(p=dropout)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    @property
    def activate(self):
        return getattr(self, self.act_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_act:
                x = self.activate(x)
        if self.with_dropout:
            x = self.dropout(x)
        return x


@UTILS.register_module
class ConvModules(nn.Module):
    """Head

    Args:
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias='auto', conv_cfg=dict(type=
        'Conv'), norm_cfg=None, act_cfg=dict(type='Relu', inplace=True),
        order=('conv', 'norm', 'act'), dropouts=None, num_convs=1):
        super().__init__()
        if dropouts is not None:
            assert num_convs == len(dropouts)
            dropout = dropouts[0]
        else:
            dropout = None
        layers = [ConvModule(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, conv_cfg, norm_cfg, act_cfg,
            order, dropout)]
        for ii in range(1, num_convs):
            if dropouts is not None:
                dropout = dropouts[ii]
            else:
                dropout = None
            layers.append(ConvModule(out_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias,
                conv_cfg, norm_cfg, act_cfg, order, dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.block(x)
        return feat


class FRN(nn.Module):

    def __init__(self, num_features, eps=1e-06):
        super(FRN, self).__init__()
        self.num_features = num_features
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1),
            requires_grad=True)
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1),
            requires_grad=True)
        self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        nu2 = torch.mean(x.pow(2), dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        x = self.gamma * x + self.beta
        return x

    def extra_repr(self):
        return '{num_features}, eps={eps}'.format(**self.__dict__)


@UTILS.register_module
class Upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'scale_bias', 'mode',
        'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, scale_bias=0, mode=
        'nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.scale_bias = scale_bias
        self.mode = mode
        self.align_corners = align_corners
        assert (self.size is None) ^ (self.scale_factor is None)

    def forward(self, x):
        if self.size:
            size = self.size
        else:
            n, c, h, w = x.size()
            new_h = int(h * self.scale_factor + self.scale_bias)
            new_w = int(w * self.scale_factor + self.scale_bias)
            size = new_h, new_w
        return F.interpolate(x, size=size, mode=self.mode, align_corners=
            self.align_corners)

    def extra_repr(self):
        if self.size is not None:
            info = 'size=' + str(self.size)
        else:
            info = 'scale_factor=' + str(self.scale_factor)
            info += ', scale_bias=' + str(self.scale_bias)
        info += ', mode=' + self.mode
        return info


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Media_Smart_vedaseg(_paritybench_base):
    pass
    def test_000(self):
        self._check(ASPPConv(*[], **{'in_channels': 4, 'out_channels': 4, 'dilation': 1, 'norm_layer': _mock_layer, 'act_layer': _mock_layer}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ASPPPooling(*[], **{'in_channels': 4, 'out_channels': 4, 'norm_layer': _mock_layer, 'act_layer': _mock_layer}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4, 'norm_layer': _mock_layer, 'act_layer': _mock_layer}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(CollectBlock(*[], **{'from_layer': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(FRN(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Head(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(TLU(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

