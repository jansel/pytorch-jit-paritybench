import sys
_module = sys.modules[__name__]
del sys
resnet_ctc = _module
resnet_fc = _module
small_satrn = _module
tps_resnet_bilstm_attn = _module
benchmark = _module
utils = _module
common = _module
inference = _module
test = _module
torch2onnx = _module
train = _module
vedastr = _module
converter = _module
attn_converter = _module
base_convert = _module
builder = _module
ctc_converter = _module
fc_converter = _module
registry = _module
criteria = _module
cross_entropy_loss = _module
ctc_loss = _module
dataloaders = _module
builder = _module
samplers = _module
balance_sampler = _module
datasets = _module
base = _module
concat_dataset = _module
fold_dataset = _module
lmdb_dataset = _module
txt_datasets = _module
logger = _module
lr_schedulers = _module
base = _module
constant_lr = _module
cosine_lr = _module
exponential_lr = _module
poly_lr = _module
step_lr = _module
metrics = _module
accuracy = _module
models = _module
bodies = _module
body = _module
component = _module
feature_extractors = _module
builder = _module
decoders = _module
bricks = _module
bricks = _module
builder = _module
pva = _module
gfpn = _module
encoders = _module
backbones = _module
resnet = _module
vgg = _module
builder = _module
enhance_modules = _module
aspp = _module
ppm = _module
rectificators = _module
spin = _module
tps_stn = _module
sequences = _module
rnn = _module
decoder = _module
encoder = _module
transformer = _module
decoder = _module
encoder = _module
position_encoder = _module
adaptive_2d_encoder = _module
encoder = _module
utils = _module
unit = _module
attention = _module
multihead_attention = _module
decoder = _module
encoder = _module
feedforward = _module
feedforward = _module
heads = _module
att_head = _module
ctc_head = _module
fc_head = _module
head = _module
transformer_head = _module
model = _module
builder = _module
conv_module = _module
fc_module = _module
norm = _module
upsample = _module
weight_init = _module
optimizers = _module
builder = _module
runners = _module
base = _module
inference_runner = _module
test_runner = _module
train_runner = _module
transforms = _module
transforms = _module
checkpoint = _module
common = _module
config = _module
misc = _module
path = _module
registry = _module

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


import torch.nn as nn


import torch.utils.data as tud


import copy


import logging


import random


from torch.utils.data import Sampler


import re


from torch.utils.data import Dataset


from torch.utils.data import ConcatDataset as _ConcatDataset


import warnings


from functools import wraps


from torch.optim import Optimizer


import math


from torch.nn import functional as F


from torchvision.models.resnet import model_urls


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import conv1x1


import torch.nn.functional as F


import torch.optim as torch_optim


from torch.backends import cudnn


from collections import OrderedDict


import time


import torchvision


from torch.utils import model_zoo


import inspect


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(self._name, list(self._module_dict.keys()))
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
            raise TypeError('module must be a class, but got {}'.format(type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


CRITERIA = Registry('criterion')


class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.criteron = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction)

    def forward(self, pred, target, *args):
        return self.criteron(pred.contiguous().view(-1, pred.shape[-1]), target.contiguous().view(-1))


class CTCLoss(nn.Module):

    def __init__(self, zero_infinity=False, blank=0, reduction='mean'):
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(zero_infinity=zero_infinity, blank=blank, reduction=reduction)

    def forward(self, pred, target, target_length, batch_size):
        pred = pred.log_softmax(2)
        input_lengths = torch.full(size=(batch_size,), fill_value=pred.size(1), dtype=torch.long)
        pred_ = pred.permute(1, 0, 2)
        cost = self.criterion(log_probs=pred_, targets=target, input_lengths=input_lengths, target_lengths=target_length)
        return cost


BODIES = Registry('body')


BRICKS = Registry('brick')


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
        raise TypeError('type must be a str or valid type, but got {}'.format(type(obj_type)))
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
            raise KeyError('{} is not in the {} registry'.format(obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(type(obj_type)))
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


def build_brick(cfg, default_args=None):
    brick = build_from_cfg(cfg, BRICKS, default_args)
    return brick


COMPONENT = Registry('component')


def build_component(cfg, default_args=None):
    component = build_from_cfg(cfg, COMPONENT, default_args)
    return component


class GBody(nn.Module):

    def __init__(self, pipelines, collect=None):
        super(GBody, self).__init__()
        self.input_to_layer = 'input'
        self.components = nn.ModuleList([build_component(component) for component in pipelines])
        if collect is not None:
            self.collect = build_brick(collect)

    @property
    def with_collect(self):
        return hasattr(self, 'collect') and self.collect is not None

    def forward(self, x):
        feats = {self.input_to_layer: x}
        for component in self.components:
            component_from = component.from_layer
            component_to = component.to_layer
            if isinstance(component_from, list):
                inp = {key: feats[key] for key in component_from}
                out = component(**inp)
            else:
                inp = feats[component_from]
                out = component(inp)
            feats[component_to] = out
        if self.with_collect:
            return self.collect(feats)
        else:
            return feats


class BaseComponent(nn.Module):

    def __init__(self, from_layer, to_layer, component):
        super(BaseComponent, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.component = component

    def forward(self, x):
        return self.component(x)


DECODERS = Registry('decoder')


def build_decoder(cfg, default_args=None):
    decoder = build_from_cfg(cfg, DECODERS, default_args)
    return decoder


BACKBONES = Registry('backbone')


def build_backbone(cfg, default_args=None):
    backbone = build_from_cfg(cfg, BACKBONES, default_args)
    return backbone


ENHANCE_MODULES = Registry('enhance_module')


def build_enhance_module(cfg, default_args=None):
    enhance_module = build_from_cfg(cfg, ENHANCE_MODULES, default_args)
    return enhance_module


def build_encoder(cfg, default_args=None):
    backbone = build_backbone(cfg['backbone'])
    enhance_cfg = cfg.get('enhance')
    if enhance_cfg:
        enhance_module = build_enhance_module(enhance_cfg)
        encoder = nn.Sequential(backbone, enhance_module)
    else:
        encoder = backbone
    return encoder


def build_feature_extractor(cfg):
    encoder = build_encoder(cfg.get('encoder'))
    if cfg.get('decoder'):
        middle = build_decoder(cfg.get('decoder'))
        if 'collect' in cfg:
            final = build_brick(cfg.get('collect'))
            feature_extractor = nn.Sequential(encoder, middle, final)
        else:
            feature_extractor = nn.Sequential(encoder, middle)
    else:
        assert 'collect' in cfg
        middle = build_brick(cfg.get('collect'))
        feature_extractor = nn.Sequential(encoder, middle)
    return feature_extractor


class FeatureExtractorComponent(BaseComponent):

    def __init__(self, from_layer, to_layer, arch):
        super(FeatureExtractorComponent, self).__init__(from_layer, to_layer, build_feature_extractor(arch))


RECTIFICATORS = Registry('Rectificator')


def build_rectificator(cfg, default_args=None):
    rectificator = build_from_cfg(cfg, RECTIFICATORS, default_args)
    return rectificator


class RectificatorComponent(BaseComponent):

    def __init__(self, from_layer, to_layer, arch):
        super(RectificatorComponent, self).__init__(from_layer, to_layer, build_rectificator(arch))


SEQUENCE_ENCODERS = Registry('sequence_encoder')


def build_sequence_encoder(cfg, default_args=None):
    sequence_encoder = build_from_cfg(cfg, SEQUENCE_ENCODERS, default_args)
    return sequence_encoder


class SequenceEncoderComponent(BaseComponent):

    def __init__(self, from_layer, to_layer, arch):
        super(SequenceEncoderComponent, self).__init__(from_layer, to_layer, build_sequence_encoder(arch))


class BrickComponent(BaseComponent):

    def __init__(self, from_layer, to_layer, arch):
        super(BrickComponent, self).__init__(from_layer, to_layer, build_brick(arch))


UTILS = Registry('utils')


def build_module(cfg, default_args=None):
    util = build_from_cfg(cfg, UTILS, default_args)
    return util


class JunctionBlock(nn.Module):
    """JunctionBlock

    Args:
    """

    def __init__(self, top_down, lateral, post, to_layer, fusion_method=None):
        super(JunctionBlock, self).__init__()
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
            if 'upsample' in top_down_:
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
                raise ValueError('There is neither top down feature nor lateral feature')
        feat = self.post_block(feat)
        return feat


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


norm_cfg = {'BN': ('bn', nn.BatchNorm2d), 'SyncBN': ('bn', nn.SyncBatchNorm), 'GN': ('gn', nn.GroupNorm)}


def build_norm_layer(cfg, num_features, postfix=''):
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
    return name, layer


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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto', conv_cfg=dict(type='Conv'), norm_cfg=None, activation='relu', inplace=True, order=('conv', 'norm', 'act'), dropout=None):
        super(ConvModule, self).__init__()
        assert isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])
        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        self.with_dropout = dropout is not None
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias
        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
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
        if self.with_activatation:
            if self.activation not in ['relu', 'tanh', 'sigmoid']:
                raise ValueError('{} is currently not supported.'.format(self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
            elif self.activation == 'tanh':
                self.activate = nn.Tanh()
            elif self.activation == 'sigmoid':
                self.activate = nn.Sigmoid()
        if self.with_dropout:
            self.dropout = nn.Dropout(p=dropout)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        if self.with_dropout:
            x = self.dropout(x)
        return x


class FusionBlock(nn.Module):
    """FusionBlock

        Args:
    """

    def __init__(self, method, from_layers, feat_strides, in_channels_list, out_channels_list, upsample, conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), activation='relu', inplace=True, common_stride=4):
        super(FusionBlock, self).__init__()
        assert method in ('add', 'concat')
        self.method = method
        self.from_layers = from_layers
        assert len(in_channels_list) == len(out_channels_list)
        self.blocks = nn.ModuleList()
        for idx in range(len(from_layers)):
            in_channels = in_channels_list[idx]
            out_channels = out_channels_list[idx]
            feat_stride = feat_strides[idx]
            ups_num = int(max(1, math.log2(feat_stride) - math.log2(common_stride)))
            head_ops = []
            for idx2 in range(ups_num):
                cur_in_channels = in_channels if idx2 == 0 else out_channels
                conv = ConvModule(cur_in_channels, out_channels, kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=activation, inplace=inplace)
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


class CollectBlock(nn.Module):
    """CollectBlock

        Args:
    """

    def __init__(self, from_layer, to_layer=None):
        super(CollectBlock, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer

    def forward(self, feats):
        if self.to_layer is None:
            if isinstance(self.from_layer, str):
                return feats[self.from_layer]
            elif isinstance(self.from_layer, list):
                return {f_layer: feats[f_layer] for f_layer in self.from_layer}
        elif isinstance(self.from_layer, str):
            feats[self.to_layer] = feats[self.from_layer]
        elif isinstance(self.from_layer, list):
            feats[self.to_layer] = {f_layer: feats[f_layer] for f_layer in self.from_layer}


class CellAttentionBlock(nn.Module):

    def __init__(self, feat, hidden, fusion_method='add', post=None, post_activation='softmax'):
        super(CellAttentionBlock, self).__init__()
        feat_ = feat.copy()
        self.feat_from = feat_.pop('from_layer')
        self.feat_block = build_module(feat_)
        self.hidden_block = build_module(hidden)
        self.fusion_method = fusion_method
        self.activate = post_activation
        if post is not None:
            self.post_block = build_module(post)
        else:
            self.post_block = nn.Sequential()

    def forward(self, feats, hidden):
        feat = feats[self.feat_from]
        b, c = feat.size(0), feat.size(1)
        feat_to_attend = feat.view(b, c, -1)
        x = self.feat_block(feat)
        y = self.hidden_block(hidden)
        assert self.fusion_method in ['add', 'dot']
        if self.fusion_method == 'add':
            attention_map = x + y
        elif self.fusion_method == 'dot':
            attention_map = x * y
        attention_map = self.post_block(attention_map)
        b, c = attention_map.size(0), attention_map.size(1)
        attention_map = attention_map.view(b, c, -1)
        assert self.activate in ['softmax', 'sigmoid']
        if self.activate == 'softmax':
            attention_map = F.softmax(attention_map, dim=2)
        elif self.activate == 'sigmoid':
            attention_map = F.sigmoid(attention_map)
        feat = feat_to_attend * attention_map
        feat = feat.sum(2)
        return feat


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, is_rnn=False, mode='fan_in', nonlinearity='leaky_relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        if is_rnn:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, bias)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(param, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    elif is_rnn:
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if not is_rnn and hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            kaiming_init(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            constant_init(m, 1)
        elif isinstance(m, nn.Linear):
            xavier_init(m)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            kaiming_init(m, is_rnn=True)


class PVABlock(nn.Module):

    def __init__(self, num_steps, in_channels, embedding_channels=512, inner_channels=512):
        super(PVABlock, self).__init__()
        self.num_steps = num_steps
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.embedding_channels = embedding_channels
        self.order_embeddings = nn.Parameter(torch.randn(self.num_steps, self.embedding_channels), requires_grad=True)
        self.v_linear = nn.Linear(self.in_channels, self.inner_channels, bias=False)
        self.o_linear = nn.Linear(self.embedding_channels, self.inner_channels, bias=False)
        self.e_linear = nn.Linear(self.inner_channels, 1, bias=False)
        init_weights(self.modules())

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        o_out = self.o_linear(self.order_embeddings).view(1, self.num_steps, 1, self.inner_channels)
        v_out = self.v_linear(x).unsqueeze(1)
        att = self.e_linear(torch.tanh(o_out + v_out)).squeeze(3)
        att = torch.softmax(att, dim=2)
        out = torch.bmm(att, x)
        return out


def build_bricks(cfgs):
    bricks = nn.ModuleList()
    for brick_cfg in cfgs:
        bricks.append(build_brick(brick_cfg))
    return bricks


logger = logging.getLogger()


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
            bottom_up[layer.to_layer] = x
        if self.fusion:
            x = self.fusion(feats)
            bottom_up['fusion'] = x
        return bottom_up


class ResNetCls(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, multi_grid=None, norm_layer=None):
        super(ResNetCls, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], multi_grid=multi_grid)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, multi_grid=None):
        norm_layer = self._norm_layer
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
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation * multi_grid[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation * multi_grid[i], norm_layer=norm_layer))
        return nn.Sequential(*layers)

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
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


MODEL_CFGS = {'resnet101': {'block': Bottleneck, 'layer': [3, 4, 23, 3], 'weights_url': model_urls['resnet101']}, 'resnet50': {'block': Bottleneck, 'layer': [3, 4, 6, 3], 'weights_url': model_urls['resnet50']}, 'resnet34': {'block': BasicBlock, 'layer': [3, 4, 6, 3], 'weights_url': model_urls['resnet34']}, 'resnet18': {'block': BasicBlock, 'layer': [2, 2, 2, 2], 'weights_url': model_urls['resnet18']}}


class ResNet(ResNetCls):
    """ResNetEncoder

    Args:
        pretrain(bool)
    """

    def __init__(self, arch, replace_stride_with_dilation=None, multi_grid=None, pretrain=True):
        cfg = MODEL_CFGS[arch]
        super().__init__(cfg['block'], cfg['layer'], replace_stride_with_dilation=replace_stride_with_dilation, multi_grid=multi_grid)
        if pretrain:
            logger.info('ResNet init weights from pretreain')
            state_dict = load_state_dict_from_url(cfg['weights_url'])
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.info('ResNet init weights')
            init_weights(self.modules())
        del self.fc, self.avgpool

    def forward(self, x):
        feats = {}
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        feats['c1'] = x0
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        feats['c2'] = x1
        x2 = self.layer2(x1)
        feats['c3'] = x2
        x3 = self.layer3(x2)
        feats['c4'] = x3
        x4 = self.layer4(x3)
        feats['c5'] = x4
        return feats


BLOCKS = {'BasicBlock': BasicBlock, 'Bottleneck': Bottleneck}


def build_torch_nn(cfg, default_args=None):
    module = build_from_cfg(cfg, nn, default_args, 'module')
    return module


class GResNet(nn.Module):

    def __init__(self, layers, zero_init_residual=False, groups=1, width_per_group=64, norm_layer=None):
        super(GResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.layers = nn.ModuleList()
        stage_layers = []
        for layer_name, layer_cfg in layers:
            if layer_name == 'conv':
                layer = build_module(layer_cfg)
                self.inplanes = layer_cfg['out_channels']
            elif layer_name == 'pool':
                layer = build_torch_nn(layer_cfg)
            elif layer_name == 'block':
                layer = self._make_layer(**layer_cfg)
            else:
                raise ValueError('Unknown layer name {}'.format(layer_name))
            stride = layer_cfg.get('stride', 1)
            max_stride = stride if isinstance(stride, int) else max(stride)
            if max_stride > 1:
                self.layers.append(nn.Sequential(*stage_layers))
                stage_layers = []
            stage_layers.append(layer)
        self.layers.append(nn.Sequential(*stage_layers))
        logger.info('GResNet init weights')
        init_weights(self.modules())

    def _make_layer(self, block_name, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        block = BLOCKS[block_name]
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            feats['c{}'.format(i)] = x
        return feats


class GVGG(nn.Module):

    def __init__(self, layers):
        super(GVGG, self).__init__()
        self.layers = nn.ModuleList()
        stage_layers = []
        for layer_name, layer_cfg in layers:
            if layer_name == 'conv':
                layer = build_module(layer_cfg)
            elif layer_name == 'pool':
                layer = build_torch_nn(layer_cfg)
            else:
                raise ValueError('Unknown layer name {}'.format(layer_name))
            stride = layer_cfg.get('stride', 1)
            max_stride = stride if isinstance(stride, int) else max(stride)
            if max_stride > 1:
                self.layers.append(nn.Sequential(*stage_layers))
                stage_layers = []
            stage_layers.append(layer)
        self.layers.append(nn.Sequential(*stage_layers))
        init_weights(self.modules())

    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            feats['c{}'.format(i)] = x
        return feats


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='nearest')


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, atrous_rates, from_layer, to_layer, dropout=None):
        super(ASPP, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.with_dropout = dropout is not None
        if self.with_dropout:
            self.dropout = nn.Dropout(p=dropout)
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


class PPM(nn.Module):

    def __init__(self, in_channels, out_channels, bins, from_layer, to_layer):
        super(PPM, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.blocks = nn.ModuleList()
        for bin_ in bins:
            self.blocks.append(nn.Sequential(nn.AdaptiveAvgPool2d(bin_), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        logger.info('PPM init weights')
        init_weights(self.modules())

    def forward(self, feats):
        feats_ = feats.copy()
        x = feats_[self.from_layer]
        h, w = x.shape[2:]
        out = [x]
        for block in self.blocks:
            feat = F.interpolate(block(x), (h, w), mode='bilinear', align_corners=True)
            out.append(feat)
        out = torch.cat(out, 1)
        feats_[self.to_layer] = out
        return feats_


class SPN(nn.Module):

    def __init__(self, cfg):
        super(SPN, self).__init__()
        self.body = build_feature_extractor(cfg['feature_extractor'])
        self.pool = build_torch_nn(cfg['pool'])
        heads = []
        for head in cfg['head']:
            heads.append(build_module(head))
        self.head = nn.Sequential(*heads)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.body(x)
        x = self.pool(x).view(batch_size, -1)
        x = self.head(x)
        return x


class AIN(nn.Module):

    def __init__(self, cfg):
        super(AIN, self).__init__()
        self.body = build_feature_extractor(cfg['feature_extractor'])

    def forward(self, x):
        x = self.body(x)
        return x


def generate_beta(K):
    betas = []
    for i in range(1, K + 2):
        p = i / (2 * (K + 1))
        beta = round(np.log(1 - p) / np.log(p), 2)
        betas.append(beta)
    for i in range(K + 2, 2 * K + 2):
        beta = round(1 / betas[i - (K + 1)], 2)
        betas.append(beta)
    return betas


class SPIN(nn.Module):

    def __init__(self, spin, k):
        super(SPIN, self).__init__()
        self.body = build_feature_extractor(spin['feature_extractor'])
        self.spn = SPN(spin['spn'])
        self.betas = generate_beta(k)
        init_weights(self.modules())

    def forward(self, x):
        b, c, h, w = x.size()
        init_img = copy.copy(x)
        x = self.body(x)
        spn_out = self.spn(x)
        omega = spn_out[:, :-1]
        g_out = init_img.requires_grad_(True)
        gamma_out = [(g_out ** beta) for beta in self.betas]
        gamma_out = torch.stack(gamma_out, axis=1).requires_grad_(True)
        fusion_img = omega[:, :, None, None, None] * gamma_out
        fusion_img = torch.sigmoid(fusion_img.sum(dim=1))
        return fusion_img


class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, output_size, eps=1e-06):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = eps
        self.output_height, self.output_width = output_size
        self.F = F
        self.C = self._build_C(self.F)
        self.P = self._build_P(self.output_width, self.output_height)
        self.register_buffer('inv_delta_C', torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())
        self.register_buffer('P_hat', torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = hat_C ** 2 * np.log(hat_C)
        delta_C = np.concatenate([np.concatenate([np.ones((F, 1)), C, hat_C], axis=1), np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1), np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)], axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        P = np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        return P.reshape([-1, 2])

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))
        C_tile = np.expand_dims(C, axis=0)
        P_diff = P_tile - C_tile
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat

    def build_P_prime(self, batch_C_prime, device=None):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).float()), dim=1)
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)
        return batch_P_prime

    def forward(self, x):
        batch_size = x.size(0)
        build_P_prime = self.build_P_prime(x.view(batch_size, self.F, 2), x.device)
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.output_height, self.output_width, 2])
        return build_P_prime_reshape


class TPS_STN(nn.Module):

    def __init__(self, F, input_size, output_size, stn):
        super(TPS_STN, self).__init__()
        self.F = F
        self.input_size = input_size
        self.output_size = output_size
        self.feature_extractor = build_feature_extractor(stn['feature_extractor'])
        self.pool = build_torch_nn(stn['pool'])
        heads = []
        for head in stn['head']:
            heads.append(build_module(head))
        self.heads = nn.Sequential(*heads)
        self.grid_generator = GridGenerator(F, output_size)
        last_fc = heads[-1].fc
        last_fc.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        last_fc.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, x):
        batch_size = x.size(0)
        batch_C_prime = self.feature_extractor(x)
        batch_C_prime = self.pool(batch_C_prime).view(batch_size, -1)
        batch_C_prime = self.heads(batch_C_prime)
        build_P_prime_reshape = self.grid_generator(batch_C_prime)
        if torch.__version__ > '1.2.0':
            out = F.grid_sample(x, build_P_prime_reshape, padding_mode='border', align_corners=True)
        else:
            out = F.grid_sample(x, build_P_prime_reshape, padding_mode='border')
        return out


class BaseCell(nn.Module):

    def __init__(self, basic_cell, input_size, hidden_size, bias=True, num_layers=1):
        super(BaseCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cells.append(basic_cell(input_size=input_size, hidden_size=hidden_size, bias=bias))
            else:
                self.cells.append(basic_cell(input_size=hidden_size, hidden_size=hidden_size, bias=bias))
        init_weights(self.modules())

    def init_hidden(self, batch_size, device=None, value=0):
        raise NotImplementedError()

    def get_output(self, hiddens):
        raise NotImplementedError()

    def get_hidden_state(self, hidden):
        raise NotImplementedError()

    def forward(self, x, pre_hiddens):
        next_hiddens = []
        hidden = None
        for i, cell in enumerate(self.cells):
            if i == 0:
                hidden = cell(x, pre_hiddens[i])
            else:
                hidden = cell(self.get_hidden_state(hidden), pre_hiddens[i])
            next_hiddens.append(hidden)
        return next_hiddens


SEQUENCE_DECODERS = Registry('sequence_decoder')


class LSTMCell(BaseCell):

    def __init__(self, input_size, hidden_size, bias=True, num_layers=1):
        super(LSTMCell, self).__init__(nn.LSTMCell, input_size, hidden_size, bias, num_layers)

    def init_hidden(self, batch_size, device=None, value=0):
        hiddens = []
        for _ in range(self.num_layers):
            hidden = torch.FloatTensor(batch_size, self.hidden_size).fill_(value), torch.FloatTensor(batch_size, self.hidden_size).fill_(value)
            hiddens.append(hidden)
        return hiddens

    def get_output(self, hiddens):
        return hiddens[-1][0]

    def get_hidden_state(self, hidden):
        return hidden[0]


class GRUCell(BaseCell):

    def __init__(self, input_size, hidden_size, bias=True, num_layers=1):
        super(GRUCell, self).__init__(nn.GRUCell, input_size, hidden_size, bias, num_layers)

    def init_hidden(self, batch_size, device=None, value=0):
        hiddens = []
        for i in range(self.num_layers):
            hidden = torch.FloatTensor(batch_size, self.hidden_size).fill_(value)
            hiddens.append(hidden)
        return hiddens

    def get_output(self, hiddens):
        return hiddens[-1]

    def get_hidden_state(self, hidden):
        return hidden


class RNN(nn.Module):

    def __init__(self, input_pool, layers, keep_order=False):
        super(RNN, self).__init__()
        self.keep_order = keep_order
        if input_pool:
            self.input_pool = build_torch_nn(input_pool)
        self.layers = nn.ModuleList()
        for i, (layer_name, layer_cfg) in enumerate(layers):
            if layer_name in ['rnn', 'fc']:
                self.layers.add_module('{}_{}'.format(i, layer_name), build_torch_nn(layer_cfg))
            else:
                raise ValueError('Unknown layer name {}'.format(layer_name))
        init_weights(self.modules())

    @property
    def with_input_pool(self):
        return hasattr(self, 'input_pool') and self.input_pool

    def forward(self, x):
        if self.with_input_pool:
            out = self.input_pool(x).squeeze(2)
        else:
            out = x
        out = out.permute(0, 2, 1)
        for layer_name, layer in self.layers.named_children():
            if 'rnn' in layer_name:
                layer.flatten_parameters()
                out, _ = layer(out)
            else:
                out = layer(out)
        if not self.keep_order:
            out = out.permute(0, 2, 1).unsqueeze(2)
        return out.contiguous()


TRANSFORMER_DECODER_LAYERS = Registry('transformer_decoder_layer')


def build_decoder_layer(cfg, default_args=None):
    decoder_layer = build_from_cfg(cfg, TRANSFORMER_DECODER_LAYERS, default_args)
    return decoder_layer


POSITION_ENCODERS = Registry('position_encoder')


def build_position_encoder(cfg, default_args=None):
    position_encoder = build_from_cfg(cfg, POSITION_ENCODERS, default_args)
    return position_encoder


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, position_encoder=None):
        super(TransformerDecoder, self).__init__()
        if position_encoder is not None:
            self.pos_encoder = build_position_encoder(position_encoder)
        self.layers = nn.ModuleList([build_decoder_layer(decoder_layer) for _ in range(num_layers)])
        logger.info('TransformerDecoder init weights')
        init_weights(self.modules())

    @property
    def with_position_encoder(self):
        return hasattr(self, 'pos_encoder') and self.pos_encoder is not None

    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        if self.with_position_encoder:
            tgt = self.pos_encoder(tgt)
        for layer in self.layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)
        return tgt


TRANSFORMER_ENCODER_LAYERS = Registry('transformer_encoder_layer')


def build_encoder_layer(cfg, default_args=None):
    encoder_layer = build_from_cfg(cfg, TRANSFORMER_ENCODER_LAYERS, default_args)
    return encoder_layer


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, position_encoder=None):
        super(TransformerEncoder, self).__init__()
        if position_encoder is not None:
            self.pos_encoder = build_position_encoder(position_encoder)
        self.layers = nn.ModuleList([build_encoder_layer(encoder_layer) for _ in range(num_layers)])
        logger.info('TransformerEncoder init weights')
        init_weights(self.modules())

    @property
    def with_position_encoder(self):
        return hasattr(self, 'pos_encoder') and self.pos_encoder is not None

    def forward(self, src, src_mask=None):
        if self.with_position_encoder:
            src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


def generate_encoder(in_channels, max_len):
    pos = torch.arange(max_len).float().unsqueeze(1)
    i = torch.arange(in_channels).float().unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, 2 * (i // 2) / in_channels)
    position_encoder = pos * angle_rates
    position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
    position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])
    return position_encoder


class Adaptive2DPositionEncoder(nn.Module):

    def __init__(self, in_channels, max_h=200, max_w=200, dropout=0.1):
        super(Adaptive2DPositionEncoder, self).__init__()
        h_position_encoder = generate_encoder(in_channels, max_h)
        h_position_encoder = h_position_encoder.transpose(0, 1).view(1, in_channels, max_h, 1)
        w_position_encoder = generate_encoder(in_channels, max_w)
        w_position_encoder = w_position_encoder.transpose(0, 1).view(1, in_channels, 1, max_w)
        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)
        self.h_scale = self.scale_factor_generate(in_channels)
        self.w_scale = self.scale_factor_generate(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    def scale_factor_generate(self, in_channels):
        scale_factor = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.Sigmoid())
        return scale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = self.pool(x)
        h_pos_encoding = self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]
        out = x + h_pos_encoding + w_pos_encoding
        out = self.dropout(out)
        return out


class PositionEncoder1D(nn.Module):

    def __init__(self, in_channels, max_len=2000, dropout=0.1):
        super(PositionEncoder1D, self).__init__()
        position_encoder = generate_encoder(in_channels, max_len)
        position_encoder = position_encoder.unsqueeze(0)
        self.register_buffer('position_encoder', position_encoder)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = x + self.position_encoder[:, :x.size(1), :]
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


TRANSFORMER_ATTENTIONS = Registry('transformer_attention')


class MultiHeadAttention(nn.Module):

    def __init__(self, in_channels, k_channels, v_channels, n_head=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.in_channels = in_channels
        self.k_channels = k_channels
        self.v_channels = v_channels
        self.n_head = n_head
        self.q_linear = nn.Linear(in_channels, n_head * k_channels)
        self.k_linear = nn.Linear(in_channels, n_head * k_channels)
        self.v_linear = nn.Linear(in_channels, n_head * v_channels)
        self.attention = ScaledDotProductAttention(temperature=k_channels ** 0.5, dropout=dropout)
        self.out_linear = nn.Linear(n_head * v_channels, in_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.q_linear(q).view(b, q_len, self.n_head, self.k_channels).transpose(1, 2)
        k = self.k_linear(k).view(b, k_len, self.n_head, self.k_channels).transpose(1, 2)
        v = self.v_linear(v).view(b, v_len, self.n_head, self.v_channels).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        out, attn = self.attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, q_len, self.n_head * self.v_channels)
        out = self.out_linear(out)
        out = self.dropout(out)
        return out, attn


def build_attention(cfg, default_args=None):
    attention = build_from_cfg(cfg, TRANSFORMER_ATTENTIONS, default_args)
    return attention


TRANSFORMER_FEEDFORWARDS = Registry('transformer_feedforward')


def build_feedforward(cfg, default_args=None):
    feedforward = build_from_cfg(cfg, TRANSFORMER_FEEDFORWARDS, default_args)
    return feedforward


class TransformerDecoderLayer1D(nn.Module):

    def __init__(self, self_attention, self_attention_norm, attention, attention_norm, feedforward, feedforward_norm):
        super(TransformerDecoderLayer1D, self).__init__()
        self.self_attention = build_attention(self_attention)
        self.self_attention_norm = build_torch_nn(self_attention_norm)
        self.attention = build_attention(attention)
        self.attention_norm = build_torch_nn(attention_norm)
        self.feedforward = build_feedforward(feedforward)
        self.feedforward_norm = build_torch_nn(feedforward_norm)

    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        attn1, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        out1 = self.self_attention_norm(tgt + attn1)
        size = src.size()
        if len(size) == 4:
            b, c, h, w = size
            src = src.view(b, c, h * w).transpose(1, 2)
            if src_mask is not None:
                src_mask = src_mask.view(b, 1, h * w)
        attn2, _ = self.attention(out1, src, src, src_mask)
        out2 = self.attention_norm(out1 + attn2)
        ffn_out = self.feedforward(out2)
        out3 = self.feedforward_norm(out2 + ffn_out)
        return out3


class TransformerEncoderLayer1D(nn.Module):

    def __init__(self, attention, attention_norm, feedforward, feedforward_norm):
        super(TransformerEncoderLayer1D, self).__init__()
        self.attention = build_attention(attention)
        self.attention_norm = build_torch_nn(attention_norm)
        self.feedforward = build_feedforward(feedforward)
        self.feedforward_norm = build_torch_nn(feedforward_norm)

    def forward(self, src, src_mask=None):
        attn_out, _ = self.attention(src, src, src, src_mask)
        out1 = self.attention_norm(src + attn_out)
        ffn_out = self.feedforward(out1)
        out2 = self.feedforward_norm(out1 + ffn_out)
        return out2


class TransformerEncoderLayer2D(nn.Module):

    def __init__(self, attention, attention_norm, feedforward, feedforward_norm):
        super(TransformerEncoderLayer2D, self).__init__()
        self.attention = build_attention(attention)
        self.attention_norm = build_torch_nn(attention_norm)
        self.feedforward = build_feedforward(feedforward)
        self.feedforward_norm = build_torch_nn(feedforward_norm)

    def norm(self, norm_layer, x):
        b, c, h, w = x.size()
        if isinstance(norm_layer, nn.LayerNorm):
            out = x.view(b, c, h * w).transpose(1, 2)
            out = norm_layer(out)
            out = out.transpose(1, 2).contiguous().view(b, c, h, w)
        else:
            out = norm_layer(x)
        return out

    def forward(self, src, src_mask=None):
        b, c, h, w = src.size()
        src = src.view(b, c, h * w).transpose(1, 2)
        if src_mask is not None:
            src_mask = src_mask.view(b, 1, h * w)
        attn_out, _ = self.attention(src, src, src, src_mask)
        out1 = src + attn_out
        out1 = out1.transpose(1, 2).contiguous().view(b, c, h, w)
        out1 = self.norm(self.attention_norm, out1)
        ffn_out = self.feedforward(out1)
        out2 = self.norm(self.feedforward_norm, out1 + ffn_out)
        return out2


class Feedforward(nn.Module):

    def __init__(self, layers):
        super(Feedforward, self).__init__()
        self.layers = [build_module(layer) for layer in layers]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        return out


HEADS = Registry('head')


def build_sequence_decoder(cfg, default_args=None):
    sequence_encoder = build_from_cfg(cfg, SEQUENCE_DECODERS, default_args)
    return sequence_encoder


class AttHead(nn.Module):

    def __init__(self, cell, generator, num_steps, num_class, input_attention_block=None, output_attention_block=None, text_transform=None, holistic_input_from=None):
        super(AttHead, self).__init__()
        if input_attention_block is not None:
            self.input_attention_block = build_brick(input_attention_block)
        self.cell = build_sequence_decoder(cell)
        self.generator = build_torch_nn(generator)
        self.num_steps = num_steps
        self.num_class = num_class
        if output_attention_block is not None:
            self.output_attention_block = build_brick(output_attention_block)
        if text_transform is not None:
            self.text_transform = build_torch_nn(text_transform)
        if holistic_input_from is not None:
            self.holistic_input_from = holistic_input_from
        self.register_buffer('embeddings', torch.diag(torch.ones(self.num_class)))
        logger.info('AttHead init weights')
        init_weights(self.modules())

    @property
    def with_holistic_input(self):
        return hasattr(self, 'holistic_input_from') and self.holistic_input_from

    @property
    def with_input_attention(self):
        return hasattr(self, 'input_attention_block') and self.input_attention_block is not None

    @property
    def with_output_attention(self):
        return hasattr(self, 'output_attention_block') and self.output_attention_block is not None

    @property
    def with_text_transform(self):
        return hasattr(self, 'text_transform') and self.text_transform

    def forward(self, feats, texts):
        batch_size = texts.size(0)
        hidden = self.cell.init_hidden(batch_size, device=texts.device)
        if self.with_holistic_input:
            holistic_input = feats[self.holistic_input_from][:, :, 0, -1]
            hidden = self.cell(holistic_input, hidden)
        out = []
        if self.training:
            use_gt = True
            assert self.num_steps == texts.size(1)
        else:
            use_gt = False
            assert texts.size(1) == 1
        for i in range(self.num_steps):
            if i == 0:
                indexes = texts[:, i]
            elif use_gt:
                indexes = texts[:, i]
            else:
                _, indexes = out[-1].max(1)
            text_feat = self.embeddings.index_select(0, indexes)
            if self.with_text_transform:
                text_feat = self.text_transform(text_feat)
            if self.with_input_attention:
                attention_feat = self.input_attention_block(feats, self.cell.get_output(hidden).unsqueeze(-1).unsqueeze(-1))
                cell_input = torch.cat([attention_feat, text_feat], dim=1)
            else:
                cell_input = text_feat
            hidden = self.cell(cell_input, hidden)
            out_feat = self.cell.get_output(hidden)
            if self.with_output_attention:
                attention_feat = self.output_attention_block(feats, self.cell.get_output(hidden).unsqueeze(-1).unsqueeze(-1))
                out_feat = torch.cat([self.cell.get_output(hidden), attention_feat], dim=1)
            out.append(self.generator(out_feat))
        out = torch.stack(out, dim=1)
        return out


class CTCHead(nn.Module):
    """CTCHead

    Args:
    """

    def __init__(self, in_channels, num_class, from_layer, pool=None):
        super(CTCHead, self).__init__()
        self.num_class = num_class
        self.from_layer = from_layer
        fc = nn.Linear(in_channels, num_class)
        self.fc = fc
        logger.info('CTCHead init weights')
        init_weights(self.modules())

    def forward(self, x_input):
        x = x_input[self.from_layer]
        x = x.mean(2).permute(0, 2, 1)
        out = self.fc(x)
        return out


class FCModule(nn.Module):
    """FCModule

    Args:
    """

    def __init__(self, in_channels, out_channels, bias=True, activation='relu', inplace=True, dropout=None, order=('fc', 'act')):
        super(FCModule, self).__init__()
        self.order = order
        self.activation = activation
        self.inplace = inplace
        self.with_activatation = activation is not None
        self.with_dropout = dropout is not None
        self.fc = nn.Linear(in_channels, out_channels, bias)
        if self.with_activatation:
            if self.activation not in ['relu', 'tanh']:
                raise ValueError('{} is currently not supported.'.format(self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
            elif self.activation == 'tanh':
                self.activate = nn.Tanh()
        if self.with_dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.order == ('fc', 'act'):
            x = self.fc(x)
            if self.with_activatation:
                x = self.activate(x)
        elif self.order == ('act', 'fc'):
            if self.with_activatation:
                x = self.activate(x)
            x = self.fc(x)
        if self.with_dropout:
            x = self.dropout(x)
        return x


class FCModules(nn.Module):
    """FCModules

    Args:
    """

    def __init__(self, in_channels, out_channels, bias=True, activation='relu', inplace=True, dropouts=None, num_fcs=1):
        super().__init__()
        if dropouts is not None:
            assert num_fcs == len(dropouts)
            dropout = dropouts[0]
        else:
            dropout = None
        layers = [FCModule(in_channels, out_channels, bias, activation, inplace, dropout)]
        for ii in range(1, num_fcs):
            if dropouts is not None:
                dropout = dropouts[ii]
            else:
                dropout = None
            layers.append(FCModule(out_channels, out_channels, bias, activation, inplace, dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.block(x)
        return feat


class FCHead(nn.Module):
    """FCHead

    Args:
    """

    def __init__(self, in_channels, out_channels, num_class, batch_max_length, from_layer, inner_channels=None, bias=True, activation='relu', inplace=True, dropouts=None, num_fcs=0, pool=None):
        super(FCHead, self).__init__()
        self.num_class = num_class
        self.batch_max_length = batch_max_length
        self.from_layer = from_layer
        if num_fcs > 0:
            inter_fc = FCModules(in_channels, inner_channels, bias, activation, inplace, dropouts, num_fcs)
            fc = nn.Linear(inner_channels, out_channels)
        else:
            inter_fc = nn.Sequential()
            fc = nn.Linear(in_channels, out_channels)
        if pool is not None:
            self.pool = build_torch_nn(pool)
        self.inter_fc = inter_fc
        self.fc = fc
        logger.info('FCHead init weights')
        init_weights(self.modules())

    @property
    def with_pool(self):
        return hasattr(self, 'pool') and self.pool is not None

    def forward(self, x_input):
        x = x_input[self.from_layer]
        batch_size = x.size(0)
        if self.with_pool:
            x = self.pool(x)
        x = x.contiguous().view(batch_size, -1)
        out = self.inter_fc(x)
        out = self.fc(out)
        return out.reshape(-1, self.batch_max_length + 1, self.num_class)


class Head(nn.Module):
    """Head

    Args:
    """

    def __init__(self, from_layer, generator):
        super(Head, self).__init__()
        self.from_layer = from_layer
        self.generator = build_module(generator)
        logger.info('Head init weights')
        init_weights(self.modules())

    def forward(self, feats):
        x = feats[self.from_layer]
        out = self.generator(x)
        return out


class TransformerHead(nn.Module):

    def __init__(self, decoder, generator, embedding, num_steps, pad_id, src_from, src_mask_from=None):
        super(TransformerHead, self).__init__()
        self.decoder = build_sequence_decoder(decoder)
        self.generator = build_torch_nn(generator)
        self.embedding = build_torch_nn(embedding)
        self.num_steps = num_steps
        self.pad_id = pad_id
        self.src_from = src_from
        self.src_mask_from = src_mask_from
        logger.info('TransformerHead init weights')
        init_weights(self.modules())

    def pad_mask(self, text):
        pad_mask = text == self.pad_id
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)
        return pad_mask

    def order_mask(self, text):
        t = text.size(1)
        order_mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
        order_mask = order_mask.unsqueeze(0)
        return order_mask

    def text_embedding(self, texts):
        tgt = self.embedding(texts)
        tgt *= math.sqrt(tgt.size(2))
        return tgt

    def forward(self, feats, texts):
        src = feats[self.src_from]
        if self.src_mask_from:
            src_mask = feats[self.src_mask_from]
        else:
            src_mask = None
        if self.training:
            tgt = self.text_embedding(texts)
            tgt_mask = self.pad_mask(texts) | self.order_mask(texts)
            out = self.decoder(tgt, src, tgt_mask, src_mask)
            out = self.generator(out)
        else:
            out = None
            for _ in range(self.num_steps):
                tgt = self.text_embedding(texts)
                tgt_mask = self.order_mask(texts)
                out = self.decoder(tgt, src, tgt_mask, src_mask)
                out = self.generator(out)
                next_text = torch.argmax(out[:, -1:, :], dim=-1)
                texts = torch.cat([texts, next_text], dim=-1)
        return out


MODELS = Registry('model')


def build_body(cfg, default_args=None):
    body = build_from_cfg(cfg, BODIES, default_args)
    return body


def build_head(cfg, default_args=None):
    head = build_from_cfg(cfg, HEADS, default_args)
    return head


class GModel(nn.Module):

    def __init__(self, body, head, need_text=True):
        super(GModel, self).__init__()
        self.body = build_body(body)
        self.head = build_head(head)
        self.need_text = need_text

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        x = self.body(inputs[0])
        if self.need_text:
            out = self.head(x, inputs[1])
        else:
            out = self.head(x)
        return out


class ConvModules(nn.Module):
    """Head

    Args:
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias='auto', conv_cfg=dict(type='Conv'), norm_cfg=None, activation='relu', inplace=True, order=('conv', 'norm', 'act'), dropouts=None, num_convs=1):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride
        if dropouts is not None:
            assert num_convs == len(dropouts)
            dropout = dropouts[0]
        else:
            dropout = None
        layers = [ConvModule(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, conv_cfg, norm_cfg, activation, inplace, order, dropout)]
        for ii in range(1, num_convs):
            if dropouts is not None:
                dropout = dropouts[ii]
            else:
                dropout = None
            layers.append(ConvModule(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, conv_cfg, norm_cfg, activation, inplace, order, dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.block(x)
        return feat


class Upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'scale_bias', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, scale_bias=0, mode='nearest', align_corners=None):
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
        return F.interpolate(x, size=size, mode=self.mode, align_corners=self.align_corners)

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


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPPConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Adaptive2DPositionEncoder,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BaseComponent,
     lambda: ([], {'from_layer': 1, 'to_layer': 1, 'component': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CollectBlock,
     lambda: ([], {'from_layer': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvModules,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FCModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FCModules,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GridGenerator,
     lambda: ([], {'F': 4, 'output_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 2])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'in_channels': 4, 'k_channels': 4, 'v_channels': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PVABlock,
     lambda: ([], {'num_steps': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionEncoder1D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Media_Smart_vedastr(_paritybench_base):
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

