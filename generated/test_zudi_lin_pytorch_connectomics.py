import sys
_module = sys.modules[__name__]
del sys
connectomics = _module
config = _module
defaults = _module
utils = _module
data = _module
augmentation = _module
augmentor = _module
build = _module
composition = _module
copy_paste = _module
cutblur = _module
cutnoise = _module
flip = _module
grayscale = _module
misalign = _module
missing_parts = _module
missing_section = _module
mixup = _module
motion_blur = _module
rescale = _module
rotation = _module
test_augmentor = _module
warp = _module
dataset = _module
build = _module
collate = _module
dataset_cond = _module
dataset_tile = _module
dataset_volume = _module
data_affinity = _module
data_bbox = _module
data_blending = _module
data_crop = _module
data_diffusion = _module
data_io = _module
data_misc = _module
data_segmentation = _module
data_transform = _module
data_weight = _module
engine = _module
base = _module
solver = _module
build = _module
lr_scheduler = _module
trainer = _module
model = _module
arch = _module
deeplab = _module
fpn = _module
misc = _module
swinunetr = _module
unet = _module
unetr = _module
backbone = _module
botnet = _module
build = _module
efficientnet = _module
repvgg = _module
resnet = _module
block = _module
att_layer = _module
basic = _module
blurpool = _module
non_local = _module
residual = _module
unetr_blocks = _module
build = _module
loss = _module
criterion = _module
loss = _module
regularization = _module
initialize = _module
misc = _module
analysis = _module
debug = _module
evaluate = _module
monitor = _module
process = _module
system = _module
visualizer = _module
conf = _module
build = _module
collate = _module
trainer = _module
main = _module
cysgan = _module
trainer = _module
main = _module
ganloss_ssl = _module
trainer = _module
main = _module
main_vae = _module
two_stream = _module
twostream = _module
dataset = _module
trainer = _module
utils = _module
vae = _module
main = _module
compare_config = _module
eval_curvilinear = _module
setup = _module
tests = _module
test_augmentations = _module
test_loss_functions = _module
test_model_blocks = _module
test_models = _module

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


from typing import Optional


import numpy as np


import torch


import torchvision.transforms.functional as tf


from scipy.ndimage.morphology import binary_dilation


from scipy.ndimage.morphology import generate_binary_structure


import random


from numpy.core.numeric import indices


from itertools import combinations


from typing import List


from collections import OrderedDict


import itertools


from typing import Union


import math


import copy


from scipy.ndimage import zoom


import torch.utils.data


import scipy


from typing import Tuple


from scipy.ndimage import distance_transform_edt


import warnings


from enum import Enum


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import Set


from typing import Type


from torch.optim.swa_utils import AveragedModel


from torch.optim.swa_utils import SWALR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


import time


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from typing import Sequence


import torch.utils.checkpoint as checkpoint


from torch.nn import LayerNorm


from torch import nn


from torch import einsum


import torch.nn.parallel


from torch.nn import functional as F


from math import sqrt


from torch.jit.annotations import Dict


from torch.utils.tensorboard import SummaryWriter


import matplotlib


from matplotlib import pyplot as plt


import torch.distributed as dist


import torch.backends.cudnn as cudnn


import torchvision.utils as vutils


from inspect import getmembers


from inspect import isclass


from inspect import isfunction


from scipy import stats


from typing import TypeVar


from abc import abstractmethod


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation(activation: str='relu') ->nn.Module:
    """Get the specified activation layer.

    Args:
        activation (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
            ``'silu'``, ``'swish'``, 'efficient_swish'`` and ``'none'``. Default: ``'relu'``
    """
    assert activation in ['relu', 'leaky_relu', 'elu', 'gelu', 'silu', 'swish', 'efficient_swish', 'none'], 'Get unknown activation key {}'.format(activation)
    activation_dict = {'relu': nn.ReLU(inplace=True), 'leaky_relu': nn.LeakyReLU(negative_slope=0.2, inplace=True), 'elu': nn.ELU(alpha=1.0, inplace=True), 'gelu': nn.GELU(), 'silu': nn.SiLU(inplace=True), 'swish': Swish(), 'efficient_swish': MemoryEfficientSwish(), 'none': nn.Identity()}
    return activation_dict[activation]


def get_norm_2d(norm: str, out_channels: int, bn_momentum: float=0.1) ->nn.Module:
    """Get the specified normalization layer for a 2D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ['bn', 'sync_bn', 'gn', 'in', 'none'], 'Get unknown normalization layer key {}'.format(norm)
    norm = {'bn': nn.BatchNorm2d, 'sync_bn': nn.SyncBatchNorm, 'in': nn.InstanceNorm2d, 'gn': lambda channels: nn.GroupNorm(16, channels), 'none': nn.Identity}[norm]
    if norm in ['bn', 'sync_bn', 'in']:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, dilation: int, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn'):
        conv3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, padding_mode=pad_mode, bias=False)
        modules = [conv3x3, get_norm_2d(norm_mode, out_channels), get_activation(act_mode)]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, act_mode: str='elu', norm_mode: str='bn'):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), get_norm_2d(norm_mode, out_channels), get_activation(act_mode))

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int=256, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn'):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), get_norm_2d(norm_mode, out_channels), get_activation(act_mode)))
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, pad_mode=pad_mode, act_mode=act_mode, norm_mode=norm_mode))
        modules.append(ASPPPooling(in_channels, out_channels, act_mode=act_mode, norm_mode=norm_mode))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), get_norm_2d(norm_mode, out_channels), get_activation(act_mode))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHeadA(nn.Sequential):

    def __init__(self, in_channels: int, num_classes: int, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', **_):
        conv3x3 = nn.Conv2d(256, 256, 3, padding=1, padding_mode=pad_mode, bias=False)
        super(DeepLabHeadA, self).__init__(ASPP(in_channels, [12, 24, 36], 256, pad_mode, act_mode, norm_mode), conv3x3, get_norm_2d(norm_mode, 256), get_activation(act_mode), nn.Conv2d(256, num_classes, 1))


class DeepLabHeadB(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', **_):
        super(DeepLabHeadB, self).__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36], 256, pad_mode, act_mode, norm_mode)
        self.conv1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1, padding_mode=pad_mode, bias=False), get_norm_2d(norm_mode, 128), get_activation(act_mode))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, padding_mode=pad_mode, bias=False), get_norm_2d(norm_mode, 128), get_activation(act_mode), nn.Conv2d(128, num_classes, 3, padding=1, padding_mode=pad_mode))

    def forward(self, x):
        x = self.aspp(x)
        H, W = self._interp_shape(x)
        x = self.conv1(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.conv2(x)
        return x

    def _interp_shape(self, x):
        H, W = x.shape[-2:]
        H = 2 * H - 1 if H % 2 == 1 else 2 * H
        W = 2 * W - 1 if W % 2 == 1 else 2 * W
        return H, W


class DeepLabHeadC(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', **_):
        super(DeepLabHeadC, self).__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36], 256, pad_mode, act_mode, norm_mode)
        self.conv = nn.Sequential(nn.Conv2d(256, 32, 1, bias=False), get_norm_2d(norm_mode, 32), get_activation(act_mode))
        self.classifier = nn.Sequential(nn.Conv2d(288, 256, 3, padding=1, padding_mode=pad_mode, bias=False), get_norm_2d(norm_mode, 256), get_activation(act_mode), nn.Conv2d(256, num_classes, 1))

    def forward(self, x, low_level_feat):
        feat_shape = low_level_feat.shape[-2:]
        x = self.aspp(x)
        x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=True)
        low_level_feat = self.conv(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.classifier(x)
        return x


class FCNHead(nn.Sequential):

    def __init__(self, in_channels: int, channels: int, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', **_):
        inter_channels = in_channels // 4
        conv3x3 = nn.Conv2d(in_channels, inter_channels, 3, padding=1, padding_mode=pad_mode, bias=False)
        layers = [conv3x3, get_norm_2d(norm_mode, inter_channels), get_activation(act_mode), nn.Conv2d(inter_channels, channels, 1)]
        super(FCNHead, self).__init__(*layers)


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model, adapted
    from https://github.com/pytorch/vision/blob/master/torchvision/models/_utils.py.

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {'return_layers': Dict[str, str]}

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_. This implementation only
    supports 2D inputs. Pretrained ResNet weights on the ImgeaNet
    dataset is loaded by default. 

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    def __init__(self, name: str, backbone_type: str, out_channel: int=1, aux_out: bool=False, **kwargs):
        super().__init__()
        assert name in ['deeplabv3a', 'deeplabv3b', 'deeplabv3c']
        backbone = resnet.__dict__[backbone_type](pretrained=True, replace_stride_with_dilation=[False, True, True], **kwargs)
        return_layers = {'layer4': 'out'}
        if aux_out:
            return_layers['layer3'] = 'aux'
        if name == 'deeplabv3c':
            return_layers['layer1'] = 'low_level_feat'
        self.backbone = IntermediateLayerGetter(backbone, return_layers)
        self.aux_classifier = None
        if aux_out:
            inplanes = 1024
            self.aux_classifier = FCNHead(1024, out_channel, **kwargs)
        head_map = {'deeplabv3a': DeepLabHeadA, 'deeplabv3b': DeepLabHeadB, 'deeplabv3c': DeepLabHeadC}
        inplanes = 2048
        self.classifier = head_map[name](inplanes, out_channel, **kwargs)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        result = OrderedDict()
        x = features['out']
        if 'low_level_feat' in features.keys():
            feat = features['low_level_feat']
            x = self.classifier(x, feat)
        else:
            x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        result['out'] = x
        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            result['aux'] = x
        return result


def get_norm_3d(norm: str, out_channels: int, bn_momentum: float=0.1) ->nn.Module:
    """Get the specified normalization layer for a 3D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ['bn', 'sync_bn', 'gn', 'in', 'none'], 'Get unknown normalization layer key {}'.format(norm)
    if norm == 'gn':
        assert out_channels % 8 == 0, 'GN requires channels to separable into 8 groups'
    norm = {'bn': nn.BatchNorm3d, 'sync_bn': nn.SyncBatchNorm, 'in': nn.InstanceNorm3d, 'gn': lambda channels: nn.GroupNorm(8, channels), 'none': nn.Identity}[norm]
    if norm in ['bn', 'sync_bn', 'in']:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


def conv3d_norm_act(in_planes, planes, kernel_size=(3, 3, 3), stride=1, groups=1, dilation=(1, 1, 1), padding=(1, 1, 1), bias=False, pad_mode='replicate', norm_mode='bn', act_mode='relu', return_list=False):
    layers = []
    layers += [nn.Conv3d(in_planes, planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, padding_mode=pad_mode, dilation=dilation, bias=bias)]
    layers += [get_norm_3d(norm_mode, planes)]
    layers += [get_activation(act_mode)]
    if return_list:
        return layers
    return nn.Sequential(*layers)


class BasicBlock3d(nn.Module):

    def __init__(self, in_planes: int, planes: int, stride: Union[int, tuple]=1, dilation: int=1, groups: int=1, projection: bool=False, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', isotropic: bool=False):
        super(BasicBlock3d, self).__init__()
        if isotropic:
            kernel_size, padding = 3, dilation
        else:
            kernel_size, padding = (1, 3, 3), (0, dilation, dilation)
        self.conv = nn.Sequential(conv3d_norm_act(in_planes, planes, kernel_size=kernel_size, dilation=dilation, stride=stride, groups=groups, padding=padding, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode), conv3d_norm_act(planes, planes, kernel_size=kernel_size, dilation=dilation, stride=1, groups=groups, padding=padding, pad_mode=pad_mode, norm_mode=norm_mode, act_mode='none'))
        self.projector = nn.Identity()
        if in_planes != planes or stride != 1 or projection == True:
            self.projector = conv3d_norm_act(in_planes, planes, kernel_size=1, padding=0, stride=stride, norm_mode=norm_mode, act_mode='none')
        self.act = get_activation(act_mode)

    def forward(self, x):
        y = self.conv(x)
        y = y + self.projector(x)
        return self.act(y)


class SELayer3d(nn.Module):

    def __init__(self, channel, reduction=4, act_mode='relu'):
        super(SELayer3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), get_activation(act_mode), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class BasicBlock3dSE(BasicBlock3d):

    def __init__(self, in_planes, planes, act_mode='relu', **kwargs):
        super().__init__(in_planes=in_planes, planes=planes, act_mode=act_mode, **kwargs)
        self.conv = nn.Sequential(self.conv, SELayer3d(planes, act_mode=act_mode))


class AbsPosEmb(nn.Module):

    def __init__(self, fmap_size, dim_head):
        super().__init__()
        depth, height, width = fmap_size
        scale = dim_head ** -0.5
        self.depth = nn.Parameter(torch.randn(depth, dim_head) * scale)
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.depth, 'dp d -> dp () () d') + rearrange(self.height, 'h d -> () h () d') + rearrange(self.width, 'w d -> () () w d')
        emb = rearrange(emb, ' dp h w d -> (dp h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits


def expand_dims(t, dims, values):
    for d in dims:
        t = t.unsqueeze(dim=d)
    expand_shape = [-1] * len(t.shape)
    for d, k in zip(dims, values):
        expand_shape[d] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, l - 1:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, heads, z, y, x, dim = q.shape
    logits = einsum('b h z y x d, r d -> b h z y x r', q, rel_k)
    logits = rearrange(logits, 'b h z y x r -> b (h z y) x r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, z, y, x, x)
    logits = expand_dims(logits, dims=[3, 5], values=[z, y])
    return logits


class RelPosEmb(nn.Module):

    def __init__(self, fmap_size, dim_head):
        super().__init__()
        depth, height, width = fmap_size
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_depth = nn.Parameter(torch.randn(depth * 2 - 1, dim_head) * scale)
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        d, h, w = self.fmap_size
        q = rearrange(q, 'b h (z y x) d -> b h z y x d', z=d, y=h, x=w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h z z1 y y1 x x1 -> b h (z y x) (z1 y1 x1)')
        q = rearrange(q, 'b h z y x d -> b h z x y d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h z z1 x x1 y y1 -> b h (z y x) (z1 y1 x1)')
        q = rearrange(q, 'b h z x y d -> b h y x z d')
        rel_logits_d = relative_logits_1d(q, self.rel_depth)
        rel_logits_d = rearrange(rel_logits_d, 'b h y y1 x x1 z z1 -> b h (z y x) (z1 y1 x1)')
        return rel_logits_w + rel_logits_h + rel_logits_d


class Attention(nn.Module):

    def __init__(self, *, dim, fmap_size, heads=4, dim_head=128, rel_pos_emb=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias=False)
        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, d, h, w = self.heads, *fmap.shape
        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) z y x -> b h (z y x) d', h=heads), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (z y x) d -> b (h d) z y x', z=d, y=h, x=w)
        return out


class BottleBlock(nn.Module):

    def __init__(self, *, dim, fmap_size, dim_out, proj_factor, downsample, heads=4, dim_head=128, rel_pos_emb=False, activation=nn.ReLU(), pad_mode='replicate'):
        super().__init__()
        self.fmap_size = fmap_size
        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)
            self.shortcut = nn.Sequential(nn.Conv3d(dim, dim_out, kernel_size, stride=stride, padding=padding, bias=False, padding_mode=pad_mode), nn.BatchNorm3d(dim_out), activation)
        else:
            self.shortcut = nn.Identity()
        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head
        self.net = nn.Sequential(nn.Conv3d(dim, attn_dim_in, 1, bias=False), nn.BatchNorm3d(attn_dim_in), activation, Attention(dim=attn_dim_in, fmap_size=fmap_size, heads=heads, dim_head=dim_head, rel_pos_emb=rel_pos_emb), nn.AvgPool3d(kernel_size, stride=stride, padding=padding) if downsample else nn.Identity(), nn.BatchNorm3d(attn_dim_out), activation, nn.Conv3d(attn_dim_out, dim_out, 1, bias=False), nn.BatchNorm3d(dim_out))
        nn.init.zeros_(self.net[-1].weight)
        self.activation = activation

    def forward(self, x):
        _, _, d, h, w = x.shape
        assert d == self.fmap_size[0] and h == self.fmap_size[1] and w == self.fmap_size[2], f'depth, height, and width [{d} {h} {w}] of feature map must match the fmap_size given at init {self.fmap_size}'
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)


class BottleStack(nn.Module):

    def __init__(self, *, dim, fmap_size, dim_out=2048, proj_factor=4, num_layers=3, heads=4, dim_head=128, downsample=True, rel_pos_emb=False, activation=nn.ReLU()):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size
        layers = []
        for i in range(num_layers):
            is_first = i == 0
            dim = dim if is_first else dim_out
            layer_downsample = is_first and downsample
            fmap_divisor = 2 if downsample and not is_first else 1
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))
            layers.append(BottleBlock(dim=dim, fmap_size=layer_fmap_size, dim_out=dim_out, proj_factor=proj_factor, heads=heads, dim_head=dim_head, downsample=layer_downsample, rel_pos_emb=rel_pos_emb, activation=activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, d, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert d == self.fmap_size[0] and h == self.fmap_size[1] and w == self.fmap_size[2], f'depth, height, and width [{d} {h} {w}] of feature map must match the fmap_size given at init {self.fmap_size}'
        return self.net(x)


class BotNet3D(nn.Module):
    """BotNet backbone for 3D semantic/instance segmentation. 
       The global average pooling and fully-connected layer are removed.
    """
    block_dict = {'residual': BasicBlock3d, 'residual_se': BasicBlock3dSE}
    num_stages = 5

    def __init__(self, block_type='residual', in_channel: int=1, filters: List[int]=[28, 36, 48, 64, 80], blocks: List[int]=[2, 2, 2, 2], isotropy: List[bool]=[False, False, False, True, True], pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', fmap_size=[17, 129, 129], **_):
        super().__init__()
        assert len(filters) == self.num_stages
        self.block = self.block_dict[block_type]
        self.shared_kwargs = {'pad_mode': pad_mode, 'act_mode': act_mode, 'norm_mode': norm_mode}
        if isotropy[0]:
            kernel_size, padding = 5, 2
        else:
            kernel_size, padding = (1, 5, 5), (0, 2, 2)
        self.layer0 = conv3d_norm_act(in_channel, filters[0], kernel_size=kernel_size, padding=padding, **self.shared_kwargs)
        self.layer1 = self._make_layer(filters[0], filters[1], blocks[0], 2, isotropy[1])
        self.layer2 = self._make_layer(filters[1], filters[2], blocks[1], 2, isotropy[2])
        self.layer3 = self._make_layer(filters[2], filters[3], blocks[2], 2, isotropy[3])
        for iso in isotropy[1:-1]:
            if iso:
                fmap_size = [math.ceil(f / 2) for f in fmap_size]
            else:
                fmap_size = fmap_size[:1] + [math.ceil(f / 2) for f in fmap_size[1:]]
        self.layer4 = BottleStack(dim=filters[3], fmap_size=fmap_size, dim_out=filters[4], proj_factor=2, num_layers=3, heads=4, dim_head=32, downsample=True, activation=get_activation(act_mode))

    def _make_layer(self, in_planes: int, planes: int, blocks: int, stride: int=1, isotropic: bool=False):
        if stride == 2 and not isotropic:
            stride = 1, 2, 2
        layers = []
        layers.append(self.block(in_planes, planes, stride=stride, isotropic=isotropic, **self.shared_kwargs))
        for _ in range(1, blocks):
            layers.append(self.block(planes, planes, stride=1, isotropic=isotropic, **self.shared_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class PartialConv3d(nn.Conv3d):

    def __init__(self, *args, **kwargs):
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False
        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False
        super(PartialConv3d, self).__init__(*args, **kwargs)
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3] * self.weight_maskUpdater.shape[4]
        self.last_size = None, None, None, None, None
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 5
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater
                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3], input.data.shape[4])
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3], input.data.shape[4])
                else:
                    mask = mask_in
                self.update_mask = F.conv3d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-08)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
        raw_out = super(PartialConv3d, self).forward(torch.mul(input, mask_in) if mask_in is not None else input)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
        if self.return_mask:
            return output, self.update_mask
        else:
            return output


def get_conv(conv_type='standard'):
    assert conv_type in ['standard', 'partial']
    if conv_type == 'partial':
        return PartialConv3d
    return nn.Conv3d


class DilatedBlock(nn.Module):

    def __init__(self, conv_type, in_channel, inplanes, dilation_factors, pad_mode):
        super().__init__()
        self.conv = nn.ModuleList([get_conv(conv_type)(in_channel, inplanes, kernel_size=3, bias=False, stride=1, dilation=dilation_factors[i], padding=dilation_factors[i], padding_mode=pad_mode) for i in range(4)])

    def forward(self, x):
        return self._conv_and_cat(x, self.conv)

    def _conv_and_cat(self, x, conv_layers):
        y = [conv(x) for conv in conv_layers]
        return torch.cat(y, dim=1)


def dwconv1xkxk(planes, kernel_size=3, stride=1, dilation=1, conv_type='standard', padding_mode='zeros'):
    """1xkxk depthwise convolution with padding"""
    padding = (kernel_size - 1) * dilation // 2
    dilation = 1, dilation, dilation
    padding = 0, padding, padding
    stride = (1, stride, stride) if isinstance(stride, int) else stride
    return get_conv(conv_type)(planes, planes, kernel_size=(1, kernel_size, kernel_size), stride=stride, padding=padding, padding_mode=padding_mode, groups=planes, bias=False, dilation=dilation)


def dwconvkxkxk(planes, kernel_size=3, stride=1, dilation=1, conv_type='standard', padding_mode='zeros'):
    """kxkxk depthwise convolution with padding"""
    padding = (kernel_size - 1) * dilation // 2
    return get_conv(conv_type)(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=planes, bias=False, dilation=dilation)


class PlanePoolingAttention3D(nn.Module):
    """
    """
    reduction = 4

    def __init__(self, channel, act_mode='relu'):
        super(PlanePoolingAttention3D, self).__init__()
        self.channel = channel
        self.pool_zy = nn.AdaptiveAvgPool3d((None, None, 1))
        self.pool_yx = nn.AdaptiveAvgPool3d((1, None, None))
        self.pool_xz = nn.AdaptiveAvgPool3d((None, 1, None))
        self.conv_zy = nn.Conv3d(channel, channel // self.reduction, (3, 3, 1), padding=(1, 1, 0))
        self.conv_yx = nn.Conv3d(channel, channel // self.reduction, (1, 3, 3), padding=(0, 1, 1))
        self.conv_xz = nn.Conv3d(channel, channel // self.reduction, (3, 1, 3), padding=(1, 0, 1))
        self.relu = get_activation(act_mode)
        self.conv1x1x1 = nn.Conv3d(channel // self.reduction, channel, 1, bias=False)

    def forward(self, x):
        _, _, l, h, w = x.size()
        x1 = self.conv_zy(self.pool_zy(x)).expand(-1, -1, l, h, w)
        x2 = self.conv_yx(self.pool_yx(x)).expand(-1, -1, l, h, w)
        x3 = self.conv_xz(self.pool_xz(x)).expand(-1, -1, l, h, w)
        fusion = self.relu(x1) + self.relu(x2) + self.relu(x3)
        fusion = self.conv1x1x1(fusion).sigmoid()
        return x * fusion


class StripPoolingAttention3D(nn.Module):
    """
    """
    reduction = 4

    def __init__(self, channel, act_mode='relu'):
        super(StripPoolingAttention3D, self).__init__()
        self.channel = channel
        self.pool_z = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_y = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_x = nn.AdaptiveAvgPool3d((1, 1, None))
        self.conv_z = nn.Conv3d(channel, channel // self.reduction, (3, 1, 1), padding=(1, 0, 0))
        self.conv_y = nn.Conv3d(channel, channel // self.reduction, (1, 3, 1), padding=(0, 1, 0))
        self.conv_x = nn.Conv3d(channel, channel // self.reduction, (1, 1, 3), padding=(0, 0, 1))
        self.relu = get_activation(act_mode)
        self.conv1x1x1 = nn.Conv3d(channel // self.reduction, channel, 1, bias=False)

    def forward(self, x):
        _, _, l, h, w = x.size()
        x1 = self.conv_z(self.pool_z(x)).expand(-1, -1, l, h, w)
        x2 = self.conv_y(self.pool_y(x)).expand(-1, -1, l, h, w)
        x3 = self.conv_x(self.pool_x(x)).expand(-1, -1, l, h, w)
        fusion = self.relu(x1) + self.relu(x2) + self.relu(x3)
        fusion = self.conv1x1x1(fusion).sigmoid()
        return x * fusion


def make_att_3d(attention, channel, act_mode='relu'):
    if attention == 'strip_pool':
        return StripPoolingAttention3D(channel, act_mode)
    elif attention == 'plane_pool':
        return PlanePoolingAttention3D(channel, act_mode)
    elif attention == 'squeeze_excitation':
        return SELayer3d(channel, reduction=8, act_mode=act_mode)
    else:
        return nn.Identity()


class InvertedResidual(nn.Module):
    """3D Inverted Residual Block with Depth-wise Convolution"""

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor=1, attention=None, conv_type='standard', bn_momentum=0.1, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', isotropic: bool=False, bias: bool=False):
        super(InvertedResidual, self).__init__()
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        conv_layer = dwconvkxkxk if isotropic else dwconv1xkxk
        DWConv = conv_layer(mid_ch, kernel_size, stride, conv_type=conv_type, padding_mode=pad_mode)
        self.layers1 = nn.Sequential(nn.Conv3d(in_ch, mid_ch, 1, bias=False), get_norm_3d(norm_mode, mid_ch, bn_momentum), get_activation(act_mode), DWConv, get_norm_3d(norm_mode, mid_ch, bn_momentum), get_activation(act_mode))
        self.layers2 = nn.Sequential(nn.Conv3d(mid_ch, out_ch, 1, bias=bias), get_norm_3d(norm_mode, out_ch, bn_momentum))
        self.attention = make_att_3d(attention, mid_ch)
        self.projector = nn.Identity()
        if DWConv.stride != (1, 1, 1):
            self.projector = nn.Sequential(nn.AvgPool3d(DWConv.stride, DWConv.stride), conv3d_norm_act(in_ch, out_ch, kernel_size=1, padding=0, stride=1, norm_mode=norm_mode, act_mode='none'))
        elif in_ch != out_ch:
            self.projector = conv3d_norm_act(in_ch, out_ch, kernel_size=1, padding=0, stride=1, norm_mode=norm_mode, act_mode='none')
        else:
            self.projector = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.layers1(x)
        out = self.attention(out)
        out = self.layers2(out)
        if any([(out.shape[i] != identity.shape[i]) for i in range(2, 5)]):
            pad = []
            for i in range(2, 5):
                if out.shape[i] != identity.shape[i] and identity.shape[i] % 2 == 1:
                    pad.extend([1, 1])
                else:
                    pad.extend([0, 0])
            identity = F.pad(identity, pad[::-1], mode='replicate')
        out += self.projector(identity)
        return out


def get_dilated_dw_convs(channels: int=64, dilation_factors: List[int]=[1, 2, 4, 8], kernel_size: int=3, stride: int=1, conv_type: str='standard', pad_mode: str='zeros', isotropic: bool=False):
    assert channels % len(dilation_factors) == 0
    num_split = len(dilation_factors)
    conv_layer = dwconvkxkxk if isotropic else dwconv1xkxk
    return nn.ModuleList([conv_layer(channels // num_split, kernel_size, stride, conv_type=conv_type, padding_mode=pad_mode, dilation=dilation_factors[i]) for i in range(num_split)])


class InvertedResidualDilated(nn.Module):
    """3D Inverted Residual Block with Dilated Depth-wise Convolution"""
    dilation_factors = [1, 2, 4, 8]

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor=1, attention=None, conv_type='standard', bn_momentum=0.1, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', isotropic: bool=True, bias: bool=False):
        super(InvertedResidualDilated, self).__init__()
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.DWConv = get_dilated_dw_convs(mid_ch, self.dilation_factors, kernel_size, stride, conv_type, pad_mode, isotropic)
        self.layers1_a = nn.Sequential(nn.Conv3d(in_ch, mid_ch, 1, bias=False), get_norm_3d(norm_mode, mid_ch, bn_momentum), get_activation(act_mode))
        self.layers1_b = nn.Sequential(get_norm_3d(norm_mode, mid_ch, bn_momentum), get_activation(act_mode))
        self.layers2 = nn.Sequential(nn.Conv3d(mid_ch, out_ch, 1, bias=bias), get_norm_3d(norm_mode, out_ch, bn_momentum))
        self.attention = make_att_3d(attention, mid_ch)
        self.projector = nn.Identity()
        if self.DWConv[0].stride != (1, 1, 1):
            self.projector = nn.Sequential(nn.AvgPool3d(self.DWConv[0].stride, self.DWConv[0].stride), conv3d_norm_act(in_ch, out_ch, kernel_size=1, padding=0, stride=1, norm_mode=norm_mode, act_mode='none'))
        elif in_ch != out_ch:
            self.projector = conv3d_norm_act(in_ch, out_ch, kernel_size=1, padding=0, stride=1, norm_mode=norm_mode, act_mode='none')
        else:
            self.projector = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.layers1_a(x)
        out = self._split_conv_cat(out, self.DWConv)
        out = self.layers1_b(out)
        out = self.attention(out)
        out = self.layers2(out)
        if any([(out.shape[i] != identity.shape[i]) for i in range(2, 5)]):
            pad = []
            for i in range(2, 5):
                if out.shape[i] != identity.shape[i] and identity.shape[i] % 2 == 1:
                    pad.extend([1, 1])
                else:
                    pad.extend([0, 0])
            identity = F.pad(identity, pad[::-1], mode='replicate')
        out += self.projector(identity)
        return out

    def _split_conv_cat(self, x, conv_layers):
        _, c, _, _, _ = x.size()
        z = []
        y = torch.split(x, c // len(self.dilation_factors), dim=1)
        for i in range(len(self.dilation_factors)):
            z.append(conv_layers[i](y[i]))
        return torch.cat(z, dim=1)


def dw_stack(block, in_ch, out_ch, kernel_size, stride, repeats, isotropic, shared):
    """ Creates a stack of inverted residual blocks. 
    """
    assert repeats >= 1
    first = block(in_ch, out_ch, kernel_size, stride, isotropic=isotropic, **shared)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(block(out_ch, out_ch, kernel_size, 1, isotropic=isotropic, **shared))
    return nn.Sequential(first, *remaining)


class EfficientNet3D(nn.Module):
    """EfficientNet backbone for 3D semantic and instance segmentation.
    """
    expansion_factor = 1
    dilation_factors = [1, 2, 4, 8]
    num_stages = 5
    block_dict = {'inverted_res': InvertedResidual, 'inverted_res_dilated': InvertedResidualDilated}

    def __init__(self, block_type: str='inverted_res', in_channel: int=1, filters: List[int]=[32, 64, 96, 128, 160], blocks: List[int]=[1, 2, 2, 2, 4], ks: List[int]=[3, 3, 5, 3, 3], isotropy: List[bool]=[False, False, False, True, True], attention: str='squeeze_excitation', bn_momentum: float=0.01, conv_type: str='standard', pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', **_):
        super(EfficientNet3D, self).__init__()
        block = self.block_dict[block_type]
        self.inplanes = filters[0]
        if block == InvertedResidualDilated:
            self.all_dilated = True
            num_conv = len(self.dilation_factors)
            self.conv1 = DilatedBlock(conv_type, in_channel, self.inplanes // num_conv, self.dilation_factors, pad_mode)
        else:
            self.all_dilated = False
            self.conv1 = get_conv(conv_type)(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1, padding_mode=pad_mode, bias=False)
        self.bn1 = get_norm_3d(norm_mode, self.inplanes, bn_momentum)
        self.relu = get_activation(act_mode)
        shared_kwargs = {'expansion_factor': self.expansion_factor, 'bn_momentum': bn_momentum, 'norm_mode': norm_mode, 'attention': attention, 'pad_mode': pad_mode, 'act_mode': act_mode}
        self.layer0 = dw_stack(block, filters[0], filters[0], kernel_size=ks[0], stride=1, repeats=blocks[0], isotropic=isotropy[0], shared=shared_kwargs)
        self.layer1 = dw_stack(block, filters[0], filters[1], kernel_size=ks[1], stride=2, repeats=blocks[1], isotropic=isotropy[1], shared=shared_kwargs)
        self.layer2 = dw_stack(block, filters[1], filters[2], kernel_size=ks[2], stride=2, repeats=blocks[2], isotropic=isotropy[2], shared=shared_kwargs)
        self.layer3 = dw_stack(block, filters[2], filters[3], kernel_size=ks[3], stride=(1, 2, 2), repeats=blocks[3], isotropic=isotropy[3], shared=shared_kwargs)
        self.layer4 = dw_stack(block, filters[3], filters[4], kernel_size=ks[4], stride=2, repeats=blocks[4], isotropic=isotropy[4], shared=shared_kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def conv_bn_3d(in_planes, planes, kernel_size, stride, padding, groups=1, dilation=1, pad_mode='zeros'):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv3d(in_planes, planes, kernel_size, stride=stride, padding=padding, padding_mode=pad_mode, groups=groups, dilation=dilation, bias=False))
    result.add_module('bn', nn.BatchNorm3d(planes))
    return result


class RepVGGBlock3D(nn.Module):
    """ 3D RepVGG Block, adapted from:
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, pad_mode='zeros', act_mode='relu', isotropic=False, deploy=False):
        super(RepVGGBlock3D, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_planes = in_planes
        self.act = get_activation(act_mode)
        self.isotropic = isotropic
        padding = dilation
        if not self.isotropic:
            dilation = 1, dilation, dilation
            kernel_size, padding = (1, 3, 3), (0, padding, padding)
        if deploy:
            self.rbr_reparam = nn.Conv3d(in_planes, planes, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, padding_mode=pad_mode, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm3d(in_planes) if planes == in_planes and stride == 1 else None
            self.rbr_dense = conv_bn_3d(in_planes, planes, kernel_size, stride, padding, groups=groups, dilation=dilation, pad_mode=pad_mode)
            self.rbr_1x1 = conv_bn_3d(in_planes, planes, 1, stride, padding=0, groups=groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(x))
        identity = 0
        if self.rbr_identity is not None:
            identity = self.rbr_identity(x)
        x = self.rbr_dense(x) + self.rbr_1x1(x) + identity
        return self.act(x)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernel_id, bias3x3 + bias1x1 + bias_id

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        pad_size = [1, 1, 1, 1, 1, 1] if self.isotropic else [1, 1, 1, 1, 0, 0]
        return torch.nn.functional.pad(kernel1x1, pad_size)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm3d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_planes // self.groups
                z_dim = 3 if self.isotropic else 1
                z_idx = 1 if self.isotropic else 0
                kernel_value = torch.zeros((self.in_planes, input_dim, z_dim, 3, 3), dtype=torch.float, device=branch.weight.device)
                for i in range(self.in_planes):
                    kernel_value[i, i % input_dim, z_idx, 1, 1] = 1.0
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias

    def load_reparam_kernel(self, kernel, bias):
        assert hasattr(self, 'rbr_reparam')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias


class RepVGG3D(nn.Module):
    """RepVGG backbone for 3D semantic/instance segmentation. 
       The global average pooling and fully-connected layer are removed.
    """
    num_stages = 5
    block = RepVGGBlock3D

    def __init__(self, in_channel: int=1, filters: List[int]=[28, 36, 48, 64, 80], blocks: List[int]=[4, 4, 4, 4], isotropy: List[bool]=[False, False, False, True, True], pad_mode: str='replicate', act_mode: str='elu', deploy: bool=False, **_):
        super().__init__()
        assert len(filters) == self.num_stages
        self.shared_kwargs = {'deploy': deploy, 'pad_mode': pad_mode, 'act_mode': act_mode}
        self.layer0 = self._make_layer(in_channel, filters[0], 1, 1, isotropy[0])
        self.layer1 = self._make_layer(filters[0], filters[1], blocks[0], 2, isotropy[1])
        self.layer2 = self._make_layer(filters[1], filters[2], blocks[1], 2, isotropy[2])
        self.layer3 = self._make_layer(filters[2], filters[3], blocks[2], 2, isotropy[3])
        self.layer4 = self._make_layer(filters[3], filters[4], blocks[3], 2, isotropy[4])

    def _make_layer(self, in_planes: int, planes: int, blocks: int, stride: int=1, isotropic: bool=False):
        if stride == 2 and not isotropic:
            stride = 1, 2, 2
        layers = []
        layers.append(self.block(in_planes, planes, stride=stride, isotropic=isotropic, **self.shared_kwargs))
        for _ in range(1, blocks):
            layers.append(self.block(planes, planes, stride=1, isotropic=isotropic, **self.shared_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def repvgg_convert_model(self):
        converted_weights = {}
        for name, module in self.named_modules():
            if hasattr(module, 'repvgg_convert'):
                kernel, bias = module.repvgg_convert()
                converted_weights[name + '.rbr_reparam.weight'] = kernel
                converted_weights[name + '.rbr_reparam.bias'] = bias
        return converted_weights

    def load_reparam_model(self, converted_weights):
        for name, param in self.named_parameters():
            if name in converted_weights.keys():
                param.data = converted_weights[name]

    @staticmethod
    def repvgg_convert_as_backbone(train_dict):
        deploy_dict = copy.deepcopy(train_dict)
        for name in train_dict.keys():
            name_split = name.split('.')
            if name in deploy_dict and name_split[0] == 'backbone' and name_split[3] == 'rbr_dense':
                sz = deploy_dict[name].shape
                in_planes, planes, isotropic = sz[1], sz[0], sz[2] == 3
                repvgg_block = RepVGGBlock3D(in_planes, planes, isotropic=isotropic)
                prefix = '.'.join(name_split[:3])
                temp_dict = {}
                for key in repvgg_block.state_dict().keys():
                    w_name = prefix + '.' + key
                    temp_dict[key] = deploy_dict[w_name]
                    del deploy_dict[w_name]
                repvgg_block.load_state_dict(temp_dict)
                kernel, bias = repvgg_block.repvgg_convert()
                deploy_dict[prefix + '.rbr_reparam.weight'] = kernel
                deploy_dict[prefix + '.rbr_reparam.bias'] = bias
        return deploy_dict


class ResNet3D(nn.Module):
    """ResNet backbone for 3D semantic/instance segmentation. 
       The global average pooling and fully-connected layer are removed.
    """
    block_dict = {'residual': BasicBlock3d, 'residual_se': BasicBlock3dSE}
    num_stages = 5

    def __init__(self, block_type: str='residual', in_channel: int=1, filters: List[int]=[28, 36, 48, 64, 80], blocks: List[int]=[2, 2, 2, 2], isotropy: List[bool]=[False, False, False, True, True], pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', **_):
        super().__init__()
        assert len(filters) == self.num_stages
        self.block = self.block_dict[block_type]
        self.shared_kwargs = {'pad_mode': pad_mode, 'act_mode': act_mode, 'norm_mode': norm_mode}
        if isotropy[0]:
            kernel_size, padding = 5, 2
        else:
            kernel_size, padding = (1, 5, 5), (0, 2, 2)
        self.layer0 = conv3d_norm_act(in_channel, filters[0], kernel_size=kernel_size, padding=padding, **self.shared_kwargs)
        self.layer1 = self._make_layer(filters[0], filters[1], blocks[0], 2, isotropy[1])
        self.layer2 = self._make_layer(filters[1], filters[2], blocks[1], 2, isotropy[2])
        self.layer3 = self._make_layer(filters[2], filters[3], blocks[2], 2, isotropy[3])
        self.layer4 = self._make_layer(filters[3], filters[4], blocks[3], 2, isotropy[4])

    def _make_layer(self, in_planes: int, planes: int, blocks: int, stride: int=1, isotropic: bool=False):
        if stride == 2 and not isotropic:
            stride = 1, 2, 2
        layers = []
        layers.append(self.block(in_planes, planes, stride=stride, isotropic=isotropic, **self.shared_kwargs))
        for _ in range(1, blocks):
            layers.append(self.block(planes, planes, stride=1, isotropic=isotropic, **self.shared_kwargs))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


backbone_dict = {'resnet': ResNet3D, 'repvgg': RepVGG3D, 'botnet': BotNet3D, 'efficientnet': EfficientNet3D}


def build_backbone(backbone_type: str, feat_keys: List[str], **kwargs):
    assert backbone_type in ['resnet', 'repvgg', 'botnet', 'efficientnet']
    return_layers = {'layer0': feat_keys[0], 'layer1': feat_keys[1], 'layer2': feat_keys[2], 'layer3': feat_keys[3], 'layer4': feat_keys[4]}
    backbone = backbone_dict[backbone_type](**kwargs)
    assert len(feat_keys) == backbone.num_stages
    return IntermediateLayerGetter(backbone, return_layers)


def kaiming_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')


def ortho_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(m.weight)


def selu_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal(m.weight, 0, sqrt(1.0 / fan_in))
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal(m.weight, 0, sqrt(1.0 / fan_in))


def xavier_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def model_init(model, mode='orthogonal'):
    """Initialization of model weights.
    """
    model_init_dict = {'xavier': xavier_init, 'kaiming': kaiming_init, 'selu': selu_init, 'orthogonal': ortho_init}
    model.apply(model_init_dict[mode])


class FPN3D(nn.Module):
    """3D feature pyramid network (FPN). This design is flexible in handling both isotropic data and anisotropic data.

    Args:
        backbone_type (str): the block type at each U-Net stage. Default: ``'resnet'``
        block_type (str): the block type in the backbone. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each FPN stage. Default: [28, 36, 48, 64, 80]
        is_isotropic (bool): whether the whole model is isotropic. Default: False
        isotropy (List[bool]): specify each U-Net stage is isotropic or anisotropic. All elements will
            be `True` if :attr:`is_isotropic` is `True`. Default: [False, False, False, True, True]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'relu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'bn'``
        init_mode (str): one of ``'xavier'``, ``'kaiming'``, ``'selu'`` or ``'orthogonal'``. Default: ``'orthogonal'``
        deploy (bool): build backbone in deploy mode (exclusive for RepVGG backbone). Default: False
    """

    def __init__(self, backbone_type: str='resnet', block_type: str='residual', feature_keys: List[str]=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'], in_channel: int=1, out_channel: int=3, filters: List[int]=[28, 36, 48, 64, 80], ks: List[int]=[3, 3, 5, 3, 3], blocks: List[int]=[2, 2, 2, 2, 2], attn: str='squeeze_excitation', is_isotropic: bool=False, isotropy: List[bool]=[False, False, False, True, True], pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', init_mode: str='orthogonal', deploy: bool=False, fmap_size=[17, 129, 129], **kwargs):
        super().__init__()
        self.filters = filters
        self.depth = len(filters)
        assert len(isotropy) == self.depth
        if is_isotropic:
            isotropy = [True] * self.depth
        self.isotropy = isotropy
        self.shared_kwargs = {'pad_mode': pad_mode, 'act_mode': act_mode, 'norm_mode': norm_mode}
        backbone_kwargs = {'block_type': block_type, 'in_channel': in_channel, 'filters': filters, 'isotropy': isotropy, 'blocks': blocks, 'deploy': deploy, 'fmap_size': fmap_size, 'ks': ks, 'attention': attn}
        backbone_kwargs.update(self.shared_kwargs)
        self.backbone = build_backbone(backbone_type, feature_keys, **backbone_kwargs)
        self.feature_keys = feature_keys
        self.latplanes = filters[0]
        self.latlayers = nn.ModuleList([conv3d_norm_act(x, self.latplanes, kernel_size=1, padding=0, **self.shared_kwargs) for x in filters])
        self.smooth = nn.ModuleList()
        for i in range(self.depth):
            kernel_size, padding = self._get_kernel_size(isotropy[i])
            self.smooth.append(conv3d_norm_act(self.latplanes, self.latplanes, kernel_size=kernel_size, padding=padding, **self.shared_kwargs))
        self.conv_out = self._get_io_conv(out_channel, isotropy[0])
        model_init(self, init_mode)

    def forward(self, x):
        z = self.backbone(x)
        return self._forward_main(z)

    def _forward_main(self, z):
        features = [self.latlayers[i](z[self.feature_keys[i]]) for i in range(self.depth)]
        out = features[self.depth - 1]
        for j in range(self.depth - 1):
            i = self.depth - 1 - j
            out = self._up_smooth_add(out, features[i - 1], self.smooth[i])
        out = self.smooth[0](out)
        out = self.conv_out(out)
        return out

    def _up_smooth_add(self, x, y, smooth):
        """Upsample, smooth and add two feature maps.
        """
        x = F.interpolate(x, size=y.shape[2:], mode='trilinear', align_corners=True)
        return smooth(x) + y

    def _get_kernel_size(self, is_isotropic, io_layer=False):
        if io_layer:
            if is_isotropic:
                return (5, 5, 5), (2, 2, 2)
            return (1, 5, 5), (0, 2, 2)
        if is_isotropic:
            return (3, 3, 3), (1, 1, 1)
        return (1, 3, 3), (0, 1, 1)

    def _get_io_conv(self, out_channel, is_isotropic):
        kernel_size_io, padding_io = self._get_kernel_size(is_isotropic, io_layer=True)
        return conv3d_norm_act(self.filters[0], out_channel, kernel_size_io, padding=padding_io, pad_mode=self.shared_kwargs['pad_mode'], bias=True, act_mode='none', norm_mode='none')


class Discriminator3D(nn.Module):
    """3D PatchGAN discriminator

    Args:
        in_channel (int): number of input channels. Default: 1
        filters (List[int]): number of filters at each U-Net stage. Default: [32, 64, 96, 96, 96]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'`
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'elu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'in'``
        dilation (int): dilation rate of the conv kernels. Default: 1
        is_isotropic (bool): whether the whole model is isotropic. Default: False
        isotropy (List[bool]): specify each discriminator layer is isotropic or anisotropic. All elements will
            be `True` if :attr:`is_isotropic` is `True`. Default: [False, False, False, True, True]
        stride_list (List[int]): list of strides for each conv layer. Default: [2, 2, 2, 2, 1]
    """

    def __init__(self, in_channel: int=1, filters: List[int]=[64, 64, 128, 128, 256], pad_mode: str='replicate', act_mode: str='leaky_relu', norm_mode: str='in', dilation: int=1, is_isotropic: bool=False, isotropy: List[bool]=[False, False, False, True, True], stride_list: List[int]=[2, 2, 2, 2, 1]) ->None:
        super().__init__()
        self.depth = len(filters)
        if is_isotropic:
            isotropy = [True] * self.depth
        assert len(filters) == len(isotropy)
        for i in range(self.depth):
            if not isotropy[i] and stride_list[i] == 2:
                stride_list[i] = 1, 2, 2
        use_bias = True if norm_mode == 'none' else False
        dilation_base = dilation
        ks, padding, dilation = self._get_kernal_size(5, isotropy[0], dilation_base)
        sequence = [nn.Conv3d(in_channel, filters[0], kernel_size=ks, stride=stride_list[0], padding=padding, padding_mode=pad_mode, dilation=dilation, bias=use_bias), get_norm_3d(norm_mode, filters[0]), get_activation(act_mode)]
        for n in range(1, self.depth):
            ks, padding, dilation = self._get_kernal_size(3, isotropy[n], dilation_base)
            sequence += [nn.Conv3d(filters[n - 1], filters[n], kernel_size=ks, stride=stride_list[n], padding=padding, padding_mode=pad_mode, dilation=dilation, bias=use_bias), get_norm_3d(norm_mode, filters[n]), get_activation(act_mode)]
        ks, padding, _ = self._get_kernal_size(3, True, 1)
        sequence += [nn.Conv3d(filters[-1], 1, kernel_size=ks, stride=1, padding=padding, padding_mode=pad_mode, bias=True)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

    def _get_kernal_size(self, ks: int, is_isotropic: bool, dilation: int=1):
        assert ks >= 3
        padding = (ks + (ks - 1) * (dilation - 1)) // 2
        if is_isotropic:
            return ks, padding, dilation
        return (1, ks, ks), (0, padding, padding), (1, dilation, dilation)


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: Type[LayerNorm]=nn.LayerNorm, spatial_dims: int=3) ->None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """
        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = h % 2 == 1 or w % 2 == 1 or d % 2 == 1
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 0::2, 1::2, :]
            x5 = x[:, 1::2, 1::2, 0::2, :]
            x6 = x[:, 0::2, 1::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = h % 2 == 1 or w % 2 == 1
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f'expecting 5D x, got {x.shape}.')
        b, d, h, w, c = x_shape
        pad_input = h % 2 == 1 or w % 2 == 1 or d % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {'merging': PatchMerging, 'mergingv2': PatchMergingV2}


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, num_heads: int, window_size: Sequence[int], qkv_bias: bool=False, attn_drop: float=0.0, proj_drop: float=0.0) ->None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__
        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1), num_heads))
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.clone()[:n, :n].reshape(-1)].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(b, d // window_size[0], window_size[0], h // window_size[1], window_size[1], w // window_size[2], window_size[2], c)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(b, d // window_size[0], h // window_size[1], w // window_size[2], window_size[0], window_size[1], window_size[2], -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, num_heads: int, window_size: Sequence[int], shift_size: Sequence[int], mlp_ratio: float=4.0, qkv_bias: bool=True, drop: float=0.0, attn_drop: float=0.0, drop_path: float=0.0, act_layer: str='GELU', norm_layer: Type[LayerNorm]=nn.LayerNorm, use_checkpoint: bool=False) ->None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode='swin')

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]
        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f'module.{layer}.0.blocks.{n_block}.'
        block_names = ['norm1.weight', 'norm1.bias', 'attn.relative_position_bias_table', 'attn.relative_position_index', 'attn.qkv.weight', 'attn.qkv.bias', 'attn.proj.weight', 'attn.proj.bias', 'norm2.weight', 'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias']
        with torch.no_grad():
            self.norm1.weight.copy_(weights['state_dict'][root + block_names[0]])
            self.norm1.bias.copy_(weights['state_dict'][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights['state_dict'][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights['state_dict'][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights['state_dict'][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights['state_dict'][root + block_names[5]])
            self.attn.proj.weight.copy_(weights['state_dict'][root + block_names[6]])
            self.attn.proj.bias.copy_(weights['state_dict'][root + block_names[7]])
            self.norm2.weight.copy_(weights['state_dict'][root + block_names[8]])
            self.norm2.bias.copy_(weights['state_dict'][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights['state_dict'][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights['state_dict'][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights['state_dict'][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights['state_dict'][root + block_names[13]])

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """
    cnt = 0
    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
            for h in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
                for w in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
            for w in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: Sequence[int], drop_path: list, mlp_ratio: float=4.0, qkv_bias: bool=False, drop: float=0.0, attn_drop: float=0.0, norm_layer: Type[LayerNorm]=nn.LayerNorm, downsample: Optional[nn.Module]=None, use_checkpoint: bool=False) ->None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=self.window_size, shift_size=self.no_shift if i % 2 == 0 else self.shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, use_checkpoint=use_checkpoint) for i in range(depth)])
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, 'b c d h w -> b d h w c')
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, 'b c h w -> b h w c')
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, 'b h w c -> b c h w')
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, in_chans: int, embed_dim: int, window_size: Sequence[int], patch_size: Sequence[int], depths: Sequence[int], num_heads: Sequence[int], mlp_ratio: float=4.0, qkv_bias: bool=True, drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0, norm_layer: Type[LayerNorm]=nn.LayerNorm, patch_norm: bool=False, use_checkpoint: bool=False, spatial_dims: int=3, downsample='merging') ->None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        """
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None, spatial_dims=spatial_dims)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=self.window_size, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, downsample=down_sample_mod, use_checkpoint=use_checkpoint)
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, 'n c d h w -> n d h w c')
                x = F.layer_norm(x, [ch])
                x = rearrange(x, 'n d h w c -> n c d h w')
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, 'n c h w -> n h w c')
                x = F.layer_norm(x, [ch])
                x = rearrange(x, 'n h w c -> n c h w')
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]


def get_output_padding(kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]) ->Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError('out_padding value should not be negative, please change the kernel size and/or stride.')
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_padding(kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]) ->Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError('padding value should not be negative, please change the kernel size and/or stride.')
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


class UnetOutBlock(nn.Module):

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]]=None):
        super().__init__()
        self.conv = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, act=None, norm=None, conv_only=False)

    def forward(self, inp):
        return self.conv(inp)


class UnetBasicBlock(nn.Module):
    """
    A CNN module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], norm_name: Union[Tuple, str], act_name: Union[Tuple, str]=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), dropout: Optional[Union[Tuple, str, float]]=None):
        super().__init__()
        self.conv1 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout=dropout, act=None, norm=None, conv_only=False)
        self.conv2 = get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, act=None, norm=None, conv_only=False)
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], norm_name: Union[Tuple, str], act_name: Union[Tuple, str]=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), dropout: Optional[Union[Tuple, str, float]]=None):
        super().__init__()
        self.conv1 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout=dropout, act=None, norm=None, conv_only=False)
        self.conv2 = get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, act=None, norm=None, conv_only=False)
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, act=None, norm=None, conv_only=False)
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, 'conv3'):
            residual = self.conv3(residual)
        if hasattr(self, 'norm3'):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], norm_name: Union[Tuple, str], res_block: bool=False) ->None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """
        super().__init__()
        if res_block:
            self.layer = UnetResBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, norm_name=norm_name)
        else:
            self.layer = UnetBasicBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, norm_name=norm_name)

    def forward(self, inp):
        return self.layer(inp)


class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Union[Sequence[int], int], upsample_kernel_size: Union[Sequence[int], int], norm_name: Union[Tuple, str], res_block: bool=False) ->None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_stride, conv_only=True, is_transposed=True)
        if res_block:
            self.conv_block = UnetResBlock(spatial_dims, out_channels + out_channels, out_channels, kernel_size=kernel_size, stride=1, norm_name=norm_name)
        else:
            self.conv_block = UnetBasicBlock(spatial_dims, out_channels + out_channels, out_channels, kernel_size=kernel_size, stride=1, norm_name=norm_name)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(self, img_size: Union[Sequence[int], int], in_channel: int, out_channel: int, depths: Sequence[int]=(2, 2, 2, 2), num_heads: Sequence[int]=(3, 6, 12, 24), feature_size: int=24, norm_name: Union[Tuple, str]='instance', drop_rate: float=0.0, attn_drop_rate: float=0.0, dropout_path_rate: float=0.0, normalize: bool=True, use_checkpoint: bool=False, spatial_dims: int=3, downsample='merging', **kwargs) ->None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` 
        """
        in_channels = in_channel
        out_channels = out_channel
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError('spatial dimension should be 2 or 3.')
        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError('input image size (img_size) should be divisible by stage-wise image resolution.')
        if not 0 <= drop_rate <= 1:
            raise ValueError('dropout rate should be between 0 and 1.')
        if not 0 <= attn_drop_rate <= 1:
            raise ValueError('attention dropout rate should be between 0 and 1.')
        if not 0 <= dropout_path_rate <= 1:
            raise ValueError('drop path rate should be between 0 and 1.')
        if feature_size % 12 != 0:
            raise ValueError('feature_size should be divisible by 12.')
        self.normalize = normalize
        self.swinViT = SwinTransformer(in_chans=in_channels, embed_dim=feature_size, window_size=window_size, patch_size=patch_size, depths=depths, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dropout_path_rate, norm_layer=nn.LayerNorm, use_checkpoint=use_checkpoint, spatial_dims=spatial_dims, downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample)
        self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
        self.encoder2 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
        self.encoder3 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=2 * feature_size, out_channels=2 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
        self.encoder4 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=4 * feature_size, out_channels=4 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
        self.encoder10 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=16 * feature_size, out_channels=16 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
        self.decoder5 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=16 * feature_size, out_channels=8 * feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        self.decoder4 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        self.decoder3 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        self.decoder2 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=feature_size * 2, out_channels=feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        self.decoder1 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights['state_dict']['module.patch_embed.proj.weight'])
            self.swinViT.patch_embed.proj.bias.copy_(weights['state_dict']['module.patch_embed.proj.bias'])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer='layers1')
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(weights['state_dict']['module.layers1.0.downsample.reduction.weight'])
            self.swinViT.layers1[0].downsample.norm.weight.copy_(weights['state_dict']['module.layers1.0.downsample.norm.weight'])
            self.swinViT.layers1[0].downsample.norm.bias.copy_(weights['state_dict']['module.layers1.0.downsample.norm.bias'])
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer='layers2')
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(weights['state_dict']['module.layers2.0.downsample.reduction.weight'])
            self.swinViT.layers2[0].downsample.norm.weight.copy_(weights['state_dict']['module.layers2.0.downsample.norm.weight'])
            self.swinViT.layers2[0].downsample.norm.bias.copy_(weights['state_dict']['module.layers2.0.downsample.norm.bias'])
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer='layers3')
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(weights['state_dict']['module.layers3.0.downsample.reduction.weight'])
            self.swinViT.layers3[0].downsample.norm.weight.copy_(weights['state_dict']['module.layers3.0.downsample.norm.weight'])
            self.swinViT.layers3[0].downsample.norm.bias.copy_(weights['state_dict']['module.layers3.0.downsample.norm.bias'])
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer='layers4')
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(weights['state_dict']['module.layers4.0.downsample.reduction.weight'])
            self.swinViT.layers4[0].downsample.norm.weight.copy_(weights['state_dict']['module.layers4.0.downsample.norm.weight'])
            self.swinViT.layers4[0].downsample.norm.bias.copy_(weights['state_dict']['module.layers4.0.downsample.norm.bias'])

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits


def norm_act_conv3d(in_planes, planes, kernel_size=(3, 3, 3), stride=1, groups=1, dilation=(1, 1, 1), padding=(1, 1, 1), bias=False, pad_mode='replicate', norm_mode='bn', act_mode='relu', return_list=False):
    layers = []
    layers += [get_norm_3d(norm_mode, in_planes)]
    layers += [get_activation(act_mode)]
    layers += [nn.Conv3d(in_planes, planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, padding_mode=pad_mode, dilation=dilation, bias=bias)]
    if return_list:
        return layers
    return nn.Sequential(*layers)


class BasicBlock3dPA(nn.Module):
    """Pre-activation 3D basic residual block.
    """

    def __init__(self, in_planes: int, planes: int, stride: Union[int, tuple]=1, dilation: int=1, groups: int=1, projection: bool=False, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', isotropic: bool=False):
        super(BasicBlock3dPA, self).__init__()
        if isotropic:
            kernel_size, padding = 3, dilation
        else:
            kernel_size, padding = (1, 3, 3), (0, dilation, dilation)
        self.conv = nn.Sequential(norm_act_conv3d(in_planes, planes, kernel_size=kernel_size, dilation=dilation, stride=stride, groups=groups, padding=padding, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode), norm_act_conv3d(planes, planes, kernel_size=kernel_size, dilation=dilation, stride=1, groups=groups, padding=padding, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode))
        self.projector = nn.Identity()
        if in_planes != planes or stride != 1 or projection == True:
            self.projector = conv3d_norm_act(in_planes, planes, kernel_size=1, padding=0, stride=stride, norm_mode=norm_mode, act_mode='none')

    def forward(self, x):
        y = self.conv(x)
        x = y + self.projector(x)
        return x


class BasicBlock3dPASE(BasicBlock3dPA):

    def __init__(self, in_planes, planes, act_mode='relu', **kwargs):
        super().__init__(in_planes=in_planes, planes=planes, act_mode=act_mode, **kwargs)
        self.conv = nn.Sequential(self.conv, SELayer3d(planes, act_mode=act_mode))


class UNet3D(nn.Module):
    """3D residual U-Net architecture. This design is flexible in handling both isotropic data and anisotropic data.

    Args:
        block_type (str): the block type at each U-Net stage. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each U-Net stage. Default: [28, 36, 48, 64, 80]
        is_isotropic (bool): whether the whole model is isotropic. Default: False
        isotropy (List[bool]): specify each U-Net stage is isotropic or anisotropic. All elements will
            be `True` if :attr:`is_isotropic` is `True`. Default: [False, False, False, True, True]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'relu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'bn'``
        init_mode (str): one of ``'xavier'``, ``'kaiming'``, ``'selu'`` or ``'orthogonal'``. Default: ``'orthogonal'``
        pooling (bool): downsample by max-pooling if `True` else using stride. Default: `False`
        blurpool (bool): apply blurpool as in Zhang 2019 (https://arxiv.org/abs/1904.11486). Default: `False`
    """
    block_dict = {'residual': BasicBlock3d, 'residual_pa': BasicBlock3dPA, 'residual_se': BasicBlock3dSE, 'residual_se_pa': BasicBlock3dPASE}

    def __init__(self, block_type='residual', in_channel: int=1, out_channel: int=3, filters: List[int]=[28, 36, 48, 64, 80], is_isotropic: bool=False, isotropy: List[bool]=[False, False, False, True, True], pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', init_mode: str='orthogonal', pooling: bool=False, blurpool: bool=False, return_feats: Optional[list]=None, **kwargs):
        super().__init__()
        self.depth = len(filters)
        self.do_return_feats = return_feats is not None
        self.return_feats = return_feats
        None
        if is_isotropic:
            isotropy = [True] * self.depth
        assert len(filters) == len(isotropy)
        block = self.block_dict[block_type]
        self.pooling, self.blurpool = pooling, blurpool
        self.shared_kwargs = {'pad_mode': pad_mode, 'act_mode': act_mode, 'norm_mode': norm_mode}
        kernel_size_io, padding_io = self._get_kernal_size(is_isotropic, io_layer=True)
        self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size_io, padding=padding_io, **self.shared_kwargs)
        self.conv_out = conv3d_norm_act(filters[0], out_channel, kernel_size_io, bias=True, padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='none')
        self.down_layers = nn.ModuleList()
        for i in range(self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[i])
            previous = max(0, i - 1)
            stride = self._get_stride(isotropy[i], previous, i)
            layer = nn.Sequential(self._make_pooling_layer(isotropy[i], previous, i), conv3d_norm_act(filters[previous], filters[i], kernel_size, stride=stride, padding=padding, **self.shared_kwargs), block(filters[i], filters[i], **self.shared_kwargs))
            self.down_layers.append(layer)
        self.up_layers = nn.ModuleList()
        for j in range(1, self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[j])
            layer = nn.ModuleList([conv3d_norm_act(filters[j], filters[j - 1], kernel_size, padding=padding, **self.shared_kwargs), block(filters[j - 1], filters[j - 1], **self.shared_kwargs)])
            self.up_layers.append(layer)
        model_init(self, mode=init_mode)

    def forward(self, x):
        x = self.conv_in(x)
        down_x = [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            x = self.down_layers[i](x)
            down_x[i] = x
        x = self.down_layers[-1](x)
        self._maybe_collect_feat(x, restart=True)
        for j in range(self.depth - 1):
            i = self.depth - 2 - j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self.up_layers[i][1](x)
            self._maybe_collect_feat(x)
        x = self.conv_out(x)
        if self.do_return_feats:
            return x, self.feats
        return x

    def _maybe_collect_feat(self, x, restart: bool=False):
        """Collect U-Net features at different pyramid levels."""
        if not self.do_return_feats:
            return
        if restart:
            self.feats = OrderedDict()
            self.feat_index = -1
        self.feat_index += 1
        if self.feat_index in self.return_feats:
            self.feats[self.feat_index] = x

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        When pooling layer is used, the input size is assumed to be even,
        therefore :attr:`align_corners` is set to `False` to avoid feature
        mis-match. When downsampling by stride, the input size is assumed
        to be 2n+1, and :attr:`align_corners` is set to `True`.
        """
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y.shape[2:], mode='trilinear', align_corners=align_corners)
        return x + y

    def _get_kernal_size(self, is_isotropic, io_layer=False):
        if io_layer:
            if is_isotropic:
                return (5, 5, 5), (2, 2, 2)
            return (1, 5, 5), (0, 2, 2)
        if is_isotropic:
            return (3, 3, 3), (1, 1, 1)
        return (1, 3, 3), (0, 1, 1)

    def _get_stride(self, is_isotropic, previous, i):
        if self.pooling or previous == i:
            return 1
        return self._get_downsample(is_isotropic)

    def _get_downsample(self, is_isotropic):
        if not is_isotropic:
            return 1, 2, 2
        return 2

    def _make_pooling_layer(self, is_isotropic, previous, i):
        if self.pooling and previous != i:
            kernel_size = stride = self._get_downsample(is_isotropic)
            return nn.MaxPool3d(kernel_size, stride)
        return nn.Identity()


def get_norm_1d(norm: str, out_channels: int, bn_momentum: float=0.1) ->nn.Module:
    """Get the specified normalization layer for a 1D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ['bn', 'sync_bn', 'gn', 'in', 'none'], 'Get unknown normalization layer key {}'.format(norm)
    norm = {'bn': nn.BatchNorm1d, 'sync_bn': nn.BatchNorm1d, 'in': nn.InstanceNorm1d, 'gn': lambda channels: nn.GroupNorm(16, channels), 'none': nn.Identity}[norm]
    if norm in ['bn', 'sync_bn', 'in']:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


class _NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, norm_layer=True, norm_mode='bn'):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=3, stride=2)
            get_norm_func = get_norm_3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=3, stride=2)
            get_norm_func = get_norm_2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=3, stride=2)
            get_norm_func = get_norm_1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        if norm_layer:
            self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0), get_norm_func(norm_mode, self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class NonLocalBlock3D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, norm_layer=True, norm_mode='bn'):
        super(NonLocalBlock3D, self).__init__(in_channels, inter_channels, dimension=3, sub_sample=sub_sample, norm_layer=norm_layer, norm_mode=norm_mode)


class UNetPlus3D(UNet3D):

    def __init__(self, filters: List[int]=[28, 36, 48, 64, 80], norm_mode: str='bn', **kwargs):
        super().__init__(filters=filters, norm_mode=norm_mode, **kwargs)
        self.feat_layers = nn.ModuleList([conv3d_norm_act(filters[-1], filters[k - 1], 1, **self.shared_kwargs) for k in range(1, self.depth)])
        self.non_local = NonLocalBlock3D(filters[-1], sub_sample=False, norm_mode=norm_mode)

    def forward(self, x):
        x = self.conv_in(x)
        down_x = [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            x = self.down_layers[i](x)
            down_x[i] = x
        x = self.down_layers[-1](x)
        x = self.non_local(x)
        feat = x
        self._maybe_collect_feat(x, restart=True)
        for j in range(self.depth - 1):
            i = self.depth - 2 - j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self._upsample_add(self.feat_layers[i](feat), x)
            x = self.up_layers[i][1](x)
            self._maybe_collect_feat(x)
        x = self.conv_out(x)
        if self.do_return_feats:
            return x, self.feats
        return x


def conv2d_norm_act(in_planes, planes, kernel_size=(3, 3), stride=1, groups=1, dilation=(1, 1), padding=(1, 1), bias=False, pad_mode='replicate', norm_mode='bn', act_mode='relu', return_list=False):
    layers = [nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, groups=groups, dilation=dilation, padding=padding, padding_mode=pad_mode, bias=bias)]
    layers += [get_norm_2d(norm_mode, planes)]
    layers += [get_activation(act_mode)]
    if return_list:
        return layers
    return nn.Sequential(*layers)


class BasicBlock2d(nn.Module):

    def __init__(self, in_planes: int, planes: int, stride: int=1, dilation: int=1, groups: int=1, projection: bool=False, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn'):
        super(BasicBlock2d, self).__init__()
        self.conv = nn.Sequential(conv2d_norm_act(in_planes, planes, kernel_size=3, dilation=dilation, stride=stride, groups=groups, padding=dilation, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode), conv2d_norm_act(planes, planes, kernel_size=3, dilation=dilation, stride=1, groups=groups, padding=dilation, pad_mode=pad_mode, norm_mode=norm_mode, act_mode='none'))
        self.projector = nn.Identity()
        if in_planes != planes or stride != 1 or projection == True:
            self.projector = conv2d_norm_act(in_planes, planes, kernel_size=1, padding=0, stride=stride, norm_mode=norm_mode, act_mode='none')
        self.act = get_activation(act_mode)

    def forward(self, x):
        y = self.conv(x)
        y = y + self.projector(x)
        return self.act(y)


class SELayer2d(nn.Module):

    def __init__(self, channel, reduction=16, act_mode='relu'):
        super(SELayer2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), get_activation(act_mode), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock2dSE(BasicBlock2d):

    def __init__(self, in_planes, planes, act_mode='relu', **kwargs):
        super().__init__(in_planes=in_planes, planes=planes, act_mode=act_mode, **kwargs)
        self.conv = nn.Sequential(self.conv, SELayer2d(planes, act_mode=act_mode))


class UNet2D(nn.Module):
    """2D residual U-Net architecture.

    Args:
        block_type (str): the block type at each U-Net stage. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each U-Net stage. Default: [32, 64, 128, 256, 512]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'leaky_relu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'gn'``
        init_mode (str): one of ``'xavier'``, ``'kaiming'``, ``'selu'`` or ``'orthogonal'``. Default: ``'orthogonal'``
        pooling (bool): downsample by max-pooling if `True` else using stride. Default: `False`
    """
    block_dict = {'residual': BasicBlock2d, 'residual_se': BasicBlock2dSE}

    def __init__(self, block_type='residual', in_channel: int=1, out_channel: int=3, filters: List[int]=[32, 64, 128, 256, 512], pad_mode: str='replicate', act_mode: str='leaky_relu', norm_mode: str='gn', init_mode: str='orthogonal', pooling: bool=False, **kwargs):
        super().__init__()
        self.depth = len(filters)
        self.pooling = pooling
        block = self.block_dict[block_type]
        self.shared_kwargs = {'pad_mode': pad_mode, 'act_mode': act_mode, 'norm_mode': norm_mode}
        self.conv_in = conv2d_norm_act(in_channel, filters[0], 5, padding=2, **self.shared_kwargs)
        self.conv_out = conv2d_norm_act(filters[0], out_channel, 5, padding=2, bias=True, pad_mode=pad_mode, act_mode='none', norm_mode='none')
        self.down_layers = nn.ModuleList()
        for i in range(self.depth):
            kernel_size, padding = 3, 1
            previous = max(0, i - 1)
            stride = self._get_stride(previous, i)
            layer = nn.Sequential(self._make_pooling_layer(previous, i), conv2d_norm_act(filters[previous], filters[i], kernel_size, stride=stride, padding=padding, **self.shared_kwargs), block(filters[i], filters[i], **self.shared_kwargs))
            self.down_layers.append(layer)
        self.up_layers = nn.ModuleList()
        for j in range(1, self.depth):
            kernel_size, padding = 3, 1
            layer = nn.ModuleList([conv2d_norm_act(filters[j], filters[j - 1], kernel_size, padding=padding, **self.shared_kwargs), block(filters[j - 1], filters[j - 1], **self.shared_kwargs)])
            self.up_layers.append(layer)
        model_init(self, mode=init_mode)

    def forward(self, x):
        x = self.conv_in(x)
        down_x = [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            x = self.down_layers[i](x)
            down_x[i] = x
        x = self.down_layers[-1](x)
        for j in range(self.depth - 1):
            i = self.depth - 2 - j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self.up_layers[i][1](x)
        x = self.conv_out(x)
        return x

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        When pooling layer is used, the input size is assumed to be even,
        therefore :attr:`align_corners` is set to `False` to avoid feature
        mis-match. When downsampling by stride, the input size is assumed
        to be 2n+1, and :attr:`align_corners` is set to `False`.
        """
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=align_corners)
        return x + y

    def _get_stride(self, previous, i):
        if self.pooling or previous == i:
            return 1
        return 2

    def _make_pooling_layer(self, previous, i):
        if self.pooling and previous != i:
            kernel_size = stride = 2
            return nn.MaxPool2d(kernel_size, stride)
        return nn.Identity()


class NonLocalBlock2D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, norm_layer=True, norm_mode='bn'):
        super(NonLocalBlock2D, self).__init__(in_channels, inter_channels, dimension=2, sub_sample=sub_sample, norm_layer=norm_layer, norm_mode=norm_mode)


class UNetPlus2D(UNet2D):

    def __init__(self, filters: List[int]=[32, 64, 128, 256, 512], norm_mode: str='gn', **kwargs):
        super().__init__(filters=filters, norm_mode=norm_mode, **kwargs)
        self.feat_layers = nn.ModuleList([conv2d_norm_act(filters[-1], filters[k - 1], 1, **self.shared_kwargs) for k in range(1, self.depth)])
        self.non_local = NonLocalBlock2D(filters[-1], sub_sample=False, norm_mode=norm_mode)

    def forward(self, x):
        x = self.conv_in(x)
        down_x = [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            x = self.down_layers[i](x)
            down_x[i] = x
        x = self.down_layers[-1](x)
        x = self.non_local(x)
        feat = x
        for j in range(self.depth - 1):
            i = self.depth - 2 - j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self._upsample_add(self.feat_layers[i](feat), x)
            x = self.up_layers[i][1](x)
        x = self.conv_out(x)
        return x


class UnetrPrUpBlock(nn.Module):
    """
    A projection upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, num_layer: int, kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], upsample_kernel_size: Union[Sequence[int], int], norm_name: Union[Tuple, str], conv_block: bool=False, res_block: bool=False) ->None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        """
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_stride, conv_only=True, is_transposed=True)
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList([nn.Sequential(get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_stride, conv_only=True, is_transposed=True), UnetResBlock(spatial_dims=spatial_dims, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, norm_name=norm_name)) for i in range(num_layer)])
            else:
                self.blocks = nn.ModuleList([nn.Sequential(get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_stride, conv_only=True, is_transposed=True), UnetBasicBlock(spatial_dims=spatial_dims, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, norm_name=norm_name)) for i in range(num_layer)])
        else:
            self.blocks = nn.ModuleList([get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_stride, conv_only=True, is_transposed=True) for i in range(num_layer)])

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(self, in_channels: int, img_size: Union[Sequence[int], int], patch_size: Union[Sequence[int], int], hidden_size: int=768, mlp_dim: int=3072, num_layers: int=12, num_heads: int=12, pos_embed: str='conv', classification: bool=False, num_classes: int=2, dropout_rate: float=0.0, spatial_dims: int=3, post_activation='Tanh') ->None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """
        super().__init__()
        if not 0 <= dropout_rate <= 1:
            raise ValueError('dropout_rate should be between 0 and 1.')
        if hidden_size % num_heads != 0:
            raise ValueError('hidden_size should be divisible by num_heads.')
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(in_channels=in_channels, img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, num_heads=num_heads, pos_embed=pos_embed, dropout_rate=dropout_rate, spatial_dims=spatial_dims)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == 'Tanh':
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, 'cls_token'):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, 'classification_head'):
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out


class UNETR(nn.Module):
    """
    Based on: "Hatamizadeh et al.,<https://arxiv.org/abs/2103.10504>". 

    Args:

    in_channels (int): dimension of input channels.
    out_channels (int): dimension of output channels.
    img_size (Tuple[int, int, int]): dimension of input image.
    feature_size (int): dimension of network feature size.
    hidden_size (int): dimension of hidden layer.
    mlp_dim (int): dimension of feedforward layer.
    num_heads (int): number of attention heads.
    pos_embed (str): position embedding layer type.
    norm_name (Union[Tuple, str]): feature normalization type and arguments.
    conv_block (bool): bool argument to determine if convolutional block is used.
    res_block (bool): bool argument to determine if residual block is used.
    dropout_rate (float): faction of the input units to drop.

    """

    def __init__(self, in_channel: int, out_channel: int, img_size: Tuple[int, int, int], feature_size: int=16, hidden_size: int=768, mlp_dim: int=3072, num_heads: int=12, pos_embed: str='perceptron', norm_name: Union[Tuple, str]='instance', conv_block: bool=False, res_block: bool=True, dropout_rate: float=0.0, **kwargs) ->None:
        super().__init__()
        if not 0 <= dropout_rate <= 1:
            raise AssertionError('dropout_rate should be between 0 and 1.')
        if hidden_size % num_heads != 0:
            raise AssertionError('hidden size should be divisible by num_heads.')
        if pos_embed not in ['conv', 'perceptron']:
            raise KeyError(f'Position embedding layer of type {pos_embed} is not supported.')
        self.num_layers = 12
        self.patch_size = 16, 16, 16
        self.feat_size = img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1], img_size[2] // self.patch_size[2]
        in_channels = in_channel
        out_channels = out_channel
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(in_channels=in_channels, img_size=img_size, patch_size=self.patch_size, hidden_size=hidden_size, mlp_dim=mlp_dim, num_layers=self.num_layers, num_heads=num_heads, pos_embed=pos_embed, classification=self.classification, dropout_rate=dropout_rate)
        self.encoder1 = UnetrBasicBlock(spatial_dims=3, in_channels=in_channels, out_channels=feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.encoder2 = UnetrPrUpBlock(spatial_dims=3, in_channels=hidden_size, out_channels=feature_size * 2, num_layer=2, kernel_size=3, stride=1, upsample_kernel_size=2, norm_name=norm_name, conv_block=conv_block, res_block=res_block)
        self.encoder3 = UnetrPrUpBlock(spatial_dims=3, in_channels=hidden_size, out_channels=feature_size * 4, num_layer=1, kernel_size=3, stride=1, upsample_kernel_size=2, norm_name=norm_name, conv_block=conv_block, res_block=res_block)
        self.encoder4 = UnetrPrUpBlock(spatial_dims=3, in_channels=hidden_size, out_channels=feature_size * 8, num_layer=0, kernel_size=3, stride=1, upsample_kernel_size=2, norm_name=norm_name, conv_block=conv_block, res_block=res_block)
        self.decoder5 = UnetrUpBlock(spatial_dims=3, in_channels=hidden_size, out_channels=feature_size * 8, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder4 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder3 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder2 = UnetrUpBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            for i in weights['state_dict']:
                None
            self.vit.patch_embedding.position_embeddings.copy_(weights['state_dict']['module.transformer.patch_embedding.position_embeddings_3d'])
            self.vit.patch_embedding.cls_token.copy_(weights['state_dict']['module.transformer.patch_embedding.cls_token'])
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.weight'])
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.bias'])
            for bname, block in self.vit.blocks.named_children():
                None
                block.loadFrom(weights, n_block=bname)
            self.vit.norm.weight.copy_(weights['state_dict']['module.transformer.norm.weight'])
            self.vit.norm.bias.copy_(weights['state_dict']['module.transformer.norm.bias'])

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits


def conv_bn_2d(in_planes, planes, kernel_size, stride, padding, groups=1, dilation=1, pad_mode='zeros'):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_planes, planes, kernel_size, stride=stride, padding=padding, padding_mode=pad_mode, groups=groups, dilation=dilation, bias=False))
    result.add_module('bn', nn.BatchNorm2d(planes))
    return result


class RepVGGBlock2D(nn.Module):
    """ 2D RepVGG Block, adapted from:
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, pad_mode='zeros', act_mode='relu', deploy=False):
        super(RepVGGBlock2D, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_planes = in_planes
        self.act = get_activation(act_mode)
        padding = dilation
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_planes, planes, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, padding_mode=pad_mode, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_planes) if planes == in_planes and stride == 1 else None
            self.rbr_dense = conv_bn_2d(in_planes, planes, kernel_size, stride, padding, groups=groups, dilation=dilation, pad_mode=pad_mode)
            self.rbr_1x1 = conv_bn_2d(in_planes, planes, 1, stride, padding=0, groups=groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(x))
        identity = 0
        if self.rbr_identity is not None:
            identity = self.rbr_identity(x)
        x = self.rbr_dense(x) + self.rbr_1x1(x) + identity
        return self.act(x)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernel_id, bias3x3 + bias1x1 + bias_id

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_planes // self.groups
                kernel_value = torch.zeros((self.in_planes, input_dim, 3, 3), dtype=torch.float, device=branch.weight.device)
                for i in range(self.in_planes):
                    kernel_value[i, i % input_dim, 1, 1] = 1.0
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias

    def load_reparam_kernel(self, kernel, bias):
        assert hasattr(self, 'rbr_reparam')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1, pad_mode: str='replicate') ->nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode=pad_mode)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[nn.Module]=None, groups: int=1, base_width: int=64, dilation: int=1, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn') ->None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride, pad_mode=pad_mode)
        self.bn1 = get_norm_2d(norm_mode, planes)
        self.relu = get_activation(act_mode)
        self.conv2 = conv3x3(planes, planes, pad_mode=pad_mode)
        self.bn2 = get_norm_2d(norm_mode, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) ->Tensor:
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


def conv1x1(in_planes: int, out_planes: int, stride: int=1) ->nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[nn.Module]=None, groups: int=1, base_width: int=64, dilation: int=1, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn') ->None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = get_norm_2d(norm_mode, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, pad_mode=pad_mode)
        self.bn2 = get_norm_2d(norm_mode, width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = get_norm_2d(norm_mode, planes * self.expansion)
        self.relu = get_activation(act_mode)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet2D(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int=10, zero_init_residual: bool=False, groups: int=1, width_per_group: int=64, replace_stride_with_dilation: Optional[List[bool]]=None, in_channel: int=3, pad_mode: str='replicate', act_mode: str='elu', norm_mode: str='bn', **_) ->None:
        super(ResNet2D, self).__init__()
        self.shared_kwargs = {'pad_mode': pad_mode, 'act_mode': act_mode, 'norm_mode': norm_mode}
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, padding_mode=pad_mode, bias=False)
        self.bn1 = get_norm_2d(norm_mode, self.inplanes)
        self.relu = get_activation(act_mode)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int=1, dilate: bool=False) ->nn.Sequential:
        norm_mode = self.shared_kwargs['norm_mode']
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), get_norm_2d(norm_mode, planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, **self.shared_kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, **self.shared_kwargs))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) ->Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) ->Tensor:
        return self._forward_impl(x)


filt_dict = {(1): np.array([1.0]), (2): np.array([1.0, 1.0]), (3): np.array([1.0, 2.0, 1.0]), (4): np.array([1.0, 3.0, 3.0, 1.0]), (5): np.array([1.0, 4.0, 6.0, 4.0, 1.0]), (6): np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0]), (7): np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])}


class ZeroPad1d(torch.nn.modules.padding.ConstantPad1d):

    def __init__(self, padding):
        super(ZeroPad1d, self).__init__(padding, 0.0)


def get_pad_layer_1d(pad_type):
    PadLayer = None
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad1d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad1d
    elif pad_type == 'zero':
        PadLayer = ZeroPad1d
    else:
        None
    return PadLayer


class BlurPool1D(nn.Module):

    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad_sizes = [(pad_size + pad_off) for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        a = filt_dict[self.filt_size]
        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))
        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_2d(pad_type):
    PadLayer = None
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        None
    return PadLayer


class BlurPool2D(nn.Module):

    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool2D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2)), int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad_sizes = [(pad_size + pad_off) for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        a = filt_dict[self.filt_size]
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer_2d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class ZeroPad3d(torch.nn.modules.padding.ConstantPad3d):

    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding, 0.0)


def get_pad_layer_3d(pad_type):
    PadLayer = None
    if pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad3d
    elif pad_type == 'zero':
        PadLayer = ZeroPad3d
    else:
        None
    return PadLayer


class BlurPool3D(nn.Module):

    def __init__(self, channels: int, pad_type: str='zero', filt_size: Union[int, List[int]]=3, stride: Union[int, List[int]]=2, pad_off=0):
        super(BlurPool3D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.stride = stride
        self.channels = channels
        if isinstance(self.filt_size, int):
            a = filt_dict[self.filt_size]
            filt = torch.Tensor(a[:, None, None] * a[None, :, None] * a[None, None, :])
            self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))] * 3
        else:
            assert len(self.filt_size) == 3
            z = filt_dict[self.filt_size[0]]
            y = filt_dict[self.filt_size[1]]
            x = filt_dict[self.filt_size[2]]
            filt = torch.Tensor(z[:, None, None] * y[None, :, None] * x[None, None, :])
            self.pad_sizes = []
            for i in range(3):
                self.pad_sizes += [int(1.0 * (filt_size[i] - 1) / 2), int(np.ceil(1.0 * (filt_size[i] - 1) / 2))]
        filt = filt / torch.sum(filt)
        self.pad_sizes = [(pad_size + pad_off) for pad_size in self.pad_sizes]
        self.register_buffer('filt', filt[None, None, :, :, :].repeat((channels, 1, 1, 1, 1)))
        self.pad = get_pad_layer_3d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off != 0:
                inp = self.pad(inp)
            return inp[:, :, ::self.stride[0], ::self.stride[1], ::self.stride[2]]
        return F.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class NonLocalBlock1D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, norm_layer=True, norm_mode='bn'):
        super(NonLocalBlock1D, self).__init__(in_channels, inter_channels, dimension=1, sub_sample=sub_sample, norm_layer=norm_layer, norm_mode=norm_mode)


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], upsample_kernel_size: Union[Sequence[int], int], norm_name: Union[Tuple, str], act_name: Union[Tuple, str]=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), dropout: Optional[Union[Tuple, str, float]]=None, trans_bias: bool=False):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_stride, dropout=dropout, bias=trans_bias, act=None, norm=None, conv_only=False, is_transposed=True)
        self.conv_block = UnetBasicBlock(spatial_dims, out_channels + out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, norm_name=norm_name, act_name=act_name)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class DiceLoss(nn.Module):
    """DICE loss.
    """

    def __init__(self, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        loss = 0.0
        for index in range(pred.size()[0]):
            iflat = pred[index].contiguous().view(-1)
            tflat = target[index].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            if self.power == 1:
                loss += 1 - (2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)
            else:
                loss += 1 - (2.0 * intersection + self.smooth) / ((iflat ** self.power).sum() + (tflat ** self.power).sum() + self.smooth)
        return loss / float(pred.size()[0])

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        if self.power == 1:
            loss = 1 - (2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)
        else:
            loss = 1 - (2.0 * intersection + self.smooth) / ((iflat ** self.power).sum() + (tflat ** self.power).sum() + self.smooth)
        return loss

    def forward(self, pred, target, weight_mask=None):
        if not target.size() == pred.size():
            raise ValueError('Target size ({}) must be the same as pred size ({})'.format(target.size(), pred.size()))
        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:
            loss = self.dice_loss_batch(pred, target)
        return loss


class WeightedMSE(nn.Module):
    """Weighted mean-squared error.
    """

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, pred, target, weight=None):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = s1 * s2
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, pred, target, weight_mask=None):
        return self.weighted_mse_loss(pred, target, weight_mask)


class WeightedMAE(nn.Module):
    """Mask weighted mean absolute error (MAE) energy function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=None):
        loss = F.l1_loss(pred, target, reduction='none')
        loss = loss * weight_mask
        return loss.mean()


class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """

    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, pred, target, weight_mask=None):
        return F.binary_cross_entropy(pred, target, weight_mask)


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted binary cross-entropy with logits.
    """

    def __init__(self, size_average=True, reduce=True, eps=0.0):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.eps = eps

    def forward(self, pred, target, weight_mask=None):
        return F.binary_cross_entropy_with_logits(pred, target.clamp(self.eps, 1 - self.eps), weight_mask)


class WeightedCE(nn.Module):
    """Mask weighted multi-class cross-entropy (CE) loss.
    """

    def __init__(self, class_weight: Optional[List[float]]=None):
        super().__init__()
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight)

    def forward(self, pred, target, weight_mask=None):
        if self.class_weight is not None:
            self.class_weight = self.class_weight
        loss = F.cross_entropy(pred, target, weight=self.class_weight, reduction='none')
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()


class WeightedLS(nn.Module):
    """Weighted CE loss with label smoothing (LS). The code is based on:
    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    """
    dim = 1

    def __init__(self, classes=10, cls_weights=None, smoothing=0.2):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.weights = 1.0
        if cls_weights is not None:
            self.weights = torch.tensor(cls_weights)

    def forward(self, pred, target, weight_mask=None):
        shape = (1, -1, 1, 1, 1) if pred.ndim == 5 else (1, -1, 1, 1)
        if isinstance(self.weights, torch.Tensor) and self.weights.ndim == 1:
            self.weights = self.weights.view(shape)
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * pred * self.weights, dim=self.dim)
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()


class WeightedBCEFocalLoss(nn.Module):
    """Weighted binary focal loss with logits.
    """

    def __init__(self, gamma=2.0, alpha=0.25, eps=0.0):
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, weight_mask=None):
        pred_sig = pred.sigmoid()
        pt = (1 - target) * (1 - pred_sig) + target * pred_sig
        at = (1 - self.alpha) * target + self.alpha * (1 - target)
        wt = at * (1 - pt) ** self.gamma
        if weight_mask is not None:
            wt *= weight_mask
        bce = F.binary_cross_entropy_with_logits(pred, target.clamp(self.eps, 1 - self.eps), reduction='none')
        return (wt * bce).mean()


class WSDiceLoss(nn.Module):

    def __init__(self, smooth=100.0, power=2.0, v2=0.85, v1=0.15):
        super().__init__()
        self.smooth = smooth
        self.power = power
        self.v2 = v2
        self.v1 = v1

    def dice_loss(self, pred, target):
        iflat = pred.reshape(pred.shape[0], -1)
        tflat = target.reshape(pred.shape[0], -1)
        wt = tflat * (self.v2 - self.v1) + self.v1
        g_pred = wt * (2 * iflat - 1)
        g = wt * (2 * tflat - 1)
        intersection = (g_pred * g).sum(-1)
        loss = 1 - (2.0 * intersection + self.smooth) / ((g_pred ** self.power).sum(-1) + (g ** self.power).sum(-1) + self.smooth)
        return loss.mean()

    def forward(self, pred, target, weight_mask=None):
        loss = self.dice_loss(pred, target)
        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives (vanilla, lsgan, and wgangp).
    Based on Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode: str='lsgan', target_real_label: float=1.0, target_fake_label: float=0.0):
        """ Initialize the GANLoss class.

        Args:
            gan_mode (str): the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool): label for a real image
            target_fake_label (bool): label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool):
        """Create label tensors with the same size as the input.

        Args:
            prediction (torch.Tensor): tpyically the prediction from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool):
        """Calculate loss given Discriminator's output and grount truth labels.

        Args:
            prediction (torch.Tensor): tpyically the prediction output from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.

    Args:
        pred (torch.Tensor): foreground logits.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None
    """

    def forward(self, pred: torch.Tensor, mask: Optional[torch.Tensor]=None):
        pred = torch.sigmoid(pred)
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=0.01)
        loss = 1.0 / diff
        if mask is not None:
            loss *= mask
        return loss.mean()


class ForegroundDTConsistency(nn.Module):
    """Consistency regularization between the binary foreground mask and
    signed distance transform.

    Args:
        pred1 (torch.Tensor): foreground logits.
        pred2 (torch.Tensor): signed distance transform.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None
    """

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor, mask: Optional[torch.Tensor]=None):
        log_prob_pos = F.logsigmoid(pred1)
        log_prob_neg = F.logsigmoid(-pred1)
        distance = torch.tanh(pred2)
        dist_pos = torch.clamp(distance, min=0.0)
        dist_neg = -torch.clamp(distance, max=0.0)
        loss_pos = -log_prob_pos * dist_pos
        loss_neg = -log_prob_neg * dist_neg
        loss = loss_pos + loss_neg
        if mask is not None:
            loss *= mask
        return loss.mean()


class ContourDTConsistency(nn.Module):
    """Consistency regularization between the instance contour map and
    signed distance transform.

    Args:
        pred1 (torch.Tensor): contour logits.
        pred2 (torch.Tensor): signed distance transform.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None.
    """

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor, mask: Optional[torch.Tensor]=None):
        contour_prob = torch.sigmoid(pred1)
        distance_abs = torch.abs(torch.tanh(pred2))
        assert contour_prob.shape == distance_abs.shape
        loss = contour_prob * distance_abs
        loss = loss ** 2
        if mask is not None:
            loss *= mask
        return loss.mean()


class FgContourConsistency(nn.Module):
    """Consistency regularization between the binary foreground map and 
    instance contour map.

    Args:
        pred1 (torch.Tensor): foreground logits.
        pred2 (torch.Tensor): contour logits.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None.
    """
    sobel = torch.tensor([1, 0, -1], dtype=torch.float32)
    eps = 1e-07

    def __init__(self, tsz_h=1) ->None:
        super().__init__()
        self.sz = 2 * tsz_h + 1
        self.sobel_x = self.sobel.view(1, 1, 1, 1, 3)
        self.sobel_y = self.sobel.view(1, 1, 1, 3, 1)

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor, mask: Optional[torch.Tensor]=None):
        fg_prob = torch.sigmoid(pred1)
        contour_prob = torch.sigmoid(pred2)
        self.sobel_x = self.sobel_x
        self.sobel_y = self.sobel_y
        edge_x = F.conv3d(fg_prob, self.sobel_x, padding=(0, 0, 1))
        edge_y = F.conv3d(fg_prob, self.sobel_y, padding=(0, 1, 0))
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + self.eps)
        edge = torch.clamp(edge, min=self.eps, max=1.0 - self.eps)
        edge = F.pad(edge, (1, 1, 1, 1, 0, 0))
        edge = F.max_pool3d(edge, kernel_size=(1, self.sz, self.sz), stride=1)
        assert edge.shape == contour_prob.shape
        loss = F.mse_loss(edge, contour_prob, reduction='none')
        if mask is not None:
            loss *= mask
        return loss.mean()


class NonoverlapReg(nn.Module):
    """Regularization to prevent overlapping prediction of pre- and post-synaptic
    masks in synaptic polarity prediction ("1" in MODEL.TARGET_OPT).

    Args:
        fg_masked (bool): mask the regularization region with predicted cleft. Defaults: True
    """

    def __init__(self, fg_masked: bool=True) ->None:
        super().__init__()
        self.fg_masked = fg_masked

    def forward(self, pred: torch.Tensor):
        pos = torch.sigmoid(pred[:, 0])
        neg = torch.sigmoid(pred[:, 1])
        loss = pos * neg
        if self.fg_masked:
            loss = loss * torch.sigmoid(pred[:, 2].detach())
        return loss.mean()


class VAELoss(nn.Module):
    """
    Computes the VAE loss function.
    KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
    """

    def __init__(self, kld_weight=0.01):
        super().__init__()
        self.kld_weight = kld_weight

    def forward(self, recons, input, mu, log_var) ->dict:
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.kld_weight * kld_loss
        loss_vis = {'recon_loss': recons_loss.detach(), 'KLD_loss': self.kld_weight * kld_loss.detach()}
        return loss, loss_vis


class VAEBase(nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def encode(self, input: Tensor) ->List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) ->Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) ->Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) ->Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) ->Tensor:
        pass


class VAE(VAEBase):
    """Variational autoencoder with convolutional layers. The input images should be square.
    """

    def __init__(self, img_channels: int, latent_dim: int, hidden_dims: List=[32, 64, 128, 256, 512], width: int=64, act_mode: str='relu', norm_mode: str='bn', **kwargs) ->None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = copy.deepcopy(hidden_dims)
        in_channels = img_channels
        sq_sz = self.calc_sz(width)
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), get_norm_2d(norm_mode, h_dim), get_activation(act_mode)))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * sq_sz, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * sq_sz, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * sq_sz)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), get_norm_2d(norm_mode, hidden_dims[i + 1]), get_activation(act_mode)))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), get_norm_2d(norm_mode, hidden_dims[-1]), get_activation(act_mode), nn.Conv2d(hidden_dims[-1], out_channels=img_channels, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.sz, self.sz)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    def calc_sz(self, width):
        down_sample = 2 ** len(self.hidden_dims)
        assert width % down_sample == 0, 'The input width/height ' + f'{width} is not divisible by {2 ** len(self.hidden_dims)}!'
        self.sz = width // down_sample
        sq_sz = self.sz ** 2
        return sq_sz


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
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock2d,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock2dSE,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock3d,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BasicBlock3dPA,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BasicBlock3dPASE,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BasicBlock3dSE,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BinaryReg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BlurPool1D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (BlurPool2D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BlurPool3D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ContourDTConsistency,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeepLabHeadA,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeepLabHeadB,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeepLabHeadC,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 256, 64, 64])], {}),
     True),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Discriminator3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (EfficientNet3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (FCNHead,
     lambda: ([], {'in_channels': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FPN3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (ForegroundDTConsistency,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GANLoss,
     lambda: ([], {}),
     lambda: ([], {'prediction': torch.rand([4, 4]), 'target_is_real': 4}),
     True),
    (InvertedResidual,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'kernel_size': 3, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (InvertedResidualDilated,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'kernel_size': 3, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NonLocalBlock1D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (NonLocalBlock2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NonoverlapReg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PatchMerging,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 8, 8])], {}),
     False),
    (PatchMergingV2,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 8, 8])], {}),
     True),
    (RepVGG3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (RepVGGBlock2D,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RepVGGBlock3D,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ResNet3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (SELayer2d,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SELayer3d,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNet2D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     False),
    (UNet3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (UNetPlus2D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     False),
    (UNetPlus3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (VAEBase,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (VAELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (WSDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedBCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedBCEFocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedMSE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ZeroPad1d,
     lambda: ([], {'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ZeroPad3d,
     lambda: ([], {'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_zudi_lin_pytorch_connectomics(_paritybench_base):
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

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

    def test_048(self):
        self._check(*TESTCASES[48])

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

    def test_052(self):
        self._check(*TESTCASES[52])

