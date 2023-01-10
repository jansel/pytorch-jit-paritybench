import sys
_module = sys.modules[__name__]
del sys
build_dataset = _module
datasets = _module
datautils = _module
lmdbutils = _module
p1dataset = _module
p2dataset = _module
evaluator = _module
models = _module
aux_classifier = _module
comp_encoder = _module
content_encoder = _module
decoder = _module
discriminator = _module
generator = _module
memory = _module
modules = _module
blocks = _module
cbam = _module
frn = _module
globalcontext = _module
modules = _module
train = _module
trainer = _module
base_trainer = _module
combined_trainer = _module
factorize_trainer = _module
trainer_utils = _module
utils = _module
logger = _module
utils = _module
visualize = _module
writer = _module

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


import random


from torch.utils.data import DataLoader


from itertools import chain


import copy


import numpy as np


from torch.utils.data import Dataset


from functools import partial


import torch.nn as nn


import torch.nn.functional as F


import math


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.multiprocessing as mp


import torch.utils.data.distributed


from torchvision import transforms


from torch.nn.parallel import DistributedDataParallel as DDP


import re


from torchvision import utils as tv_utils


class AuxClassifier(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.feat_class = ()
        self.n_last = 0

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, self.feat_class):
                features.append(x)
        if self.n_last:
            features = features[-self.n_last:]
        return x, features


class ComponentConditionBlock(nn.Module):

    def __init__(self, in_shape, n_comps):
        super().__init__()
        self.in_shape = in_shape
        self.bias = nn.Parameter(torch.zeros(n_comps, in_shape[0], 1, 1), requires_grad=True)

    def forward(self, x, comp_id):
        b = self.bias[comp_id]
        out = x + b
        return out


class ComponentEncoder(nn.Module):

    def __init__(self, body, head, final_shape, skip_shape, sigmoid=False, skip_layer_idx=None):
        super().__init__()
        self.body = nn.ModuleList(body)
        self.head = nn.ModuleList(head)
        self.final_shape = final_shape
        self.skip_shape = skip_shape
        self.skip_layer_idx = skip_layer_idx
        self.sigmoid = sigmoid

    def forward(self, x, comp_id=None):
        x = x.repeat((1, 1, 1, 1))
        ret_feats = {}
        for layer in self.body:
            if isinstance(layer, ComponentConditionBlock):
                x = layer(x, comp_id)
            else:
                x = layer(x)
        for lidx, layer in enumerate(self.head):
            x = layer(x)
            if lidx == self.skip_layer_idx:
                ret_feats['skip'] = x
        ret_feats['last'] = x
        if self.sigmoid:
            ret_feats = {k: nn.Sigmoid()(v) for k, v in ret_feats.items()}
        return ret_feats


class ContentEncoder(nn.Module):

    def __init__(self, layers, sigmoid=False):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.sigmoid = sigmoid

    def forward(self, x):
        x = x.repeat((1, 1, 1, 1))
        out = self.net(x)
        if self.sigmoid:
            out = nn.Sigmoid()(out)
        return out


class FilterResponseNorm(nn.Module):
    """ Filter Response Normalization """

    def __init__(self, num_features, ndim, eps=None, learnable_eps=False):
        """
        Args:
            num_features
            ndim
            eps: if None is given, use the paper value as default.
                from paper, fixed_eps=1e-6 and learnable_eps_init=1e-4.
            learnable_eps: turn eps to learnable parameter, which is recommended on
                fully-connected or 1x1 activation map.
        """
        super().__init__()
        if eps is None:
            if learnable_eps:
                eps = 0.0001
            else:
                eps = 1e-06
        self.num_features = num_features
        self.init_eps = eps
        self.learnable_eps = learnable_eps
        self.ndim = ndim
        self.mean_dims = list(range(2, 2 + ndim))
        self.weight = nn.Parameter(torch.ones([1, num_features] + [1] * ndim))
        self.bias = nn.Parameter(torch.zeros([1, num_features] + [1] * ndim))
        if learnable_eps:
            self.eps = nn.Parameter(torch.as_tensor(eps))
        else:
            self.register_buffer('eps', torch.as_tensor(eps))

    def forward(self, x):
        nu2 = x.pow(2).mean(self.mean_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        x = x * self.weight + self.bias
        return x

    def extra_repr(self):
        return 'num_features={}, init_eps={}, ndim={}'.format(self.num_features, self.init_eps, self.ndim)


FilterResponseNorm2d = partial(FilterResponseNorm, ndim=2)


class TLU(nn.Module):
    """ Thresholded Linear Unit """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return torch.max(x, self.tau)

    def extra_repr(self):
        return 'num_features={}'.format(self.num_features)


def dispatcher(dispatch_fn):

    def decorated(key, *args):
        if callable(key):
            return key
        if key is None:
            key = 'none'
        return dispatch_fn(key, *args)
    return decorated


@dispatcher
def norm_dispatch(norm):
    return {'none': nn.Identity, 'in': partial(nn.InstanceNorm2d, affine=False), 'bn': nn.BatchNorm2d, 'frn': FilterResponseNorm2d}[norm.lower()]


@dispatcher
def activ_dispatch(activ, norm=None):
    if norm_dispatch(norm) == FilterResponseNorm2d:
        activ = 'tlu'
    return {'none': nn.Identity, 'relu': nn.ReLU, 'lrelu': partial(nn.LeakyReLU, negative_slope=0.2), 'tlu': TLU}[activ.lower()]


@dispatcher
def pad_dispatch(pad_type):
    return {'zero': nn.ZeroPad2d, 'replicate': nn.ReplicationPad2d, 'reflect': nn.ReflectionPad2d}[pad_type.lower()]


def spectral_norm(module):
    """ init & apply spectral norm """
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()
    return nn.utils.spectral_norm(module)


@dispatcher
def w_norm_dispatch(w_norm):
    return {'spectral': spectral_norm, 'none': lambda x: x}[w_norm.lower()]


class ConvBlock(nn.Module):
    """ pre-active conv block """

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, norm='none', activ='relu', bias=True, upsample=False, downsample=False, w_norm='none', pad_type='zero', dropout=0.0, size=None):
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        activ = activ_dispatch(activ, norm)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)
        self.upsample = upsample
        self.downsample = downsample
        assert (norm == FilterResponseNorm2d) == (activ == TLU), 'Use FRN and TLU together'
        if norm == FilterResponseNorm2d and size == 1:
            self.norm = norm(C_in, learnable_eps=True)
        else:
            self.norm = norm(C_in)
        if activ == TLU:
            self.activ = activ(C_in)
        else:
            self.activ = activ()
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.pad = pad(padding)
        self.conv = w_norm(nn.Conv2d(C_in, C_out, kernel_size, stride, bias=bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv(self.pad(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class Integrator(nn.Module):

    def __init__(self, C, norm='none', activ='none', C_in=None, C_content=0):
        super().__init__()
        C_in = (C_in or C) + C_content
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, comps, x=None, content=None):
        """
        Args:
            comps [B, 3, mem_shape]: component features
        """
        if content is not None:
            inputs = torch.cat([comps, content], dim=1)
        else:
            inputs = comps
        out = self.integrate_layer(inputs)
        if x is not None:
            out = torch.cat([x, out], dim=1)
        return out


class Decoder(nn.Module):

    def __init__(self, layers, skips=None, out='sigmoid'):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        if skips is not None:
            self.skip_idx, self.skip_layer = skips
        if out == 'sigmoid':
            self.out = nn.Sigmoid()
        elif out == 'tanh':
            self.out = nn.Tanh()
        else:
            raise ValueError(out)

    def forward(self, x, skip_feat=None, content_feats=None):
        for i, layer in enumerate(self.layers):
            if i == self.skip_idx:
                x = self.skip_layer(skip_feat, x=x)
            if i == 0:
                x = layer(x, content=content_feats)
            else:
                x = layer(x)
        return self.out(x)


class ProjectionDiscriminator(nn.Module):
    """ Multi-task discriminator """

    def __init__(self, C, n_fonts, n_chars, w_norm='spectral', activ='none'):
        super().__init__()
        self.activ = activ_dispatch(activ)()
        w_norm = w_norm_dispatch(w_norm)
        self.font_emb = w_norm(nn.Embedding(n_fonts, C))
        self.char_emb = w_norm(nn.Embedding(n_chars, C))

    def forward(self, x, font_indice, char_indice):
        x = self.activ(x)
        font_emb = self.font_emb(font_indice)
        char_emb = self.char_emb(char_indice)
        font_out = torch.einsum('bchw,bc->bhw', x.float(), font_emb.float()).unsqueeze(1)
        char_out = torch.einsum('bchw,bc->bhw', x.float(), char_emb.float()).unsqueeze(1)
        return [font_out, char_out]

    def extend_font(self, font_idx):
        """ extend font by cloning font index """
        nn.utils.remove_spectral_norm(self.font_emb)
        self.font_emb.weight.data = torch.cat([self.font_emb.weight, self.font_emb.weight[font_idx].unsqueeze(0)])
        self.font_emb.num_embeddings += 1
        self.font_emb = nn.utils.spectral_norm(self.font_emb)

    def extend_chars(self, n_chars):
        nn.utils.remove_spectral_norm(self.char_emb)
        mean_emb = self.char_emb.weight.mean(0, keepdim=True).repeat(n_chars, 1)
        self.char_emb.weight.data = torch.cat([self.char_emb.weight, mean_emb])
        self.char_emb.num_embeddings += n_chars
        self.char_emb = nn.utils.spectral_norm(self.char_emb)


class CustomDiscriminator(nn.Module):
    """
    spectral norm + ResBlock + Multi-task Discriminator (No patchGAN)
    """

    def __init__(self, feats, gap, projD):
        super().__init__()
        self.feats = feats
        self.gap = gap
        self.projD = projD

    def forward(self, x, font_indice, char_indice, out_feats='none'):
        assert out_feats in {'none', 'all'}
        feats = []
        for layer in self.feats:
            x = layer(x)
            feats.append(x)
        x = self.gap(x)
        ret = self.projD(x, font_indice, char_indice)
        if out_feats == 'all':
            ret += feats
        ret = tuple(map(lambda i: i, ret))
        return ret


def reduce_features(feats, reduction='mean'):
    if reduction == 'mean':
        return torch.stack(feats).mean(dim=0)
    elif reduction == 'first':
        return feats[0]
    elif reduction == 'none':
        return feats
    elif reduction == 'sign':
        return (torch.stack(feats).mean(dim=0) > 0.5).float()
    else:
        raise ValueError(reduction)


class CombMemory:

    def __init__(self):
        self.memory = {}
        self.reset()

    def write(self, style_ids, comp_ids, sc_feats):
        assert len(style_ids) == len(comp_ids) == len(sc_feats), 'Input sizes are different'
        for style_id, comp_id, sc_feat in zip(style_ids, comp_ids, sc_feats):
            self.write_point(style_id, comp_id, sc_feat)

    def write_point(self, style_id, comp_id, sc_feat):
        sc_feat = sc_feat.squeeze()
        self.memory.setdefault(style_id.item(), {}).setdefault(comp_id.item(), []).append(sc_feat)

    def read_point(self, style_id, comp_id, reduction='mean'):
        style_id = int(style_id)
        comp_id = int(comp_id)
        sc_feats = self.memory[style_id][comp_id]
        return reduce_features(sc_feats, reduction)

    def read_char(self, style_id, comp_ids, reduction='mean'):
        char_feats = []
        for comp_id in comp_ids:
            comp_feat = self.read_point(style_id, comp_id, reduction)
            char_feats.append(comp_feat)
        char_feats = torch.stack(char_feats)
        return char_feats

    def reset(self):
        self.memory = {}


class SingleMemory:

    def __init__(self):
        self.memory = {}
        self.reset()

    def write(self, ids, feats):
        assert len(ids) == len(feats), 'Input sizes are different'
        for id_, feat in zip(ids, feats):
            self.write_point(id_, feat)

    def write_point(self, id_, feat):
        feat = feat.squeeze()
        self.memory.setdefault(int(id_), []).append(feat)

    def read_point(self, id_, reduction='mean'):
        feats = self.memory[int(id_)]
        return reduce_features(feats, reduction)

    def get_var(self, id_):
        feats = torch.stack(self.memory[int(id_)])
        mean_feats = torch.stack([feats.mean(0)] * len(feats))
        var = F.mse_loss(feats, mean_feats)
        return var

    def get_all_var(self):
        var = sum([self.get_var(id_) for id_ in self.memory.keys()])
        var = var / len(self.memory)
        return var

    def read(self, ids, reduction='mean'):
        feats = []
        for id_ in ids:
            id_ = int(id_)
            feats.append(self.read_point(id_, reduction))
        return torch.stack(feats)

    def reset(self):
        self.memory = {}


class FactMemory:

    def __init__(self):
        self.style = SingleMemory()
        self.comp = SingleMemory()

    def write_style_point(self, style_id, style_feat):
        self.style.write_point(style_id, style_feat)

    def write_styles(self, ids, feats):
        self.style.write(ids, feats)

    def write_comp_point(self, comp_id, comp_feat):
        self.comp.write_point(comp_id, comp_feat)

    def write_comps(self, ids, feats):
        self.comp.write(ids, feats)

    def read_char(self, style_id, comp_ids, reduction='mean'):
        style_feat = self.style.read_point(style_id, reduction)
        comp_feat = self.comp.read(comp_ids, reduction)
        char_feat = (style_feat * comp_feat).sum(1)
        return char_feat

    def read_point(self, style_id, comp_id, reduction='mean'):
        style_feat = self.style.read_point(style_id, reduction)
        comp_feat = self.comp.read_point(comp_id, reduction)
        feat = (style_feat * comp_feat).sum(0)
        return feat

    def get_all_var(self):
        style_vars = self.style.get_all_var()
        comp_vars = self.comp.get_all_var()
        return style_vars + comp_vars

    def reset(self):
        self.style.reset()
        self.comp.reset()


class Memory(nn.Module):
    STYLE_id = -1

    def __init__(self):
        super().__init__()
        self.comb_memory = CombMemory()
        self.fact_memory = FactMemory()

    def write_fact(self, style_ids, comp_ids, style_feats, comp_feats):
        self.fact_memory.write_styles(style_ids, style_feats)
        self.fact_memory.write_comps(comp_ids, comp_feats)

    def write_point_fact(self, style_id, comp_id, style_feat, comp_feat):
        self.fact_memory.write_style_point(style_id, style_feat)
        self.fact_memory.write_comp_point(comp_id, comp_feat)

    def write_comb(self, style_ids, comp_ids, sc_feats):
        self.comb_memory.write(style_ids, comp_ids, sc_feats)

    def write_point_comb(self, style_id, comp_id, sc_feat):
        self.comb_memory.write_point(style_id, comp_id, sc_feat)

    def read_char_both(self, style_id, comp_id_char, reduction='mean'):
        sc_feat = []
        for comp_id in comp_id_char:
            saved_comp_ids = self.comb_memory.memory.get(style_id, [])
            if comp_id in saved_comp_ids:
                feat = self.comb_memory.read_point(style_id, comp_id, reduction)
            else:
                feat = self.fact_memory.read_point(style_id, comp_id, reduction)
            sc_feat.append(feat)
        sc_feat = torch.stack(sc_feat)
        return sc_feat

    def read_chars(self, style_ids, comp_ids, reduction='mean', type='both'):
        sc_feats = []
        read_funcs = {'both': self.read_char_both, 'comb': self.comb_memory.read_char, 'fact': self.fact_memory.read_char}
        read_char = read_funcs[type]
        for style_id, comp_id_char in zip(style_ids, comp_ids):
            sc_feat = read_char(style_id, comp_id_char, reduction)
            sc_feats.append(sc_feat)
        return sc_feats

    def read_style(self, ids, reduction='mean'):
        return self.fact_memory.style.read(ids, reduction)

    def read_comp(self, ids, reduction='mean'):
        return self.fact_memory.comp.read(ids, reduction)

    def read_comb(self, style_ids, comp_ids, reduction='mean'):
        sc_feats = []
        for style_id, comp_id_char in zip(style_ids, comp_ids):
            sc_feat = self.comb_memory.read_char(style_id, comp_id_char, reduction)
            sc_feats.append(sc_feat)
        return sc_feats

    def get_fact_var(self):
        return self.fact_memory.get_all_var()

    def reset_memory(self):
        self.comb_memory.reset()
        self.fact_memory.reset()


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class GlobalContext(nn.Module):
    """ Global-context """

    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        C_bottleneck = int(C * bottleneck_ratio)
        w_norm = w_norm_dispatch(w_norm)
        self.k_proj = w_norm(nn.Conv2d(C, 1, 1))
        self.transform = nn.Sequential(w_norm(nn.Linear(C, C_bottleneck)), nn.LayerNorm(C_bottleneck), nn.ReLU(), w_norm(nn.Linear(C_bottleneck, C)))

    def forward(self, x):
        context_logits = self.k_proj(x)
        context_weights = F.softmax(context_logits.flatten(1), dim=1)
        context = torch.einsum('bci,bi->bc', x.flatten(2), context_weights)
        out = self.transform(context)
        return out[..., None, None]


class GCBlock(nn.Module):
    """ Global-context block """

    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        self.gc = GlobalContext(C, bottleneck_ratio, w_norm)

    def forward(self, x):
        gc = self.gc(x)
        return x + gc


class ResBlock(nn.Module):
    """ Pre-activate ResBlock with spectral normalization """

    def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=False, downsample=False, norm='none', w_norm='none', activ='relu', pad_type='zero', dropout=0.0, scale_var=False):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var
        self.conv1 = ConvBlock(C_in, C_out, kernel_size, 1, padding, norm, activ, upsample=upsample, w_norm=w_norm, pad_type=pad_type, dropout=dropout)
        self.conv2 = ConvBlock(C_out, C_out, kernel_size, 1, padding, norm, activ, w_norm=w_norm, pad_type=pad_type, dropout=dropout)
        if C_in != C_out or upsample or downsample:
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))

    def forward(self, x):
        """
        normal: pre-activ + convs + skip-con
        upsample: pre-activ + upsample + convs + skip-con
        downsample: pre-activ + convs + downsample + skip-con
        => pre-activ + (upsample) + convs + (downsample) + skip-con
        """
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        if hasattr(self, 'skip'):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        out = out + x
        if self.scale_var:
            out = out / np.sqrt(2)
        return out


def comp_enc_builder(C_in, C, norm='none', activ='relu', pad_type='reflect', sigmoid=True, skip_scale_var=False, n_comps=None):
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type, scale_var=skip_scale_var)
    body = [ConvBlk(C_in, C, 3, 1, 1, norm='none', activ='none'), ConvBlk(C * 1, C * 2, 3, 1, 1, downsample=True), GCBlock(C * 2), ConvBlk(C * 2, C * 4, 3, 1, 1, downsample=True), CBAM(C * 4), ComponentConditionBlock((128, 32, 32), n_comps)]
    head = [ResBlk(C * 4, C * 4, 3, 1), CBAM(C * 4), ResBlk(C * 4, C * 4, 3, 1), ResBlk(C * 4, C * 8, 3, 1, downsample=True), CBAM(C * 8), ResBlk(C * 8, C * 8)]
    skip_layer_idx = 2
    final_shape = C * 8, 16, 16
    skip_shape = C * 4, 32, 32
    return ComponentEncoder(body, head, final_shape, skip_shape, sigmoid, skip_layer_idx)


def content_enc_builder(C_in, C, C_out, norm='none', activ='relu', content_sigmoid=False, pad_type='zero'):
    if not C_out:
        return None
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type)
    layers = [ConvBlk(C_in, C, 3, 1, 1, norm='none', activ='none'), ConvBlk(C * 1, C * 2, 3, 2, 1), ConvBlk(C * 2, C * 4, 3, 2, 1), ConvBlk(C * 4, C * 8, 3, 2, 1), ConvBlk(C * 8, C_out, 3, 1, 1)]
    return ContentEncoder(layers, content_sigmoid)


def dec_builder(C, C_out, norm='IN', activ='relu', pad_type='reflect', out='sigmoid', C_content=0):
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type)
    IntegrateBlk = partial(Integrator, norm='none', activ='none')
    layers = [IntegrateBlk(C * 8, C_content=C_content), ResBlk(C * 8, C * 8, 3, 1), ResBlk(C * 8, C * 8, 3, 1), ResBlk(C * 8, C * 8, 3, 1), ConvBlk(C * 8, C * 4, 3, 1, 1, upsample=True), ConvBlk(C * 8, C * 2, 3, 1, 1, upsample=True), ConvBlk(C * 2, C * 1, 3, 1, 1, upsample=True), ConvBlk(C * 1, C_out, 3, 1, 1)]
    skips = 5, IntegrateBlk(C * 4)
    return Decoder(layers, skips, out=out)


class ParamBlock(nn.Module):

    def __init__(self, C_out, shape):
        super().__init__()
        w = torch.randn((C_out, *shape))
        b = torch.randn((C_out,))
        self.shape = shape
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x):
        b = self.b.reshape((1, *self.b.shape, 1, 1, 1)).repeat(x.size(0), 1, *self.shape)
        return self.w * x + b


def decompose_block_builder(emb_dim, in_shape, num_blocks=2):
    blocks = [ParamBlock(emb_dim, (in_shape[0], 1, 1)) for _ in range(num_blocks)]
    return blocks


class Generator(nn.Module):

    def __init__(self, C_in, C, C_out, comp_enc, emb_block, dec, content_enc, n_comps):
        super().__init__()
        self.component_encoder = comp_enc_builder(C_in, C, **comp_enc, n_comps=n_comps)
        self.mem_shape = self.component_encoder.final_shape
        assert self.mem_shape[-1] == self.mem_shape[-2]
        self.memory = Memory()
        self.skip_shape = self.component_encoder.skip_shape
        self.skip_memory = Memory()
        if emb_block['emb_dim']:
            self.emb_style, self.emb_comp = decompose_block_builder(**emb_block, in_shape=self.mem_shape)
            self.skip_emb_style, self.skip_emb_comp = decompose_block_builder(**emb_block, in_shape=self.skip_shape)
        C_content = content_enc['C_out']
        self.content_encoder = content_enc_builder(C_in, C, **content_enc)
        self.decoder = dec_builder(C, C_out, **dec, C_content=C_content)

    def reset_memory(self):
        self.memory.reset_memory()
        self.skip_memory.reset_memory()

    def get_fact_memory_var(self):
        var = self.memory.get_fact_var() + self.skip_memory.get_fact_var()
        return var

    def encode_write_fact(self, style_ids, comp_ids, style_imgs, write_comb=False, reset_memory=True):
        if reset_memory:
            self.reset_memory()
        feats = self.component_encoder(style_imgs, comp_ids)
        feat_sc = feats['last']
        feat_style = self.emb_style(feat_sc.unsqueeze(1))
        feat_comp = self.emb_comp(feat_sc.unsqueeze(1))
        self.memory.write_fact(style_ids, comp_ids, feat_style, feat_comp)
        if write_comb:
            self.memory.write_comb(style_ids, comp_ids, feat_sc)
        skip_sc = feats['skip']
        skip_style = self.skip_emb_style(skip_sc.unsqueeze(1))
        skip_comp = self.skip_emb_comp(skip_sc.unsqueeze(1))
        self.skip_memory.write_fact(style_ids, comp_ids, skip_style, skip_comp)
        if write_comb:
            self.skip_memory.write_comb(style_ids, comp_ids, skip_sc)
        return feat_style, feat_comp

    def encode_write_comb(self, style_ids, comp_ids, style_imgs, reset_memory=True):
        if reset_memory:
            self.reset_memory()
        feats = self.component_encoder(style_imgs, comp_ids)
        feat_scs = feats['last']
        self.memory.write_comb(style_ids, comp_ids, feat_scs)
        skip_scs = feats['skip']
        self.skip_memory.write_comb(style_ids, comp_ids, skip_scs)
        return feat_scs

    def read_memory(self, target_style_ids, target_comp_ids, reset_memory=True, phase='comb', try_comb=False, reduction='mean'):
        if phase == 'fact' and try_comb:
            phase = 'both'
        feats = self.memory.read_chars(target_style_ids, target_comp_ids, reduction=reduction, type=phase)
        skips = self.skip_memory.read_chars(target_style_ids, target_comp_ids, reduction=reduction, type=phase)
        feats = torch.stack([x.mean(0) for x in feats])
        skips = torch.stack([x.mean(0) for x in skips])
        if reset_memory:
            self.reset_memory()
        return feats, skips

    def read_decode(self, target_style_ids, target_comp_ids, content_imgs, reset_memory=True, reduction='mean', phase='fact', try_comb=False):
        feat_scs, skip_scs = self.read_memory(target_style_ids, target_comp_ids, reset_memory, phase=phase, reduction=reduction, try_comb=try_comb)
        content_feats = self.content_encoder(content_imgs)
        out = self.decoder(feat_scs, skip_scs, content_feats=content_feats)
        if reset_memory:
            self.reset_memory()
        return out

    def infer(self, in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids, content_imgs, phase, reduction='mean', try_comb=False):
        in_style_ids = in_style_ids
        in_comp_ids = in_comp_ids
        in_imgs = in_imgs
        trg_style_ids = trg_style_ids
        content_imgs = content_imgs
        if phase == 'comb':
            self.encode_write_comb(in_style_ids, in_comp_ids, in_imgs)
        elif phase == 'fact':
            self.encode_write_fact(in_style_ids, in_comp_ids, in_imgs, write_comb=False)
        else:
            raise NotImplementedError
        out = self.read_decode(trg_style_ids, trg_comp_ids, content_imgs=content_imgs, reduction=reduction, phase=phase, try_comb=try_comb)
        return out


FilterResponseNorm1d = partial(FilterResponseNorm, ndim=1, learnable_eps=True)


class LinearBlock(nn.Module):
    """ pre-active linear block """

    def __init__(self, C_in, C_out, norm='none', activ='relu', bias=True, w_norm='none', dropout=0.0):
        super().__init__()
        activ = activ_dispatch(activ, norm)
        if norm.lower() == 'bn':
            norm = nn.BatchNorm1d
        elif norm.lower() == 'frn':
            norm = FilterResponseNorm1d
        elif norm.lower() == 'none':
            norm = nn.Identity
        else:
            raise ValueError(f'LinearBlock supports BN only (but {norm} is given)')
        w_norm = w_norm_dispatch(w_norm)
        self.norm = norm(C_in)
        self.activ = activ()
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.linear = w_norm(nn.Linear(C_in, C_out, bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.linear(x)


class Upsample1x1(nn.Module):
    """ Upsample 1x1 to 2x2 using Linear """

    def __init__(self, C_in, C_out, norm='none', activ='relu', w_norm='none'):
        assert norm != 'IN', 'Do not use instance norm for 1x1 spatial size'
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.proj = ConvBlock(C_in, C_out * 4, 1, 1, 0, size=1, norm=norm, activ=activ, w_norm=w_norm)

    def forward(self, x):
        x = self.proj(x)
        B, C = x.shape[:2]
        return x.view(B, C // 4, 2, 2)


class HourGlass(nn.Module):
    """ U-net like hourglass module """

    def __init__(self, C_in, C_max, size, n_downs, n_mids=1, block_type='conv', down_stride=True, norm='none', activ='relu', w_norm='none', pad_type='zero', use_up1x1=False, norm1x1=None):
        """
        Args:
            C_max: maximum C_out of left downsampling block's output
            down_stride: downsampling via stride. If False, downsampling via avg_pool2d.
            use_up1x1: use Upsample1x1 block
            norm1x1: norm for 1x1
        """
        if down_stride:
            assert block_type == 'conv'
        norm1x1 = norm1x1 or norm
        super().__init__()
        self.C_in = C_in
        self.use_up1x1 = use_up1x1
        self.size = size
        ConvBlk = partial(ConvBlock, norm=norm, activ=activ, w_norm=w_norm, pad_type=pad_type)
        if block_type == 'conv':
            Block = ConvBlk
        elif block_type == 'res':
            ResBlk = partial(ResBlock, norm=norm, activ=activ, w_norm=w_norm, pad_type=pad_type)
            Block = ResBlk
        else:
            raise ValueError(block_type)
        self.lefts = nn.ModuleList()
        c_in = C_in
        for i in range(n_downs):
            c_out = min(c_in * 2, C_max)
            if down_stride:
                self.lefts.append(ConvBlk(c_in, c_out, stride=2))
            else:
                self.lefts.append(Block(c_in, c_out, downsample=True))
            c_in = c_out
            size //= 2
        mid_norm = norm1x1 if size == 1 else norm
        self.mids = nn.Sequential(*[ConvBlk(c_in, c_out, kernel_size=1, padding=0, size=size, norm=mid_norm) for _ in range(n_mids)])
        self.rights = nn.ModuleList()
        for i, lb in enumerate(self.lefts[::-1]):
            c_out = lb.C_in
            c_in = lb.C_out
            channel_in = c_in * 2 if i else c_in
            if i == 0 and use_up1x1:
                block = Upsample1x1(channel_in, c_out, norm=norm1x1, activ=activ, w_norm=w_norm)
            else:
                block = Block(channel_in, c_out, upsample=True)
            self.rights.append(block)

    def forward(self, x):
        features = []
        for lb in self.lefts:
            x = lb(x)
            features.append(x)
        if self.use_up1x1:
            assert x.shape[-2:] == torch.Size((1, 1))
        for i, (rb, lf) in enumerate(zip(self.rights, features[::-1])):
            if i:
                x = torch.cat([x, lf], dim=1)
            x = rb(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AuxClassifier,
     lambda: ([], {'layers': [_mock_layer()]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CBAM,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelGate,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ComponentConditionBlock,
     lambda: ([], {'in_shape': [4, 4], 'n_comps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (ComponentEncoder,
     lambda: ([], {'body': [_mock_layer()], 'head': [_mock_layer()], 'final_shape': 4, 'skip_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FilterResponseNorm,
     lambda: ([], {'num_features': 4, 'ndim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GCBlock,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalContext,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Integrator,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearBlock,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpatialGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TLU,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_clovaai_lffont(_paritybench_base):
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

