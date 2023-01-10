import sys
_module = sys.modules[__name__]
del sys
conf = _module
infer = _module
ptlflow = _module
data = _module
datasets = _module
flow_transforms = _module
split_autoflow = _module
models = _module
base_model = _module
base_model = _module
craft = _module
corr = _module
craft = _module
extractor = _module
gma = _module
setrans = _module
setrans_ablation = _module
update = _module
utils = _module
csflow = _module
csflow = _module
dicl = _module
dicl = _module
loss_functions = _module
fastflownet = _module
fastflownet = _module
flowformer = _module
attention = _module
cnn = _module
convnext = _module
decoder = _module
encoder = _module
encoders = _module
flowformer = _module
gma = _module
gru = _module
mlpmixer = _module
twins = _module
utils = _module
flownet = _module
flownet2 = _module
flownet_base = _module
flownet_fusion = _module
flownetc = _module
flownetcs = _module
flownetcss = _module
flownets = _module
flownetsd = _module
losses = _module
submodules = _module
corr = _module
extractor = _module
gma = _module
gma_utils = _module
update = _module
utils = _module
gmflow = _module
backbone = _module
geometry = _module
gmflow = _module
matching = _module
position = _module
transformer = _module
trident_conv = _module
utils = _module
gmflownet = _module
corr = _module
extractor = _module
gma = _module
gmflownet = _module
loss = _module
swin_transformer = _module
update = _module
augmentor = _module
drop = _module
flow_viz = _module
frame_utils = _module
helpers = _module
utils = _module
weight_init = _module
hd3 = _module
decoder = _module
dla = _module
dla_up = _module
hd3 = _module
hd3_ops = _module
hd3losses = _module
vgg = _module
irr = _module
irr_modules = _module
irr_pwc = _module
losses = _module
pwc_modules = _module
pwcnet = _module
pwcnet_irr = _module
lcv = _module
corr_lcv = _module
extractor = _module
lcv_raft = _module
update = _module
utils = _module
liteflownet = _module
liteflownet = _module
liteflownet2 = _module
liteflownet3 = _module
warp = _module
maskflownet = _module
maskflownet = _module
pwcnet = _module
raft = _module
corr = _module
extractor = _module
raft = _module
update = _module
utils = _module
scopeflow = _module
irr_modules = _module
irr_pwc_v2 = _module
losses = _module
pwc_modules = _module
scv = _module
compute_sparse_correlation = _module
extractor = _module
knn = _module
scv = _module
update = _module
utils = _module
starflow = _module
irr_modules = _module
pwc_modules = _module
starflow = _module
vcn = _module
conv4d = _module
submodule = _module
vcn = _module
callbacks = _module
logger = _module
correlation = _module
dummy_datasets = _module
external = _module
flowpy = _module
raft = _module
selflow = _module
flow_metrics = _module
flow_utils = _module
flowpy_torch = _module
io_adapter = _module
timer = _module
utils = _module
speed_benchmark = _module
summary_metrics = _module
test = _module
test_datasets = _module
test_get_dataset = _module
test_checkpoints = _module
test_models = _module
test_flowpy = _module
test_correlation = _module
test_flow_utils = _module
test_flowpy_torch = _module
test_infer = _module
test_speed_benchmark = _module
test_summary_metrics = _module
test_test = _module
test_train = _module
test_validate = _module
train = _module
validate = _module

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


from typing import Tuple


from typing import Union


import numpy as np


import torch


import logging


from typing import Optional


from torch import hub


from typing import Callable


from typing import Sequence


from torch.utils.data import Dataset


from collections.abc import KeysView


import random


import torch.nn.functional as F


import torchvision.transforms as tt


import warnings


from abc import abstractmethod


from typing import Any


import torch.optim as optim


from torch.utils.data import DataLoader


import torch.nn as nn


from torch import nn


from torch import einsum


import math


import copy


from torch.nn import Parameter


from scipy import interpolate


from torchvision.ops import DeformConv2d


from torch.autograd import Variable


from functools import partial


from torch.nn import init


from torch.functional import norm


from torch.nn import functional as F


from torch.nn.modules.utils import _pair


import torch.utils.checkpoint as checkpoint


from torchvision.transforms import ColorJitter


from torch.nn.init import _calculate_fan_in_and_fan_out


import torch.nn.functional as tf


from torchvision import ops


from torch.nn.parameter import Parameter


from torch.nn import Module


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.utils import _quadruple


from torch.nn import Conv2d


import torch.utils.data


import time


from torchvision.utils import make_grid


import re


from typing import IO


from collections import namedtuple


import pandas as pd


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


class CorrBlock:

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        corr = CorrBlock.corr(fmap1, fmap2)
        corr = corr.float()
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            if min(corr.shape[2:4]) > 2 * radius + 1:
                corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class CorrBlockSingleScale(nn.Module):

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__()
        self.radius = radius
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        self.corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        corr = self.corr
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl
        corr = bilinear_sampler(corr, coords_lvl)
        out = corr.view(batch, h1, w1, -1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


def drop_path(x, drop_prob: float=0.0, training: bool=False, scale_by_keep: bool=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

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
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float=0.0, scale_by_keep: bool=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LearnedSoftAggregate(nn.Module):

    def __init__(self, num_feat, group_dim, keepdim=False):
        super(LearnedSoftAggregate, self).__init__()
        self.group_dim = group_dim
        self.num_feat = num_feat
        self.feat2score = nn.Linear(num_feat, 1)
        self.keepdim = keepdim

    def forward(self, x, score_basis=None):
        if score_basis is None:
            score_basis = x
        if self.num_feat == 1:
            mode_scores = self.feat2score(score_basis.unsqueeze(-1)).squeeze(-1)
        else:
            mode_scores = self.feat2score(score_basis)
        attn_probs = mode_scores.softmax(dim=self.group_dim)
        x_aggr = (x * attn_probs).sum(dim=self.group_dim, keepdim=self.keepdim)
        return x_aggr


class MMPrivateOutput(nn.Module):

    def __init__(self, config):
        super(MMPrivateOutput, self).__init__()
        self.num_modes = config.num_modes
        self.feat_dim = config.feat_dim
        feat_dim_allmode = self.feat_dim * self.num_modes
        self.group_linear = nn.Conv1d(feat_dim_allmode, feat_dim_allmode, 1, groups=self.num_modes)
        self.resout_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, shortcut):
        x = self.group_linear(x)
        x_comb = x + shortcut
        shape_4d = x.shape[0], self.num_modes, self.feat_dim, x.shape[2]
        x_comb_4d = x.view(shape_4d).permute([0, 1, 3, 2])
        x_drop_4d = self.dropout(x_comb_4d)
        x_normed = self.resout_norm_layer(x_drop_4d)
        return x_normed


class MMSharedMid(nn.Module):

    def __init__(self, config):
        super(MMSharedMid, self).__init__()
        self.num_modes = config.num_modes
        self.feat_dim = config.feat_dim
        feat_dim_allmode = self.feat_dim * self.num_modes
        self.shared_linear = nn.Linear(self.feat_dim, self.feat_dim)
        self.mid_act_fn = config.act_fun

    def forward(self, x):
        if len(x.shape) == 3:
            shape_4d = x.shape[0], self.num_modes, self.feat_dim, x.shape[2]
            x_4d = x.view(shape_4d).permute([0, 1, 3, 2])
            reshaped = True
        else:
            x_4d = x
            reshaped = False
        x_trans = self.shared_linear(x_4d)
        x_act = self.mid_act_fn(x_trans)
        if reshaped:
            x_act = x_act.permute([0, 1, 3, 2]).reshape(x.shape)
        return x_act


class MMSharedOutput(nn.Module):

    def __init__(self, config):
        super(MMSharedOutput, self).__init__()
        self.num_modes = config.num_modes
        self.feat_dim = config.feat_dim
        self.shared_linear = nn.Linear(self.feat_dim, self.feat_dim)
        self.resout_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, shortcut):
        shape_4d = x.shape[0], self.num_modes, self.feat_dim, x.shape[2]
        if len(x.shape) == 3:
            x_4d = x.view(shape_4d).permute([0, 1, 3, 2])
        else:
            x_4d = x
        if len(shortcut.shape) == 3:
            shortcut_4d = shortcut.view(shape_4d).permute([0, 1, 3, 2])
        else:
            shortcut_4d = shortcut
        x_trans = self.shared_linear(x_4d)
        x_comb = x_trans + shortcut_4d
        x_drop = self.dropout(x_comb)
        x_normed = self.resout_norm_layer(x_drop)
        return x_normed


def print0(*print_args, **kwargs):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        None


class ExpandedFeatTrans(nn.Module):

    def __init__(self, config, name):
        super(ExpandedFeatTrans, self).__init__()
        self.config = config
        self.name = name
        self.in_feat_dim = config.in_feat_dim
        self.feat_dim = config.feat_dim
        self.num_modes = config.num_modes
        self.feat_dim_allmode = self.feat_dim * self.num_modes
        self.first_linear = nn.Linear(self.in_feat_dim, self.feat_dim_allmode, bias=config.v_has_bias)
        self.base_initializer_range = config.base_initializer_range
        self.has_FFN = getattr(config, 'has_FFN', True)
        self.has_input_skip = getattr(config, 'has_input_skip', False)
        self.drop_path = DropPath(config.drop_path_prob) if config.drop_path_prob > 0.0 else nn.Identity()
        print0('{}: v_has_bias: {}, has_FFN: {}, has_input_skip: {}'.format(self.name, config.v_has_bias, self.has_FFN, self.has_input_skip))
        self.pool_modes_keepdim = False
        self.pool_modes_feat = config.pool_modes_feat
        if self.pool_modes_feat == 'softmax':
            agg_basis_feat_dim = self.feat_dim
            self.feat_softaggr = LearnedSoftAggregate(agg_basis_feat_dim, group_dim=1, keepdim=self.pool_modes_keepdim)
        if self.has_FFN:
            self.intermediate = MMSharedMid(self.config)
            if config.trans_output_type == 'shared':
                self.output = MMSharedOutput(config)
            elif config.trans_output_type == 'private':
                self.output = MMPrivateOutput(config)
        if self.has_input_skip:
            self.input_skip_coeff = Parameter(torch.ones(1))
            self.skip_layer_norm = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)

    def add_identity_bias(self):
        if self.config.feattrans_lin1_idbias_scale > 0:
            identity_weight = torch.diag(torch.ones(self.feat_dim)) * self.base_initializer_range * self.config.feattrans_lin1_idbias_scale
            self.first_linear.weight.data[:self.feat_dim, :self.feat_dim] = self.first_linear.weight.data[:self.feat_dim, :self.feat_dim] * 0.5 + identity_weight

    def forward(self, input_feat, attention_probs):
        B, U2, IF = input_feat.shape
        U1 = attention_probs.shape[2]
        F = self.feat_dim
        M = self.num_modes
        mm_first_feat = self.first_linear(input_feat)
        mm_first_feat = mm_first_feat.transpose(1, 2)
        mm_first_feat_4d = mm_first_feat.view(B, M, F, U2).transpose(2, 3)
        mm_first_feat_fusion = torch.matmul(attention_probs, mm_first_feat_4d)
        mm_first_feat_fusion_3d = mm_first_feat_fusion.transpose(2, 3).reshape(B, M * F, U1)
        mm_first_feat = mm_first_feat_fusion_3d
        if self.has_FFN:
            mm_mid_feat = self.intermediate(mm_first_feat)
            mm_last_feat = self.output(mm_mid_feat, mm_first_feat)
            mm_trans_feat = mm_last_feat
        else:
            mm_trans_feat = mm_first_feat_fusion
        if self.pool_modes_feat == 'softmax':
            trans_feat = self.feat_softaggr(mm_trans_feat)
        elif self.pool_modes_feat == 'max':
            trans_feat = mm_trans_feat.max(dim=1)[0]
        elif self.pool_modes_feat == 'mean':
            trans_feat = mm_trans_feat.mean(dim=1)
        elif self.pool_modes_feat == 'none':
            trans_feat = mm_trans_feat
        if self.has_input_skip:
            trans_feat = self.input_skip_coeff * input_feat + self.drop_path(trans_feat)
            trans_feat = self.skip_layer_norm(trans_feat)
        return trans_feat


class MultiHeadFeatTrans(nn.Module):

    def __init__(self, config, name):
        super(MultiHeadFeatTrans, self).__init__()
        self.config = config
        self.name = name
        self.in_feat_dim = config.in_feat_dim
        self.feat_dim = config.feat_dim
        self.num_modes = config.num_modes
        self.feat_dim_onehead = self.feat_dim // self.num_modes
        self.feat_dim_allhead = self.feat_dim_onehead * self.num_modes
        self.first_linear = nn.Linear(self.in_feat_dim, self.feat_dim_allhead)
        None
        config.num_modes = 1
        self.intermediate = MMSharedMid(self.config)
        if config.trans_output_type == 'shared':
            self.output = MMSharedOutput(config)
        elif config.trans_output_type == 'private':
            self.output = MMPrivateOutput(config)
        self.apply_attn_early = config.apply_attn_early

    def add_identity_bias(self):
        if self.config.feattrans_lin1_idbias_scale > 0:
            identity_weight = torch.diag(torch.ones(self.feat_dim)) * self.config.initializer_range * self.config.feattrans_lin1_idbias_scale
            self.first_linear.weight.data[:self.feat_dim] = self.first_linear.weight.data[:self.feat_dim] * 0.5 + identity_weight

    def forward(self, input_feat, attention_probs, attention_scores):
        mm_first_feat = self.first_linear(input_feat)
        mm_first_feat = mm_first_feat.permute(0, 2, 1)
        if self.apply_attn_early:
            shape_4d = mm_first_feat.shape[0], self.num_modes, self.feat_dim_onehead, mm_first_feat.shape[2]
            mm_first_feat_4d = mm_first_feat.view(shape_4d).permute([0, 1, 3, 2])
            mm_first_feat_fusion = torch.matmul(attention_probs, mm_first_feat_4d)
            mm_first_feat_fusion_3d = mm_first_feat_fusion.permute([0, 1, 3, 2]).reshape(mm_first_feat.shape)
            mm_first_feat = mm_first_feat_fusion_3d
        mm_mid_feat = self.intermediate(mm_first_feat)
        mm_last_feat = self.output(mm_mid_feat, mm_first_feat)
        if attention_probs is not None and not self.apply_attn_early:
            mm_trans_feat = torch.matmul(attention_probs, mm_last_feat)
        else:
            mm_trans_feat = mm_last_feat
        trans_feat = mm_trans_feat.squeeze(1)
        return trans_feat


class SETransInitWeights(nn.Module):
    """ An abstract class to handle weights initialization """

    def __init__(self, config, *inputs, **kwargs):
        super(SETransInitWeights, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
            type(module.weight)      # <class 'torch.nn.parameter.Parameter'>
            type(module.weight.data) # <class 'torch.Tensor'>
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            base_initializer_range = self.config.base_initializer_range
            module.weight.data.normal_(mean=0.0, std=base_initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CrossAttFeatTrans(SETransInitWeights):

    def __init__(self, config, name):
        super(CrossAttFeatTrans, self).__init__(config)
        self.config = config
        self.name = name
        self.num_modes = config.num_modes
        self.in_feat_dim = config.in_feat_dim
        self.feat_dim = config.feat_dim
        self.attention_mode_dim = self.in_feat_dim // self.num_modes
        self.att_size_allmode = self.num_modes * self.attention_mode_dim
        self.query = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.key = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.base_initializer_range = config.base_initializer_range
        self.out_attn_scores_only = config.out_attn_scores_only
        self.out_attn_probs_only = config.out_attn_probs_only
        self.ablate_multihead = config.ablate_multihead
        if self.out_attn_scores_only or self.out_attn_probs_only:
            self.out_trans = None
            if self.num_modes > 1:
                self.attn_softaggr = LearnedSoftAggregate(1, group_dim=1, keepdim=True)
        elif self.ablate_multihead:
            self.out_trans = MultiHeadFeatTrans(config, name + '-out_trans')
        else:
            self.out_trans = ExpandedFeatTrans(config, name + '-out_trans')
        self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.tie_qk_scheme = config.tie_qk_scheme
        print0('{}: in_feat_dim: {}, feat_dim: {}, modes: {}, qk_have_bias: {}'.format(self.name, self.in_feat_dim, self.feat_dim, self.num_modes, config.qk_have_bias))
        if config.pos_code_type == 'bias':
            self.pos_code_weight = config.pos_code_weight
            print0('Positional biases weight: {:.3}'.format(self.pos_code_weight))
        else:
            self.pos_code_weight = 1
        self.attn_clip = config.attn_clip
        if 'attn_diag_cycles' in config.__dict__:
            self.attn_diag_cycles = config.attn_diag_cycles
        else:
            self.attn_diag_cycles = 1000
        self.max_attn = 0
        self.clamp_count = 0
        self.call_count = 0
        self.apply(self.init_weights)
        self.apply(tie_qk)
        self.apply(add_identity_bias)

    def tie_qk(self, tie_qk_scheme=None):
        if tie_qk_scheme is not None:
            self.tie_qk_scheme = tie_qk_scheme
        if self.tie_qk_scheme == 'shared':
            self.key.weight = self.query.weight
            if self.key.bias is not None:
                self.key.bias = self.query.bias
        elif self.tie_qk_scheme == 'loose':
            self.key.weight.data.copy_(self.query.weight)
            if self.key.bias is not None:
                self.key.bias.data.copy_(self.query.bias)

    def add_identity_bias(self):
        identity_weight = torch.diag(torch.ones(self.attention_mode_dim)) * self.base_initializer_range * self.config.query_idbias_scale
        repeat_count = self.in_feat_dim // self.attention_mode_dim
        identity_weight = identity_weight.repeat([1, repeat_count])
        self.key.weight.data[:self.attention_mode_dim] = self.key.weight.data[:self.attention_mode_dim] * 0.5 + identity_weight

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_modes, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_feat, key_feat=None, pos_biases=None, attention_mask=None):
        if key_feat is None:
            key_feat = query_feat
        mixed_query_layer = self.query(query_feat)
        mixed_key_layer = self.key(key_feat)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_mode_dim)
        with torch.no_grad():
            curr_max_attn = attention_scores.max().item()
            curr_avg_attn = attention_scores.abs().mean().item()
        if curr_max_attn > self.max_attn:
            self.max_attn = curr_max_attn
        if curr_max_attn > self.attn_clip:
            attention_scores = torch.clamp(attention_scores, -self.attn_clip, self.attn_clip)
            self.clamp_count += 1
        self.call_count += 1
        if self.training:
            if self.call_count % self.attn_diag_cycles == 0:
                print0('max-attn: {:.2f}, avg-attn: {:.2f}, clamp-count: {}'.format(self.max_attn, curr_avg_attn, self.clamp_count))
                self.max_attn = 0
                self.clamp_count = 0
        if pos_biases is not None:
            attention_scores = attention_scores + self.pos_code_weight * pos_biases
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        if self.out_attn_scores_only:
            if self.num_modes > 1:
                attention_scores = self.attn_softaggr(attention_scores)
            return attention_scores
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.att_dropout(attention_probs)
        if self.out_attn_probs_only:
            return attention_probs
        else:
            out_feat = self.out_trans(key_feat, attention_probs)
            return out_feat


class LearnedSinuPosEmbedder(nn.Module):

    def __init__(self, pos_dim, pos_embed_dim, omega=1, affine=True):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_embed_dim = pos_embed_dim
        self.pos_fc = nn.Linear(self.pos_dim, self.pos_embed_dim, bias=True)
        self.pos_mix_norm_layer = nn.LayerNorm(self.pos_embed_dim, eps=1e-12, elementwise_affine=affine)
        self.omega = omega
        print0('Learnable Sinusoidal positional encoding')

    def forward(self, pos_normed):
        pos_embed_sum = 0
        pos_embed0 = self.pos_fc(pos_normed)
        pos_embed_sin = torch.sin(self.omega * pos_embed0[:, :, 0::2])
        pos_embed_cos = torch.cos(self.omega * pos_embed0[:, :, 1::2])
        pos_embed_mix = torch.stack((pos_embed_sin, pos_embed_cos), dim=3).view(pos_embed0.shape)
        pos_embed_out = self.pos_mix_norm_layer(pos_embed_mix)
        return pos_embed_out


class RandPosEmbedder(nn.Module):

    def __init__(self, pos_dim, pos_embed_dim, shape, affine):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_embed_dim = pos_embed_dim
        height, width = shape
        self.pos_embed = nn.Embedding(height * width, pos_embed_dim)
        self.pos_embed_norm_layer = nn.LayerNorm(self.pos_embed_dim, eps=1e-12, elementwise_affine=affine)
        None

    def forward(self, pos_normed):
        B, N, D = pos_normed.shape
        pos_embed_1 = self.pos_embed.weight
        pos_embed_out_1 = self.pos_embed_norm_layer(pos_embed_1)
        pos_embed_out = pos_embed_out_1.unsqueeze(0).repeat((B, 1, 1))
        return pos_embed_out


def positionalencoding2d(pos_embed_dim, height, width):
    """
    :param pos_embed_dim: dimension of the model embeddings
    :param height: height of the positions
    :param width: width of the positions
    :return: height * width * pos_embed_dim matrix
    """
    if pos_embed_dim % 4 != 0:
        raise ValueError('Cannot use sin/cos positional encoding with odd dimension (got dim={:d})'.format(pos_embed_dim))
    pe = torch.zeros(pos_embed_dim, height, width)
    pos_embed_dim = int(pos_embed_dim / 2)
    div_term = torch.exp(torch.arange(0.0, pos_embed_dim, 2) * -(math.log(10000.0) / pos_embed_dim))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:pos_embed_dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:pos_embed_dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[pos_embed_dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[pos_embed_dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe = pe.permute(1, 2, 0)
    return pe


class SinuPosEmbedder(nn.Module):

    def __init__(self, pos_dim, pos_embed_dim, shape, affine):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_embed_dim = pos_embed_dim
        self.pos_embed = positionalencoding2d(pos_embed_dim, shape[0], shape[1])
        self.pos_embed = self.pos_embed.reshape((shape[0] * shape[1], pos_embed_dim))
        None

    def forward(self, pos_normed):
        B, N, D = pos_normed.shape
        pos_embed_out = self.pos_embed.unsqueeze(0).repeat((B, 1, 1))
        return pos_embed_out


class SlidingPosBiases2D(nn.Module):

    def __init__(self, pos_dim=2, pos_bias_radius=7, max_pos_size=(200, 200)):
        super().__init__()
        self.pos_dim = pos_dim
        self.R = R = pos_bias_radius
        pos_bias_shape = [(pos_bias_radius * 2 + 1) for i in range(pos_dim)]
        self.biases = Parameter(torch.zeros(pos_bias_shape))
        if self.pos_dim == 2:
            all_h1s, all_w1s, all_h2s, all_w2s = [], [], [], []
            for i in range(max_pos_size[0]):
                i_h1s, i_w1s, i_h2s, i_w2s = [], [], [], []
                for j in range(max_pos_size[1]):
                    h1s, w1s, h2s, w2s = torch.meshgrid(torch.tensor(i), torch.tensor(j), torch.arange(i, i + 2 * R + 1), torch.arange(j, j + 2 * R + 1))
                    i_h1s.append(h1s)
                    i_w1s.append(w1s)
                    i_h2s.append(h2s)
                    i_w2s.append(w2s)
                i_h1s = torch.cat(i_h1s, dim=1)
                i_w1s = torch.cat(i_w1s, dim=1)
                i_h2s = torch.cat(i_h2s, dim=1)
                i_w2s = torch.cat(i_w2s, dim=1)
                all_h1s.append(i_h1s)
                all_w1s.append(i_w1s)
                all_h2s.append(i_h2s)
                all_w2s.append(i_w2s)
            all_h1s = torch.cat(all_h1s, dim=0)
            all_w1s = torch.cat(all_w1s, dim=0)
            all_h2s = torch.cat(all_h2s, dim=0)
            all_w2s = torch.cat(all_w2s, dim=0)
        else:
            breakpoint()
        self.register_buffer('all_h1s', all_h1s, persistent=False)
        self.register_buffer('all_w1s', all_w1s, persistent=False)
        self.register_buffer('all_h2s', all_h2s, persistent=False)
        self.register_buffer('all_w2s', all_w2s, persistent=False)
        print0(f'Sliding-window Positional Biases, r: {R}, max size: {max_pos_size}')

    def forward(self, feat_shape, device):
        R = self.R
        spatial_shape = feat_shape[-self.pos_dim:]
        padded_pos_shape = list(spatial_shape) + [(2 * R + spatial_shape[i]) for i in range(self.pos_dim)]
        padded_pos_biases = torch.zeros(padded_pos_shape, device=device)
        if self.pos_dim == 2:
            H, W = spatial_shape
            all_h1s = self.all_h1s[:H, :W]
            all_w1s = self.all_w1s[:H, :W]
            all_h2s = self.all_h2s[:H, :W]
            all_w2s = self.all_w2s[:H, :W]
            padded_pos_biases[all_h1s, all_w1s, all_h2s, all_w2s] = self.biases
        pos_biases = padded_pos_biases[:, :, R:-R, R:-R]
        return pos_biases


class ZeroEmbedder(nn.Module):

    def __init__(self, pos_embed_dim):
        super().__init__()
        self.pos_embed_dim = pos_embed_dim
        None

    def forward(self, pos_normed):
        B, N, D = pos_normed.shape
        zero_pos_embed = torch.zeros(B, N, self.pos_embed_dim, requires_grad=False)
        return zero_pos_embed


class SETransInputFeatEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.in_feat_dim
        self.pos_embed_dim = self.feat_dim
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.comb_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
        self.pos_code_type = config.pos_code_type
        if config.pos_code_type != 'bias':
            self.pos_code_weight = config.pos_code_weight
            print0('Positional embedding weight: {:.3}'.format(self.pos_code_weight))
        else:
            self.pos_code_weight = 0
        if config.pos_code_type == 'lsinu':
            self.pos_coder = LearnedSinuPosEmbedder(config.pos_dim, self.pos_embed_dim, omega=1, affine=False)
        elif config.pos_code_type == 'rand':
            self.pos_coder = RandPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif config.pos_code_type == 'sinu':
            self.pos_coder = SinuPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif config.pos_code_type == 'zero':
            self.pos_coder = ZeroEmbedder(self.pos_embed_dim)
        elif config.pos_code_type == 'bias':
            self.pos_coder = SlidingPosBiases2D(config.pos_dim, config.pos_bias_radius)
        self.cached_pos_code = None
        self.cached_feat_shape = None

    def pos_code_lookup_cache(self, vis_feat_shape, device, voxels_pos_normed):
        if self.pos_code_type == 'bias':
            if self.training or self.cached_pos_code is None or self.cached_feat_shape != vis_feat_shape:
                self.cached_pos_code = self.pos_coder(vis_feat_shape, device)
                self.cached_feat_shape = vis_feat_shape
        elif self.training or self.cached_pos_code is None or self.cached_feat_shape != voxels_pos_normed.shape:
            self.cached_pos_code = self.pos_coder(voxels_pos_normed)
            self.cached_feat_shape = voxels_pos_normed.shape
        return self.cached_pos_code

    def forward(self, vis_feat, voxels_pos, return_pos_biases=True):
        batch, dim, ht, wd = vis_feat.shape
        if self.pos_code_type != 'bias':
            voxels_pos_normed = voxels_pos / voxels_pos.max()
            voxels_pos_normed = voxels_pos_normed.view(batch, ht * wd, -1)
            pos_embed = self.pos_code_lookup_cache(vis_feat.shape, vis_feat.device, voxels_pos_normed)
            pos_biases = None
        else:
            pos_embed = 0
            if return_pos_biases:
                pos_biases = self.pos_code_lookup_cache(vis_feat.shape, vis_feat.device, None)
                pos_biases = pos_biases.reshape(1, 1, ht * wd, ht * wd)
            else:
                pass
        vis_feat = vis_feat.view(batch, dim, ht * wd).transpose(1, 2)
        feat_comb = vis_feat + self.pos_code_weight * pos_embed
        feat_normed = self.comb_norm_layer(feat_comb)
        feat_normed = self.dropout(feat_normed)
        if return_pos_biases:
            return feat_normed, pos_biases
        else:
            return feat_normed


def gen_all_indices(shape, device):
    indices = torch.arange(shape.numel(), device=device).view(shape)
    out = []
    for dim_size in reversed(shape):
        out.append(indices % dim_size)
        indices = torch.div(indices, dim_size, rounding_mode='trunc')
    return torch.stack(tuple(reversed(out)), len(shape))


class TransCorrBlock(CorrBlock, nn.Module):

    def __init__(self, config, num_levels=4, radius=4, do_corr_global_norm=False):
        nn.Module.__init__(self)
        self.num_levels = num_levels
        self.radius = radius
        self.config = config
        self.setrans = CrossAttFeatTrans(self.config, 'Inter-frame correlation block')
        self.vispos_encoder = SETransInputFeatEncoder(self.config)
        self.coords2 = None
        self.do_corr_global_norm = do_corr_global_norm

    def update(self, fmap1, fmap2, fmap1o, fmap2o, coords1, coords2=None):
        self.corr_pyramid = []
        coords1 = coords1.permute(0, 2, 3, 1).flip(-1)
        if coords2 is None:
            coords2 = gen_all_indices(fmap2.shape[2:], device=fmap2.device)
            coords2 = coords2.unsqueeze(0).repeat(fmap2.shape[0], 1, 1, 1)
        vispos1, pos_biases = self.vispos_encoder(fmap1, coords1, return_pos_biases=True)
        vispos2 = self.vispos_encoder(fmap2, coords2, return_pos_biases=False)
        batch, dim, ht, wd = fmap1.shape
        if fmap1o is not None and fmap2o is not None:
            vispos1o = self.vispos_encoder(fmap1o, coords1, return_pos_biases=False)
            vispos2o = self.vispos_encoder(fmap2o, coords2, return_pos_biases=False)
            corr_1t2o = self.corr(ht, wd, vispos1, vispos2o, pos_biases)
            corr_1o2t = self.corr(ht, wd, vispos1o, vispos2, pos_biases)
            corr = torch.cat([corr_1t2o, corr_1o2t], dim=3)
        else:
            corr = self.corr(ht, wd, vispos1, vispos2, pos_biases)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        if 'SAVECORR' in os.environ:
            corr_savepath = os.environ['SAVECORR']
            corr2 = corr.detach().cpu().reshape(batch, h1, w1, h2, w2)
            torch.save(corr2, corr_savepath)
            None
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def corr(self, ht, wd, vispos1, vispos2, pos_biases):
        batch, ht_wd, dim = vispos1.shape
        assert ht_wd == ht * wd
        corr = self.setrans(vispos1, vispos2, pos_biases)
        if self.do_corr_global_norm:
            B, C, H, W = corr.shape
            corr_3d = corr.view(B, C, H * W)
            corr_normed = F.layer_norm(corr_3d, (corr_3d.shape[2],), eps=1e-12)
            corr = corr_normed.view(B, C, H, W)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr


class SequenceLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
        self.max_flow = args.max_flow

    def forward(self, outputs, inputs):
        """ Loss function defined over sequence of flow predictions """
        flow_preds = outputs['flow_preds']
        flow_gt = inputs['flows'][:, 0]
        valid = inputs['valids'][:, 0]
        n_predictions = len(flow_preds)
        flow_loss = 0.0
        mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()
        return flow_loss


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, dilation=(1, 1), kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, dilation=dilation[0])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, dilation=dilation[1])
        self.relu = nn.ReLU(inplace=True)
        self.projector = nn.Conv2d(in_planes, planes, kernel_size=1)

    def forward(self, x):
        y = x
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        if self.projector is not None:
            x = self.projector(x)
        return self.relu(x + y)


class BottleneckBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class RelPosEmb(nn.Module):

    def __init__(self, max_pos_size, dim_head):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)
        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))
        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)
        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)
        return height_score + width_score


class Attention(nn.Module):

    def __init__(self, *, args, dim, max_pos_size=100, heads=4, dim_head=128):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape
        q, k = self.to_qk(fmap).chunk(2, dim=1)
        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q
        if self.args.position_only:
            sim = self.pos_emb(q)
        elif self.args.position_and_content:
            sim_content = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
            sim_pos = self.pos_emb(q)
            sim = sim_content + sim_pos
        else:
            sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)
        return attn


class Aggregate(nn.Module):

    def __init__(self, args, dim, heads=4, dim_head=128):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape
        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        if self.project is not None:
            out = self.project(out)
        out = fmap + self.gamma * out
        return out


class SelfAttVisPosTrans(nn.Module):

    def __init__(self, config, name):
        nn.Module.__init__(self)
        self.config = copy.copy(config)
        self.name = name
        self.out_attn_only = config.out_attn_scores_only or config.out_attn_probs_only
        self.attn_mask_radius = config.attn_mask_radius
        self.setrans = CrossAttFeatTrans(self.config, name)
        self.vispos_encoder = SETransInputFeatEncoder(self.config)

    def forward(self, x):
        coords = gen_all_indices(x.shape[2:], device=x.device)
        if self.attn_mask_radius > 0:
            coords2 = coords.reshape(-1, 2)
            coords_diff = coords2.unsqueeze(0) - coords2.unsqueeze(1)
            attn_mask = (coords_diff.abs().max(dim=2)[0] > self.attn_mask_radius).float()
            attn_mask = (attn_mask * -1000000000.0).unsqueeze(0).unsqueeze(0)
        else:
            attn_mask = None
        coords = coords.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        x_vispos, pos_biases = self.vispos_encoder(x, coords, return_pos_biases=True)
        x_trans = self.setrans(x_vispos, pos_biases=pos_biases, attention_mask=attn_mask)
        if self.name == 'F2 transformer' and 'SAVEF2' in os.environ:
            f2_attention_probs = self.setrans.attention_probs.detach().cpu()
            f2_attention_probs = f2_attention_probs.mean(dim=1, keepdim=False)
            f2_savepath = os.environ['SAVEF2']
            batch, C, h1, w1 = x.shape
            f2attn = f2_attention_probs.reshape(batch, h1, w1, h1, w1)
            torch.save(f2attn, f2_savepath)
            print0(f'F2 attention tensor saved to {f2_savepath}')
        if not self.out_attn_only:
            x_trans_shape = x_trans.shape
            x_trans = x_trans.permute(0, 2, 1).reshape(x.shape)
        return x_trans


class MMPrivateMid(nn.Module):

    def __init__(self, config):
        super(MMPrivateMid, self).__init__()
        self.num_modes = config.num_modes
        self.feat_dim = config.feat_dim
        feat_dim_allmode = self.feat_dim * self.num_modes
        self.group_linear = nn.Conv1d(feat_dim_allmode, feat_dim_allmode, 1, groups=self.num_modes)
        self.mid_act_fn = config.act_fun

    def forward(self, x):
        x_trans = self.group_linear(x)
        x_act = self.mid_act_fn(x_trans)
        return x


class FlowHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):

    def __init__(self, args, input_dim=128):
        super().__init__()
        self.convc1 = nn.Conv2d(input_dim, 256, 1)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(192 + 64, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, input_dim=input_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class GMAUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        net = self.gru(net, inp_cat)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class ConvBNReLU(nn.Module):
    """Conv with BN and ReLU, used for Strip Corr Module"""

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class StripCrossCorrMap_v2(nn.Module):
    """Strip Cross Corr Augmentation Module by Hao, version2.0"""

    def __init__(self, in_chan=256, out_chan=256, *args, **kwargs):
        super(StripCrossCorrMap_v2, self).__init__()
        self.conv1_1 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1_2 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv2_1 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv2_2 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        fmap1, fmap2 = x
        fmap1_w = self.conv1_1(fmap1)
        batchsize, c_middle, h, w = fmap1_w.size()
        fmap1_w = fmap1_w.view(batchsize, c_middle, -1)
        fmap1_h = self.conv1_2(fmap1)
        batchsize, c_middle, h, w = fmap1_h.size()
        fmap1_h = fmap1_h.view(batchsize, c_middle, -1)
        fmap2_w = self.conv2_1(fmap2)
        fmap2_w = F.avg_pool2d(fmap2_w, [h, 1])
        fmap2_w = fmap2_w.view(batchsize, c_middle, -1).permute(0, 2, 1)
        fmap2_h = self.conv2_2(fmap2)
        fmap2_h = F.avg_pool2d(fmap2_h, [1, w])
        fmap2_h = fmap2_h.view(batchsize, c_middle, -1).permute(0, 2, 1)
        strip_corr_map_w = torch.bmm(fmap2_w, fmap1_w).view(batchsize, w, h, w, 1).permute(0, 2, 3, 4, 1)
        strip_corr_map_h = torch.bmm(fmap2_h, fmap1_h).view(batchsize, h, h, w, 1).permute(0, 2, 3, 1, 4)
        return (strip_corr_map_w + strip_corr_map_h).view(batchsize, h, w, 1, h, w), strip_corr_map_w, strip_corr_map_h

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, torch.nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SmallMotionEncoder(nn.Module):

    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, None, delta_flow


class SmallEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class BasicMotionEncoder_v2(nn.Module):
    """Get Motion Feature from CSFlow, by Hao"""

    def __init__(self, args):
        super(BasicMotionEncoder_v2, self).__init__()
        cor_planes = 2 * (args.corr_levels * (2 * args.corr_radius + 1) ** 2)
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class DICL_MODULE(nn.Module):

    def __init__(self):
        super(DICL_MODULE, self).__init__()
        self.match = nn.Sequential(BasicConv(64, 96, kernel_size=3, padding=1, dilation=1), BasicConv(96, 128, kernel_size=3, stride=2, padding=1), BasicConv(128, 128, kernel_size=3, padding=1, dilation=1), BasicConv(128, 64, kernel_size=3, padding=1, dilation=1), BasicConv(64, 32, kernel_size=4, padding=1, stride=2, deconv=True), nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, x):
        x = self.match(x)
        return x


class FlowEntropy(nn.Module):

    def __init__(self):
        super(FlowEntropy, self).__init__()

    def forward(self, x):
        x = torch.squeeze(x, 1)
        B, U, V, H, W = x.shape
        x = x.view(B, -1, H, W)
        x = F.softmax(x, dim=1).view(B, U, V, H, W)
        global_entropy = (-x * torch.clamp(x, 1e-09, 1 - 1e-09).log()).sum(1).sum(1)[:, np.newaxis]
        global_entropy /= np.log(x.shape[1] * x.shape[2])
        return global_entropy


class FlowRegression(nn.Module):

    def __init__(self, maxU, maxV, flow_reg_by_max):
        super(FlowRegression, self).__init__()
        self.maxU = maxU
        self.maxV = maxV
        self.flow_reg_by_max = flow_reg_by_max

    def forward(self, x):
        assert x.is_contiguous() == True
        sizeU = 2 * self.maxU + 1
        sizeV = 2 * self.maxV + 1
        x = x.squeeze(1)
        B, _, _, H, W = x.shape
        dispU = torch.reshape(torch.arange(-self.maxU, self.maxU + 1, dtype=torch.float32), [1, sizeU, 1, 1, 1])
        dispU = dispU.expand(B, -1, sizeV, H, W).contiguous()
        dispU = dispU.view(B, sizeU * sizeV, H, W)
        dispV = torch.reshape(torch.arange(-self.maxV, self.maxV + 1, dtype=torch.float32), [1, 1, sizeV, 1, 1])
        dispV = dispV.expand(B, sizeU, -1, H, W).contiguous()
        dispV = dispV.view(B, sizeU * sizeV, H, W)
        x = x.view(B, sizeU * sizeV, H, W)
        if self.flow_reg_by_max:
            x = F.softmax(x, dim=1)
        else:
            x = F.softmin(x, dim=1)
        flowU = (x * dispU).sum(dim=1)
        flowV = (x * dispV).sum(dim=1)
        flow = torch.cat((flowU.unsqueeze(1), flowV.unsqueeze(1)), dim=1)
        return flow


class DAP(nn.Module):

    def __init__(self, md=3, dap_by_temperature=False):
        super(DAP, self).__init__()
        self.dap_by_temperature = dap_by_temperature
        dimC = (2 * md + 1) ** 2
        self.dap_layer = BasicConv(dimC, dimC, kernel_size=1, padding=0, stride=1, bn=False, relu=False)
        if self.dap_by_temperature:
            self.dap_layer = BasicConv(dimC, 1, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

    def forward(self, x):
        x = x.squeeze(1)
        bs, du, dv, h, w = x.shape
        x = x.view(bs, du * dv, h, w)
        if self.dap_by_temperature:
            temp = self.dap_layer(x) + 1e-06
            x = x * temp
        else:
            x = self.dap_layer(x)
        return x.view(bs, du, dv, h, w).unsqueeze(1)


BatchNorm = nn.BatchNorm2d


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), BatchNorm(self.expansion * planes))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.deconv = deconv
        if deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, bn=False, relu=True, kernel_size=kernel, stride=2, padding=1)
        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert x.size() == rem.size()
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class FeatureGA(nn.Module):

    def __init__(self):
        super(FeatureGA, self).__init__()
        self.conv_start = nn.Sequential(BasicConv(3, 32, kernel_size=3, padding=1), BasicConv(32, 32, kernel_size=3, stride=2, padding=1), BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)
        self.conv5a = BasicConv(128, 160, kernel_size=3, stride=2, padding=1)
        self.conv6a = BasicConv(160, 192, kernel_size=3, stride=2, padding=1)
        self.deconv6a = Conv2x(192, 160, deconv=True)
        self.deconv5a = Conv2x(160, 128, deconv=True)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)
        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)
        self.conv5b = Conv2x(128, 160)
        self.conv6b = Conv2x(160, 192)
        self.deconv6b = Conv2x(192, 160, deconv=True)
        self.outconv_6 = BasicConv(160, 32, kernel_size=3, padding=1)
        self.deconv5b = Conv2x(160, 128, deconv=True)
        self.outconv_5 = BasicConv(128, 32, kernel_size=3, padding=1)
        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.outconv_4 = BasicConv(96, 32, kernel_size=3, padding=1)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.outconv_3 = BasicConv(64, 32, kernel_size=3, padding=1)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.outconv_2 = BasicConv(48, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.conv5a(x)
        rem5 = x
        x = self.conv6a(x)
        rem6 = x
        x = self.deconv6a(x, rem5)
        rem5 = x
        x = self.deconv5a(x, rem4)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x
        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        rem4 = x
        x = self.conv5b(x, rem5)
        rem5 = x
        x = self.conv6b(x, rem6)
        x = self.deconv6b(x, rem5)
        x6 = self.outconv_6(x)
        x = self.deconv5b(x, rem4)
        x5 = self.outconv_5(x)
        x = self.deconv4b(x, rem3)
        x4 = self.outconv_4(x)
        x = self.deconv3b(x, rem2)
        x3 = self.outconv_3(x)
        x = self.deconv2b(x, rem1)
        x2 = self.outconv_2(x)
        return None, x2, x3, x4, x5, x6


def EPE(input_flow, target_flow):
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()


def realEPE(output, target, sparse=False, valid_range=None, extra_mask=None, use_valid_range=False):
    b, _, h, w = target.size()
    upsampled_output = output
    if use_valid_range and valid_range is not None:
        mask = (target[:, 0, :, :].abs() <= valid_range[1]) & (target[:, 1, :, :].abs() <= valid_range[0])
        mask = mask.unsqueeze(1).expand(-1, 2, -1, -1).float()
        upsampled_output = upsampled_output * mask
        target = target * mask
    return EPE(upsampled_output, target, sparse, mean=True, extra_mask=extra_mask)


class MultiScale_UP(nn.Module):

    def __init__(self, loss_type='L1', weight=[1.0, 0.5, 0.25], valid_range=None, removezero=False, use_valid_range=False):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        self.valid_range = valid_range
        self.removezero = removezero
        self.use_valid_range = use_valid_range

    def forward(self, preds, inputs):
        loss = 0
        loss_list = []
        target = inputs['flows'][:, 0]
        output = preds['flow_preds']
        extra_mask = inputs.get('extra_mask')
        b, _, h, w = target.size()
        for i, cur_output in enumerate(output):
            realflow = F.interpolate(cur_output, (h, w), mode='bilinear', align_corners=True)
            realflow[:, 0, :, :] = realflow[:, 0, :, :] * (w / cur_output.shape[3])
            realflow[:, 1, :, :] = realflow[:, 1, :, :] * (h / cur_output.shape[2])
            with torch.no_grad():
                if i == 0:
                    epe = realEPE(realflow, target, extra_mask=extra_mask)
            if self.loss_type == 'L2':
                lossvalue = torch.norm(realflow - target, p=2, dim=1)
            elif self.loss_type == 'robust':
                lossvalue = (realflow - target).abs().sum(dim=1) + 1e-08
                lossvalue = lossvalue ** 0.4
            elif self.loss_type == 'L1':
                lossvalue = (realflow - target).abs().sum(dim=1)
            else:
                raise NotImplementedError
            if self.use_valid_range and self.valid_range is not None:
                with torch.no_grad():
                    mask = (target[:, 0, :, :].abs() <= self.valid_range[i][1]) & (target[:, 1, :, :].abs() <= self.valid_range[i][0])
            else:
                with torch.no_grad():
                    mask = torch.ones(target[:, 0, :, :].shape).type_as(target)
            lossvalue = lossvalue * mask.float()
            if extra_mask is not None:
                val = extra_mask > 0
                lossvalue = lossvalue[val]
                cur_loss = lossvalue.mean() * self.weight[i]
                assert lossvalue.shape[0] == extra_mask.sum()
            else:
                cur_loss = lossvalue.mean() * self.weight[i]
            loss += cur_loss
            loss_list.append(cur_loss)
        loss = loss / len(output)
        return {'loss': loss, 'loss_list': loss_list, 'epe': epe}


class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):

    def __init__(self):
        super(L2, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target, p=2, dim=1).mean()
        return lossvalue


class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        output = output['flow_preds']
        target = target['flows'][:, 0]
        lossvalue = self.loss(output, target)
        return lossvalue


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        output = output['flow_preds']
        target = target['flows'][:, 0]
        lossvalue = self.loss(output, target)
        return lossvalue


class Decoder(nn.Module):

    def __init__(self, inplane, block, classes, up_classes):
        super(Decoder, self).__init__()
        self.mapping = block(inplane, 128)
        self.cls = nn.Sequential(BatchNorm(128), nn.ReLU(inplace=True), nn.Conv2d(128, classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.up = None
        if up_classes > 0:
            self.up = nn.Sequential(BatchNorm(128), nn.ReLU(inplace=True), nn.ConvTranspose2d(128, up_classes, kernel_size=4, stride=2, padding=1, bias=False), BatchNorm(up_classes), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.mapping(x)
        prob = self.cls(out)
        up_feat = self.up(out) if self.up else None
        return prob, up_feat


class BroadMultiHeadAttention(nn.Module):

    def __init__(self, dim, heads):
        super(BroadMultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim / heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        Q = rearrange(Q.squeeze(), 'i (heads d) -> heads i d', heads=self.heads)
        K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads)
        dots = einsum('hid, bhjd -> bhij', Q, K) * self.scale
        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)
        B, _, _ = K.shape
        _, N, _ = Q.shape
        V = rearrange(V, 'b j (heads d) -> b heads j d', heads=self.heads)
        out = einsum('bhij, bhjd -> bhid', attn, V)
        out = rearrange(out, 'b heads n d -> b n (heads d)', b=B, n=N)
        return out


class MultiHeadAttention(nn.Module):
    """ MultiHeadAttention modified from SwinTransformer
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_proj=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_proj = use_proj
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if self.use_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """ Forward function.
        Args:
            q: input queries with shape of (B, Nq, C)
            k: input keys with shape of (B, Nk, C)
            v: input values with shape of (B, Nk, C)
            mask: (0/-inf) mask with shape of (Nq, Nk) or None
        """
        B, N_q, C = q.shape
        N_kv = k.shape[1]
        dim_per_head = C // self.num_heads
        q = self.Wq(q).reshape(B, N_q, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        k = self.Wk(k).reshape(B, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        v = self.Wv(v).reshape(B, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        if self.use_proj:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class MultiHeadAttentionRelative(nn.Module):

    def __init__(self, dim, heads):
        super(MultiHeadAttentionRelative, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim / heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K, Q_r, K_r):
        """
            Q: [BH1W1, 1, dim]
            K: [BH1W1, H3W3, dim]
            Q_r: [BH1W1, H3W3, dim]
            K_r: [BH1W1, H3W3, dim]
        """
        Q = rearrange(Q, 'b i (heads d) -> b heads i d', heads=self.heads)
        K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads)
        K_r = rearrange(K_r, 'b j (heads d) -> b heads j d', heads=self.heads)
        Q_r = rearrange(Q_r, 'b j (heads d) -> b heads j d', heads=self.heads)
        c_c = einsum('bhid, bhjd -> bhij', Q, K) * self.scale
        c_p = einsum('bhid, bhjd -> bhij', Q, K_r) * self.scale
        p_c = einsum('bhijd, bhikd -> bhijk', Q_r[:, :, :, None, :], K[:, :, :, None, :]) * self.scale
        p_c = torch.squeeze(p_c, dim=4)
        p_c = p_c.permute(0, 1, 3, 2)
        dots = c_c + c_p + p_c
        return self.attend(dots)

    def forward(self, Q, K, V, Q_r, K_r):
        attn = self.attend_with_rpe(Q, K, Q_r, K_r)
        B, HW, _ = Q.shape
        V = rearrange(V, 'b j (heads d) -> b heads j d', heads=self.heads)
        out = einsum('bhij, bhjd -> bhid', attn, V)
        out = rearrange(out, 'b heads hw d -> b hw (heads d)', b=B, hw=HW)
        return out


class ConvNets(nn.Module):

    def __init__(self, in_dim, out_dim, inter_dim, depth, stride=1):
        super(ConvNets, self).__init__()
        self.conv_first = nn.Conv2d(in_dim, inter_dim, kernel_size=3, padding=1, stride=stride)
        self.conv_last = nn.Conv2d(inter_dim, out_dim, kernel_size=3, padding=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.inter_convs = nn.ModuleList([ResidualBlock(inter_dim, inter_dim, norm_fn='none', stride=1) for i in range(depth)])

    def forward(self, x):
        x = self.relu(self.conv_first(x))
        for inter_conv in self.inter_convs:
            x = inter_conv(x)
        x = self.conv_last(x)
        return x


class BasicFuseMotion(nn.Module):

    def __init__(self, args):
        super(BasicFuseMotion, self).__init__()
        cor_planes = args.motion_feature_dim
        out_planes = args.query_latent_dim
        self.normf1 = nn.InstanceNorm2d(128)
        self.normf2 = nn.InstanceNorm2d(128)
        self.convf1 = nn.Conv2d(2, 128, 3, padding=1)
        self.convf2 = nn.Conv2d(128, 128, 3, padding=1)
        self.convf3 = nn.Conv2d(128, 64, 3, padding=1)
        s = 1
        self.normc1 = nn.InstanceNorm2d(256 * s)
        self.normc2 = nn.InstanceNorm2d(256 * s)
        self.normc3 = nn.InstanceNorm2d(256 * s)
        self.convc1 = nn.Conv2d(cor_planes + 128, 256 * s, 1, padding=0)
        self.convc2 = nn.Conv2d(256 * s, 256 * s, 3, padding=1)
        self.convc3 = nn.Conv2d(256 * s, 256 * s, 3, padding=1)
        self.convc4 = nn.Conv2d(256 * s, 256 * s, 3, padding=1)
        self.conv = nn.Conv2d(256 * s + 64, out_planes, 1, padding=0)

    def forward(self, flow, feat, context1=None):
        flo = F.relu(self.normf1(self.convf1(flow)))
        flo = F.relu(self.normf2(self.convf2(flo)))
        flo = self.convf3(flo)
        feat = torch.cat([feat, context1], dim=1)
        feat = F.relu(self.normc1(self.convc1(feat)))
        feat = F.relu(self.normc2(self.convc2(feat)))
        feat = F.relu(self.normc3(self.convc3(feat)))
        feat = self.convc4(feat)
        feat = torch.cat([flo, feat], dim=1)
        feat = F.relu(self.conv(feat))
        return feat


class DirectMeanMaskPredictor(nn.Module):

    def __init__(self, args):
        super(DirectMeanMaskPredictor, self).__init__()
        self.flow_head = FlowHead(args.predictor_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(args.predictor_dim, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, motion_features):
        delta_flow = self.flow_head(motion_features)
        mask = 0.25 * self.mask(motion_features)
        return mask, delta_flow


class BaiscMeanPredictor(nn.Module):

    def __init__(self, args, hidden_dim=128):
        super(BaiscMeanPredictor, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, latent, flow):
        motion_features = self.encoder(flow, latent)
        delta_flow = self.flow_head(motion_features)
        mask = 0.25 * self.mask(motion_features)
        return mask, delta_flow


class BasicRPEEncoder(nn.Module):

    def __init__(self, args):
        super(BasicRPEEncoder, self).__init__()
        self.args = args
        dim = args.query_latent_dim
        self.encoder = nn.Sequential(nn.Linear(2, dim // 2), nn.ReLU(inplace=True), nn.Linear(dim // 2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, rpe_tokens):
        return self.encoder(rpe_tokens)


Size_ = Tuple[int, int]


class GlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def LinearPositionEmbeddingSine(x, dim=128, NORMALIZE_FACOR=1 / 200):
    freq_bands = torch.linspace(0, dim // 4 - 1, dim // 4)
    return torch.cat([torch.sin(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR), torch.cos(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR), torch.sin(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR), torch.cos(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR)], dim=-1)


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)


class GlobalSubSampleAttnRPE(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.sr_ratio - W % self.sr_ratio) % self.sr_ratio
        pad_b = (self.sr_ratio - H % self.sr_ratio) % self.sr_ratio
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        padded_size = Hp, Wp
        padded_N = Hp * Wp
        x = x.view(B, -1, C)
        coords = coords_grid(B, *padded_size)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        q = self.q(x + coords_enc).reshape(B, padded_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *padded_size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        coords = coords_grid(B, padded_size[0] // self.sr_ratio, padded_size[1] // self.sr_ratio)
        coords = coords.view(B, 2, -1).permute(0, 2, 1) * self.sr_ratio
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        k = self.k(x + coords_enc).reshape(B, padded_size[0] // self.sr_ratio * (padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, padded_size[0] // self.sr_ratio * (padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, Hp, Wp, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSubSampleAttnRPEContext(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1, vert_c_dim=0):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.vert_c_dim = vert_c_dim
        self.context_proj = nn.Linear(256, vert_c_dim)
        self.q = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_key = nn.Conv2d(dim + vert_c_dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_value = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        C_qk = C + self.vert_c_dim
        H, W = size
        context = context.repeat(B // context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H * W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)
        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)
        pad_l = pad_t = 0
        pad_r = (self.sr_ratio - W % self.sr_ratio) % self.sr_ratio
        pad_b = (self.sr_ratio - H % self.sr_ratio) % self.sr_ratio
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        padded_size = Hp, Wp
        padded_N = Hp * Wp
        x = x.view(B, -1, C)
        x_qk = x_qk.view(B, -1, C_qk)
        coords = coords_grid(B, *padded_size)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk)
        q = self.q(x_qk + coords_enc).reshape(B, padded_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_key is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *padded_size)
            x_qk = x_qk.permute(0, 2, 1).reshape(B, C_qk, *padded_size)
            x = self.sr_value(x).reshape(B, C, -1).permute(0, 2, 1)
            x_qk = self.sr_key(x_qk).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x_qk = self.norm(x_qk)
        coords = coords_grid(B, padded_size[0] // self.sr_ratio, padded_size[1] // self.sr_ratio)
        coords = coords.view(B, 2, -1).permute(0, 2, 1) * self.sr_ratio
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        k = self.k(x_qk + coords_enc).reshape(B, padded_size[0] // self.sr_ratio * (padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, padded_size[0] // self.sr_ratio * (padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, Hp, Wp, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupAttnRPE(nn.Module):
    """ Latent cost tokens attend to different group
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1, cfg=None):
        super(GroupAttnRPE, self).__init__()
        assert ws != 1
        assert cfg is not None
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        assert cfg.cost_latent_token_num % 5 == 0, 'cost_latent_token_num should be divided by 5.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.cfg = cfg
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        batch_num = B // 5
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        padded_N = Hp * Wp
        coords = coords_grid(B, Hp, Wp)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        coords_enc = coords_enc.reshape(B, Hp, Wp, C)
        q = self.q(x + coords_enc).reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        q = q.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        v = self.v(x)
        k = self.k(x + coords_enc)
        kv = torch.cat([k, v], dim=-1)
        kv_up = torch.cat([kv[:batch_num, self.ws:Hp, :, :], kv[:batch_num, Hp - self.ws:Hp, :, :]], dim=1)
        kv_down = torch.cat([kv[batch_num:batch_num * 2, :self.ws, :, :], kv[batch_num:batch_num * 2, :Hp - self.ws, :, :]], dim=1)
        kv_left = torch.cat([kv[batch_num * 2:batch_num * 3, :, self.ws:Wp, :], kv[batch_num * 2:batch_num * 3, :, Wp - self.ws:Wp, :]], dim=2)
        kv_right = torch.cat([kv[batch_num * 3:batch_num * 4, :, :self.ws, :], kv[batch_num * 3:batch_num * 4, :, :Wp - self.ws, :]], dim=2)
        kv_center = kv[batch_num * 4:batch_num * 5, :, :, :]
        kv_shifted = torch.cat([kv_up, kv_down, kv_left, kv_right, kv_center], dim=0)
        k, v = torch.split(kv_shifted, [self.dim, self.dim], dim=-1)
        k = k.reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        k = k.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        v = v.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupAttnRPEContext(nn.Module):
    """ Latent cost tokens attend to different group
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1, cfg=None, vert_c_dim=0):
        super(GroupAttnRPEContext, self).__init__()
        assert ws != 1
        assert cfg is not None
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        assert cfg.cost_latent_token_num % 5 == 0, 'cost_latent_token_num should be divided by 5.'
        assert vert_c_dim > 0, 'vert_c_dim should not be 0'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.vert_c_dim = vert_c_dim
        self.cfg = cfg
        self.context_proj = nn.Linear(256, vert_c_dim)
        self.q = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        C_qk = C + self.vert_c_dim
        H, W = size
        batch_num = B // 5
        context = context.repeat(B // context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H * W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)
        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        padded_N = Hp * Wp
        coords = coords_grid(B, Hp, Wp)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk)
        coords_enc = coords_enc.reshape(B, Hp, Wp, C_qk)
        q = self.q(x_qk + coords_enc).reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        q = q.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        v = self.v(x)
        k = self.k(x_qk + coords_enc)
        kv = torch.cat([k, v], dim=-1)
        kv_up = torch.cat([kv[:batch_num, self.ws:Hp, :, :], kv[:batch_num, Hp - self.ws:Hp, :, :]], dim=1)
        kv_down = torch.cat([kv[batch_num:batch_num * 2, :self.ws, :, :], kv[batch_num:batch_num * 2, :Hp - self.ws, :, :]], dim=1)
        kv_left = torch.cat([kv[batch_num * 2:batch_num * 3, :, self.ws:Wp, :], kv[batch_num * 2:batch_num * 3, :, Wp - self.ws:Wp, :]], dim=2)
        kv_right = torch.cat([kv[batch_num * 3:batch_num * 4, :, :self.ws, :], kv[batch_num * 3:batch_num * 4, :, :Wp - self.ws, :]], dim=2)
        kv_center = kv[batch_num * 4:batch_num * 5, :, :, :]
        kv_shifted = torch.cat([kv_up, kv_down, kv_left, kv_right, kv_center], dim=0)
        k, v = torch.split(kv_shifted, [self.dim, self.dim], dim=-1)
        k = k.reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        k = k.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        v = v.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocallyGroupedAttn(nn.Module):
    """ LSA: self attention within a group
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocallyGroupedAttnRPE(nn.Module):
    """ LSA: self attention within a group
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1):
        assert ws != 1
        super(LocallyGroupedAttnRPE, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        v = self.v(x).reshape(B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        coords = coords_grid(B, self.ws, self.ws)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C).view(B, self.ws, self.ws, C)
        x = x + coords_enc[:, None, None, :, :, :]
        q = self.q(x).reshape(B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        k = self.k(x).reshape(B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocallyGroupedAttnRPEContext(nn.Module):
    """ LSA: self attention within a group
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1, vert_c_dim=0):
        assert ws != 1
        super(LocallyGroupedAttnRPEContext, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.vert_c_dim = vert_c_dim
        self.context_proj = nn.Linear(256, vert_c_dim)
        self.q = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        C_qk = C + self.vert_c_dim
        context = context.repeat(B // context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H * W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)
        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        x_qk = x_qk.reshape(B, _h, self.ws, _w, self.ws, C_qk).transpose(2, 3)
        v = self.v(x).reshape(B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        coords = coords_grid(B, self.ws, self.ws)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk).view(B, self.ws, self.ws, C_qk)
        x_qk = x_qk + coords_enc[:, None, None, :, :, :]
        q = self.q(x_qk).reshape(B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        k = self.k(x_qk).reshape(B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None, with_rpe=False, vert_c_dim=0, groupattention=False, cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if groupattention:
            assert with_rpe, 'Not implementing groupattention without rpe'
            if vert_c_dim > 0:
                self.attn = GroupAttnRPEContext(dim, num_heads, attn_drop, drop, ws, cfg, vert_c_dim)
            else:
                self.attn = GroupAttnRPE(dim, num_heads, attn_drop, drop, ws, cfg)
        elif ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, drop)
        elif ws == 1:
            if with_rpe:
                if vert_c_dim > 0:
                    self.attn = GlobalSubSampleAttnRPEContext(dim, num_heads, attn_drop, drop, sr_ratio, vert_c_dim)
                else:
                    self.attn = GlobalSubSampleAttnRPE(dim, num_heads, attn_drop, drop, sr_ratio)
            else:
                self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        elif with_rpe:
            if vert_c_dim > 0:
                self.attn = LocallyGroupedAttnRPEContext(dim, num_heads, attn_drop, drop, ws, vert_c_dim)
            else:
                self.attn = LocallyGroupedAttnRPE(dim, num_heads, attn_drop, drop, ws)
        else:
            self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size: Size_, context=None):
        x = x + self.drop_path(self.attn(self.norm1(x), size, context))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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


class TwinsSelfAttentionLayer(nn.Module):

    def __init__(self, args):
        super(TwinsSelfAttentionLayer, self).__init__()
        self.args = args
        embed_dim = 256
        num_heads = 8
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        dpr = 0.0
        drop_rate = 0.0
        attn_drop_rate = 0.0
        self.local_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=ws, with_rpe=True)
        self.global_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=1, with_rpe=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, x, tgt, size):
        x = self.local_block(x, size)
        x = self.global_block(x, size)
        tgt = self.local_block(tgt, size)
        tgt = self.global_block(tgt, size)
        return x, tgt


class CrossGlobalSubSampleAttnRPE(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, tgt, size: Size_):
        B, N, C = x.shape
        coords = coords_grid(B, *size)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        q = self.q(x + coords_enc).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr is not None:
            tgt = tgt.permute(0, 2, 1).reshape(B, C, *size)
            tgt = self.sr(tgt).reshape(B, C, -1).permute(0, 2, 1)
            tgt = self.norm(tgt)
        coords = coords_grid(B, size[0] // self.sr_ratio, size[1] // self.sr_ratio)
        coords = coords.view(B, 2, -1).permute(0, 2, 1) * self.sr_ratio
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        k = self.k(tgt + coords_enc).reshape(B, size[0] // self.sr_ratio * (size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(tgt).reshape(B, size[0] // self.sr_ratio * (size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None, with_rpe=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossGlobalSubSampleAttnRPE(dim, num_heads, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, src, tgt, size: Size_):
        src_shortcut, tgt_shortcut = src, tgt
        src, tgt = self.norm1(src), self.norm1(tgt)
        src = src_shortcut + self.drop_path(self.attn(src, tgt, size))
        tgt = tgt_shortcut + self.drop_path(self.attn(tgt, src, size))
        src = src + self.drop_path(self.mlp(self.norm2(src)))
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))
        return src, tgt


class TwinsCrossAttentionLayer(nn.Module):

    def __init__(self, args):
        super(TwinsCrossAttentionLayer, self).__init__()
        self.args = args
        embed_dim = 256
        num_heads = 8
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        dpr = 0.0
        drop_rate = 0.0
        attn_drop_rate = 0.0
        self.local_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=ws, with_rpe=True)
        self.global_block = CrossBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=1, with_rpe=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, x, tgt, size):
        x = self.local_block(x, size)
        tgt = self.local_block(tgt, size)
        x, tgt = self.global_block(x, tgt, size)
        return x, tgt


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = normalized_shape,

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, layer_scale_init_value=1e-06):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + x
        return x


class ConvNextLayer(nn.Module):

    def __init__(self, dim, depth=4):
        super().__init__()
        self.net = nn.Sequential(*[ConvNextBlock(dim=dim) for j in range(depth)])

    def forward(self, x):
        return self.net(x)

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num += np.prod(param.size())
        return num


class CrossAttentionLayer(nn.Module):

    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, dropout=0.0):
        super(CrossAttentionLayer, self).__init__()
        assert qk_dim % num_heads == 0, f'dim {qk_dim} should be divided by num_heads {num_heads}.'
        assert v_dim % num_heads == 0, f'dim {v_dim} should be divided by num_heads {num_heads}.'
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5
        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)
        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ffn = nn.Sequential(nn.Linear(query_token_dim, query_token_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(query_token_dim, query_token_dim), nn.Dropout(dropout))

    def forward(self, query, tgt_token):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = query
        query = self.norm1(query)
        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)
        x = self.multi_head_attn(q, k, v)
        x = short_cut + self.proj_drop(self.proj(x))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class MemoryDecoderLayer(nn.Module):

    def __init__(self, dim, cfg):
        super(MemoryDecoderLayer, self).__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        query_token_dim, tgt_token_dim = cfg.query_latent_dim, cfg.cost_latent_dim
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.cross_attend = CrossAttentionLayer(qk_dim, v_dim, query_token_dim, tgt_token_dim, add_flow_token=cfg.add_flow_token, dropout=cfg.dropout)

    def forward(self, query, key, value, memory, coords1, size, size_h3w3):
        """
            x:      [B*H1*W1, 1, C]
            memory: [B*H1*W1, H2'*W2', C]
            coords1 [B, 2, H2, W2]
            size: B, C, H1, W1
            1. Note that here coords0 and coords1 are in H2, W2 space.
               Should first convert it into H2', W2' space.
            2. We assume the upper-left point to be [0, 0], instead of letting center of upper-left patch to be [0, 0]
        """
        x_global, k, v = self.cross_attend(query, key, value, memory, coords1, self.patch_size, size_h3w3)
        B, C, H1, W1 = size
        C = self.cfg.query_latent_dim
        x_global = x_global.view(B, H1, W1, C).permute(0, 3, 1, 2)
        return x_global, k, v


class ReverseCostExtractor(nn.Module):

    def __init__(self, cfg):
        super(ReverseCostExtractor, self).__init__()
        self.cfg = cfg

    def forward(self, cost_maps, coords0, coords1):
        """
            cost_maps   -   B*H1*W1, cost_heads_num, H2, W2
            coords      -   B, 2, H1, W1
        """
        BH1W1, heads, H2, W2 = cost_maps.shape
        B, _, H1, W1 = coords1.shape
        assert H1 == H2 and W1 == W2
        assert BH1W1 == B * H1 * W1
        cost_maps = cost_maps.reshape(B, H1 * W1 * heads, H2, W2)
        coords = coords1.permute(0, 2, 3, 1)
        corr = bilinear_sampler(cost_maps, coords)
        corr = rearrange(corr, 'b (h1 w1 heads) h2 w2 -> (b h2 w2) heads h1 w1', b=B, heads=heads, h1=H1, w1=W1, h2=H2, w2=W2)
        r = 4
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        centroid = coords0.permute(0, 2, 3, 1).reshape(BH1W1, 1, 1, 2)
        delta = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords = centroid + delta
        corr = bilinear_sampler(corr, coords)
        corr = corr.view(B, H1, W1, -1).permute(0, 3, 1, 2)
        return corr


def initialize_flow(img):
    """ Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W)
    mean_init = coords_grid(N, H, W)
    return mean, mean_init


class MemoryDecoder(nn.Module):

    def __init__(self, cfg):
        super(MemoryDecoder, self).__init__()
        dim = self.dim = cfg.query_latent_dim
        self.cfg = cfg
        self.flow_token_encoder = nn.Sequential(nn.Conv2d(81 * cfg.cost_heads_num, dim, 1, 1), nn.GELU(), nn.Conv2d(dim, dim, 1, 1))
        self.proj = nn.Conv2d(256, 256, 1)
        self.depth = cfg.decoder_depth
        self.decoder_layer = MemoryDecoderLayer(dim, cfg)
        if self.cfg.gma:
            self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
            self.att = Attention(args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128)
        else:
            self.update_block = BasicUpdateBlock(self.cfg, hidden_dim=128)

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def encode_flow_token(self, cost_maps, coords):
        """
            cost_maps   -   B*H1*W1, cost_heads_num, H2, W2
            coords      -   B, 2, H1, W1
        """
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        r = 4
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        centroid = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords = centroid + delta
        corr = bilinear_sampler(cost_maps, coords)
        corr = corr.view(batch, h1, w1, -1).permute(0, 3, 1, 2)
        return corr

    def forward(self, cost_memory, context, data={}, flow_init=None):
        """
            memory: [B*H1*W1, H2'*W2', C]
            context: [B, D, H1, W1]
        """
        cost_maps = data['cost_maps']
        coords0, coords1 = initialize_flow(context)
        if flow_init is not None:
            coords1 = coords1 + flow_init
        flow_predictions = []
        context = self.proj(context)
        net, inp = torch.split(context, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        if self.cfg.gma:
            attention = self.att(inp)
        size = net.shape
        key, value = None, None
        for idx in range(self.depth):
            coords1 = coords1.detach()
            cost_forward = self.encode_flow_token(cost_maps, coords1)
            query = self.flow_token_encoder(cost_forward)
            query = query.permute(0, 2, 3, 1).contiguous().view(size[0] * size[2] * size[3], 1, self.dim)
            cost_global, key, value = self.decoder_layer(query, key, value, cost_memory, coords1, size, data['H3W3'])
            if self.cfg.only_global:
                corr = cost_global
            else:
                corr = torch.cat([cost_global, cost_forward], dim=1)
            flow = coords1 - coords0
            if self.cfg.gma:
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)
            else:
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)
        if self.training:
            return flow_predictions
        else:
            return flow_predictions[-1], coords1 - coords0


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class GroupVerticalSelfAttentionLayer(nn.Module):

    def __init__(self, dim, cfg, num_heads=8, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, dropout=0.0):
        super(GroupVerticalSelfAttentionLayer, self).__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        embed_dim = dim
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        dpr = 0.0
        drop_rate = dropout
        attn_drop_rate = 0.0
        self.block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=ws, with_rpe=True, vert_c_dim=cfg.vert_c_dim, groupattention=True, cfg=self.cfg)

    def forward(self, x, size, context=None):
        x = self.block(x, size, context)
        return x


class VerticalSelfAttentionLayer(nn.Module):

    def __init__(self, dim, cfg, num_heads=8, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, dropout=0.0):
        super(VerticalSelfAttentionLayer, self).__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        embed_dim = dim
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        dpr = 0.0
        drop_rate = dropout
        attn_drop_rate = 0.0
        self.local_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=ws, with_rpe=True, vert_c_dim=cfg.vert_c_dim)
        self.global_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=1, with_rpe=True, vert_c_dim=cfg.vert_c_dim)

    def forward(self, x, size, context=None):
        x = self.local_block(x, size, context)
        x = self.global_block(x, size, context)
        return x

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num += np.prod(param.size())
        return num


class SelfAttentionLayer(nn.Module):

    def __init__(self, dim, cfg, num_heads=8, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, dropout=0.0):
        super(SelfAttentionLayer, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multi_head_attn = MultiHeadAttention(dim, num_heads)
        self.q, self.k, self.v = nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = x
        x = self.norm1(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        x = self.multi_head_attn(q, k, v)
        x = self.proj(x)
        x = short_cut + self.proj_drop(x)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num += np.prod(param.size())
        return num


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(dense(dim, dim * expansion_factor), nn.GELU(), nn.Dropout(dropout), dense(dim * expansion_factor, dim), nn.Dropout(dropout))


class PreNormResidual(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class MLPMixerLayer(nn.Module):

    def __init__(self, dim, cfg, drop_path=0.0, dropout=0.0):
        super(MLPMixerLayer, self).__init__()
        K = cfg.cost_latent_token_num
        expansion_factor = cfg.mlp_expansion_factor
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.mlpmixer = nn.Sequential(PreNormResidual(dim, FeedForward(K, expansion_factor, dropout, chan_first)), PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last)))

    def compute_params(self):
        num = 0
        for param in self.mlpmixer.parameters():
            num += np.prod(param.size())
        return num

    def forward(self, x):
        """
            x: [BH1W1, K, D]
        """
        return self.mlpmixer(x)


class CostPerceiverEncoder(nn.Module):

    def __init__(self, cfg):
        super(CostPerceiverEncoder, self).__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.patch_embed = PatchEmbed(in_chans=self.cfg.cost_heads_num, patch_size=self.patch_size, embed_dim=cfg.cost_latent_input_dim, pe=cfg.pe)
        self.depth = cfg.encoder_depth
        self.latent_tokens = nn.Parameter(torch.randn(1, cfg.cost_latent_token_num, cfg.cost_latent_dim))
        query_token_dim, tgt_token_dim = cfg.cost_latent_dim, cfg.cost_latent_input_dim * 2
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.input_layer = CrossAttentionLayer(qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout)
        if cfg.use_mlp:
            self.encoder_layers = nn.ModuleList([MLPMixerLayer(cfg.cost_latent_dim, cfg, dropout=cfg.dropout) for idx in range(self.depth)])
        else:
            self.encoder_layers = nn.ModuleList([SelfAttentionLayer(cfg.cost_latent_dim, cfg, dropout=cfg.dropout) for idx in range(self.depth)])
        if self.cfg.vertical_conv:
            self.vertical_encoder_layers = nn.ModuleList([ConvNextLayer(cfg.cost_latent_dim) for idx in range(self.depth)])
        else:
            self.vertical_encoder_layers = nn.ModuleList([VerticalSelfAttentionLayer(cfg.cost_latent_dim, cfg, dropout=cfg.dropout) for idx in range(self.depth)])
        self.cost_scale_aug = None

    def forward(self, cost_volume, data, context=None):
        B, heads, H1, W1, H2, W2 = cost_volume.shape
        cost_maps = cost_volume.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * H1 * W1, self.cfg.cost_heads_num, H2, W2)
        data['cost_maps'] = cost_maps
        if self.cost_scale_aug is not None:
            scale_factor = torch.FloatTensor(B * H1 * W1, self.cfg.cost_heads_num, H2, W2).uniform_(self.cost_scale_aug[0], self.cost_scale_aug[1])
            cost_maps = cost_maps * scale_factor
        x, size = self.patch_embed(cost_maps)
        data['H3W3'] = size
        H3, W3 = size
        x = self.input_layer(self.latent_tokens, x)
        short_cut = x
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.cfg.vertical_conv:
                x = x.view(B, H1 * W1, self.cfg.cost_latent_token_num, -1).permute(0, 3, 1, 2).reshape(B * self.cfg.cost_latent_token_num, -1, H1, W1)
                x = self.vertical_encoder_layers[idx](x)
                x = x.view(B, self.cfg.cost_latent_token_num, -1, H1 * W1).permute(0, 2, 3, 1).reshape(B * H1 * W1, self.cfg.cost_latent_token_num, -1)
            else:
                x = x.view(B, H1 * W1, self.cfg.cost_latent_token_num, -1).permute(0, 2, 1, 3).reshape(B * self.cfg.cost_latent_token_num, H1 * W1, -1)
                x = self.vertical_encoder_layers[idx](x, (H1, W1), context)
                x = x.view(B, self.cfg.cost_latent_token_num, H1 * W1, -1).permute(0, 2, 1, 3).reshape(B * H1 * W1, self.cfg.cost_latent_token_num, -1)
        if self.cfg.cost_encoder_res is True:
            x = x + short_cut
        return x


class twins_svt_large(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large', pretrained=pretrained)
        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]

    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            if i == layer - 1:
                break
        return x

    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):
            for param in embed.parameters():
                num += np.prod(param.size())
            for param in drop.parameters():
                num += np.prod(param.size())
            for param in blocks.parameters():
                num += np.prod(param.size())
            for param in pos_blk.parameters():
                num += np.prod(param.size())
            if i == layer - 1:
                break
        for param in self.svt.head.parameters():
            num += np.prod(param.size())
        return num


class MemoryEncoder(nn.Module):

    def __init__(self, cfg):
        super(MemoryEncoder, self).__init__()
        self.cfg = cfg
        if cfg.fnet == 'twins':
            self.feat_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'basicencoder':
            self.feat_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        else:
            exit()
        self.channel_convertor = nn.Conv2d(cfg.encoder_latent_dim, cfg.encoder_latent_dim, 1, padding=0, bias=False)
        self.cost_perceiver_encoder = CostPerceiverEncoder(cfg)

    def corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = rearrange(fmap1, 'b (heads d) h w -> b heads (h w) d', heads=self.cfg.cost_heads_num)
        fmap2 = rearrange(fmap2, 'b (heads d) h w -> b heads (h w) d', heads=self.cfg.cost_heads_num)
        corr = einsum('bhid, bhjd -> bhij', fmap1, fmap2)
        corr = corr.permute(0, 2, 1, 3).view(batch * ht * wd, self.cfg.cost_heads_num, ht, wd)
        corr = corr.view(batch, ht * wd, self.cfg.cost_heads_num, ht * wd).permute(0, 2, 1, 3)
        corr = corr.view(batch, self.cfg.cost_heads_num, ht, wd, ht, wd)
        return corr

    def forward(self, img1, img2, data, context=None):
        feat_s = self.feat_encoder(img1)
        feat_t = self.feat_encoder(img2)
        feat_s = self.channel_convertor(feat_s)
        feat_t = self.channel_convertor(feat_t)
        B, C, H, W = feat_s.shape
        size = H, W
        if self.cfg.feat_cross_attn:
            feat_s = feat_s.flatten(2).transpose(1, 2)
            feat_t = feat_t.flatten(2).transpose(1, 2)
            for layer in self.layers:
                feat_s, feat_t = layer(feat_s, feat_t, size)
            feat_s = feat_s.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            feat_t = feat_t.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        cost_volume = self.corr(feat_s, feat_t)
        x = self.cost_perceiver_encoder(cost_volume, data, context)
        return x


class twins_svt_large_context(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large_context', pretrained=pretrained)

    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            if i == layer - 1:
                break
        return x


class CrossGlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, tgt, size: Size_):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr is not None:
            tgt = tgt.permute(0, 2, 1).reshape(B, C, *size)
            tgt = self.sr(tgt).reshape(B, C, -1).permute(0, 2, 1)
            tgt = self.norm(tgt)
        kv = self.kv(tgt).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PosConv(nn.Module):

    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim))
        self.stride = stride

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return [('proj.%d.weight' % i) for i in range(4)]


class Twins(nn.Module):
    """ Twins Vision Transfomer (Revisiting Spatial Attention)
    Adapted from PVT (PyramidVisionTransformer) class at https://github.com/whai362/PVT.git
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), wss=None, block_cls=Block, init_weight=True):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        img_size = to_2tuple(img_size)
        prev_chs = in_chans
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[k], ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.pos_block = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if init_weight:
            self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set([('pos_block.' + n) for n, p in self.pos_block.named_parameters()])

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class MultiScale(nn.Module):

    def __init__(self, startScale=4, numScales=5, l_weight=0.32, norm='L1'):
        super(MultiScale, self).__init__()
        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.l_type = norm
        self.div_flow = 0.05
        assert len(self.loss_weights) == self.numScales
        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()
        self.multiScales = [nn.AvgPool2d(self.startScale * 2 ** scale, self.startScale * 2 ** scale) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-' + self.l_type, 'EPE']

    def forward(self, output, target):
        output = output['flow_preds']
        target = target['flows'][:, 0]
        lossvalue = 0
        if type(output) is tuple or type(output) is list:
            target = self.div_flow * target
            for i, flow_pred_ in enumerate(output):
                target_ = self.multiScales[i](target)
                lossvalue += self.loss_weights[i] * self.loss(flow_pred_, target_)
        else:
            lossvalue += self.loss(output, target)
        return lossvalue


class tofp16(nn.Module):

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):

    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


class MultiScaleTridentConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, strides=1, paddings=0, dilations=1, dilation=1, groups=1, num_branch=1, test_branch_idx=-1, bias=False, norm=None, activation=None):
        super(MultiScaleTridentConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_branch = num_branch
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        self.dilation = dilation
        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        if isinstance(strides, int):
            strides = [strides] * self.num_branch
        self.paddings = [_pair(padding) for padding in paddings]
        self.dilations = [_pair(dilation) for dilation in dilations]
        self.strides = [_pair(stride) for stride in strides]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation
        assert len({self.num_branch, len(self.paddings), len(self.strides)}) == 1
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch
        if self.training or self.test_branch_idx == -1:
            outputs = [F.conv2d(input, self.weight, self.bias, stride, padding, self.dilation, self.groups) for input, stride, padding in zip(inputs, self.strides, self.paddings)]
        else:
            outputs = [F.conv2d(inputs[0], self.weight, self.bias, self.strides[self.test_branch_idx] if self.test_branch_idx == -1 else self.strides[-1], self.paddings[self.test_branch_idx] if self.test_branch_idx == -1 else self.paddings[-1], self.dilation, self.groups)]
        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]
        return outputs


class CNNEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d, num_output_scales=1, **kwargs):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales
        feature_dims = [64, 96, 128]
        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(feature_dims[2], stride=stride, norm_layer=norm_layer)
        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)
        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = 1, 2, 4, 8
            elif self.num_branch == 3:
                strides = 1, 2, 4
            elif self.num_branch == 2:
                strides = 1, 2
            else:
                raise ValueError
            self.trident_conv = MultiScaleTridentConv(output_dim, output_dim, kernel_size=3, strides=strides, paddings=1, num_branch=self.num_branch)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)
        else:
            out = [x]
        return out


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def single_head_full_attention(q, k, v):
    assert q.dim() == k.dim() == v.dim() == 3
    scores = torch.matmul(q, k.permute(0, 2, 1)) / q.size(2) ** 0.5
    attn = torch.softmax(scores, dim=2)
    out = torch.matmul(attn, v)
    return out


def merge_splits(splits, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(new_b, num_splits * h, num_splits * w, c)
    else:
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(new_b, c, num_splits * h, num_splits * w)
    return merge


def split_feature(feature, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)
    else:
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)
    return feature


def single_head_split_window_attention(q, k, v, num_splits=1, with_shift=False, h=None, w=None, attn_mask=None):
    assert q.dim() == k.dim() == v.dim() == 3
    assert h is not None and w is not None
    assert q.size(1) == h * w
    b, _, c = q.size()
    b_new = b * num_splits * num_splits
    window_size_h = h // num_splits
    window_size_w = w // num_splits
    q = q.view(b, h, w, c)
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)
    scale_factor = c ** 0.5
    if with_shift:
        assert attn_mask is not None
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2
        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
    q = split_feature(q, num_splits=num_splits, channel_last=True)
    k = split_feature(k, num_splits=num_splits, channel_last=True)
    v = split_feature(v, num_splits=num_splits, channel_last=True)
    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)) / scale_factor
    if with_shift:
        scores += attn_mask.repeat(b, 1, 1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.view(b_new, -1, c))
    out = merge_splits(out.view(b_new, h // num_splits, w // num_splits, c), num_splits=num_splits, channel_last=True)
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))
    out = out.view(b, -1, c)
    return out


class TransformerLayer(nn.Module):

    def __init__(self, d_model=256, nhead=1, attention_type='swin', no_ffn=False, ffn_dim_expansion=4, with_shift=False, **kwargs):
        super(TransformerLayer, self).__init__()
        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn
        self.with_shift = with_shift
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False), nn.GELU(), nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False))
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target, height=None, width=None, shifted_window_attn_mask=None, attn_num_splits=None, **kwargs):
        query, key, value = source, target, target
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        if self.attention_type == 'swin' and attn_num_splits > 1:
            if self.nhead > 1:
                raise NotImplementedError
            else:
                message = single_head_split_window_attention(query, key, value, num_splits=attn_num_splits, with_shift=self.with_shift, h=height, w=width, attn_mask=shifted_window_attn_mask)
        else:
            message = single_head_full_attention(query, key, value)
        message = self.merge(message)
        message = self.norm1(message)
        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)
        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self, d_model=256, nhead=1, attention_type='swin', ffn_dim_expansion=4, with_shift=False, **kwargs):
        super(TransformerBlock, self).__init__()
        self.self_attn = TransformerLayer(d_model=d_model, nhead=nhead, attention_type=attention_type, no_ffn=True, ffn_dim_expansion=ffn_dim_expansion, with_shift=with_shift)
        self.cross_attn_ffn = TransformerLayer(d_model=d_model, nhead=nhead, attention_type=attention_type, ffn_dim_expansion=ffn_dim_expansion, with_shift=with_shift)

    def forward(self, source, target, height=None, width=None, shifted_window_attn_mask=None, attn_num_splits=None, **kwargs):
        source = self.self_attn(source, source, height=height, width=width, shifted_window_attn_mask=shifted_window_attn_mask, attn_num_splits=attn_num_splits)
        source = self.cross_attn_ffn(source, target, height=height, width=width, shifted_window_attn_mask=shifted_window_attn_mask, attn_num_splits=attn_num_splits)
        return source


def generate_shift_window_attn_mask(input_resolution, window_size_h, window_size_w, shift_size_h, shift_size_w, device=torch.device('cuda')):
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1))
    h_slices = slice(0, -window_size_h), slice(-window_size_h, -shift_size_h), slice(-shift_size_h, None)
    w_slices = slice(0, -window_size_w), slice(-window_size_w, -shift_size_w), slice(-shift_size_w, None)
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = split_feature(img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True)
    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class FeatureTransformer(nn.Module):

    def __init__(self, num_layers=6, d_model=128, nhead=1, attention_type='swin', ffn_dim_expansion=4, **kwargs):
        super(FeatureTransformer, self).__init__()
        self.attention_type = attention_type
        self.d_model = d_model
        self.nhead = nhead
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, nhead=nhead, attention_type=attention_type, ffn_dim_expansion=ffn_dim_expansion, with_shift=True if attention_type == 'swin' and i % 2 == 1 else False) for i in range(num_layers)])
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1, attn_num_splits=None, **kwargs):
        b, c, h, w = feature0.shape
        assert self.d_model == c
        feature0 = feature0.flatten(-2).permute(0, 2, 1)
        feature1 = feature1.flatten(-2).permute(0, 2, 1)
        if self.attention_type == 'swin' and attn_num_splits > 1:
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits
            shifted_window_attn_mask = generate_shift_window_attn_mask(input_resolution=(h, w), window_size_h=window_size_h, window_size_w=window_size_w, shift_size_h=window_size_h // 2, shift_size_w=window_size_w // 2, device=feature0.device)
        else:
            shifted_window_attn_mask = None
        concat0 = torch.cat((feature0, feature1), dim=0)
        concat1 = torch.cat((feature1, feature0), dim=0)
        for layer in self.layers:
            concat0 = layer(concat0, concat1, height=h, width=w, shifted_window_attn_mask=shifted_window_attn_mask, attn_num_splits=attn_num_splits)
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)
        feature0, feature1 = concat0.chunk(chunks=2, dim=0)
        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return feature0, feature1


class FeatureFlowAttention(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels, **kwargs):
        super(FeatureFlowAttention, self).__init__()
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow, local_window_attn=False, local_window_radius=1, **kwargs):
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow, local_window_radius=local_window_radius)
        b, c, h, w = feature0.size()
        query = feature0.view(b, c, h * w).permute(0, 2, 1)
        query = self.q_proj(query)
        key = self.k_proj(query)
        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)
        scores = torch.matmul(query, key.permute(0, 2, 1)) / c ** 0.5
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, value)
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)
        return out

    def forward_local_window_attn(self, feature0, flow, local_window_radius=1):
        assert flow.size(1) == 2
        assert local_window_radius > 0
        b, c, h, w = feature0.size()
        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)).reshape(b * h * w, 1, c)
        kernel_size = 2 * local_window_radius + 1
        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)
        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size, padding=local_window_radius)
        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)
        flow_window = F.unfold(flow, kernel_size=kernel_size, padding=local_window_radius)
        flow_window = flow_window.view(b, 2, kernel_size ** 2, h, w).permute(0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, 2)
        scores = torch.matmul(feature0_reshape, feature0_window) / c ** 0.5
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, flow_window).view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()
        return out


class IPTHeadEncoder(nn.Module):
    """docstring for IPTHead"""

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(IPTHeadEncoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        half_out_dim = max(output_dim // 2, 64)
        self.layer1 = ResidualBlock(64, half_out_dim, self.norm_fn, stride=2)
        self.layer2 = ResidualBlock(half_out_dim, output_dim, self.norm_fn, stride=2)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class BasicConvEncoder(nn.Module):
    """docstring for BasicConvEncoder"""

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicConvEncoder, self).__init__()
        self.norm_fn = norm_fn
        half_out_dim = max(output_dim // 2, 64)
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm3 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
            self.norm2 = nn.BatchNorm2d(half_out_dim)
            self.norm3 = nn.BatchNorm2d(output_dim)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
            self.norm2 = nn.InstanceNorm2d(half_out_dim)
            self.norm3 = nn.InstanceNorm2d(output_dim)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, half_out_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(half_out_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        x = F.relu(self.norm3(self.conv3(x)), inplace=True)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class AnchorEncoderBlock(nn.Module):

    def __init__(self, anchor_dist, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.anchor_dist = anchor_dist
        self.half_anchor_dist = anchor_dist // 2
        self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.FFN = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """ 
        inputs: batches with N*C*H*W
    """
        N, C, H, W = inputs.shape
        x = inputs
        anchors = inputs[:, :, self.half_anchor_dist::self.anchor_dist, self.half_anchor_dist::self.anchor_dist].clone()
        x = x.reshape(N, C, H * W).transpose(-1, -2)
        anchors = anchors.reshape(N, C, anchors.shape[2] * anchors.shape[3]).transpose(-1, -2)
        anchors_new = self.dropout(self.selfAttn(anchors, x, x)[0])
        residual = self.dropout(self.selfAttn(x, anchors_new, anchors_new)[0])
        norm_1 = self.layer_norm_1(x + residual)
        x_linear = self.dropout(self.FFN(norm_1))
        x_new = self.layer_norm_2(norm_1 + x_linear)
        outputs = x_new.transpose(-1, -2).reshape(N, C, H, W)
        return outputs


class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.FFN = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """ 
        x: input batches with N*C*H*W
    """
        N, C, H, W = x.shape
        x = x.reshape(N, C, H * W).transpose(-1, -2)
        residual = self.dropout(self.selfAttn(x, x, x)[0])
        norm_1 = self.layer_norm_1(x + residual)
        x_linear = self.dropout(self.FFN(norm_1))
        x_new = self.layer_norm_2(norm_1 + x_linear)
        outputs = x_new.transpose(-1, -2).reshape(N, C, H, W)
        return outputs


class ReduceEncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(d_model, d_model, 2, 2), nn.Conv2d(d_model, d_model, 2, 2))
        self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.FFN = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """ 
        x: input batches with N*C*H*W
    """
        N, C, H, W = x.shape
        x_reduced = self.reduce(x)
        x = x.reshape(N, C, H * W).transpose(-1, -2)
        x_reduced = x_reduced.reshape(N, C, -1).transpose(-1, -2)
        residual = self.dropout(self.selfAttn(x, x_reduced, x_reduced)[0])
        norm_1 = self.layer_norm_1(x + residual)
        x_linear = self.dropout(self.FFN(norm_1))
        x_new = self.layer_norm_2(norm_1 + x_linear)
        outputs = x_new.transpose(-1, -2).reshape(N, C, H, W)
        return outputs


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LayerEncoderBlock(nn.Module):

    def __init__(self, win_size, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.win_size = win_size
        self.down_factor = 4
        self.unfold_stride = int(self.win_size // self.down_factor)
        self.stride_list = [math.floor(win_size / self.down_factor ** idx) for idx in range(8) if win_size / self.down_factor ** idx >= 1]
        self.reduce = nn.Sequential(nn.AvgPool2d(self.down_factor, self.down_factor))
        self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.crossAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layerNormSelf = nn.LayerNorm(d_model)
        self.layerNormCross = nn.LayerNorm(d_model)
        self.FFN = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.layer_norm_out = nn.LayerNorm(d_model)

    def Circular_pad2D(self, x, pad_right, pad_bottom):
        """
            x: (N, H, W, C)
            x_pad: (N, H_pad, W_pad, C)
        """
        N, H, W, C = x.shape
        H_pad = H + pad_bottom
        W_pad = W + pad_right
        H_repeat = math.ceil(H_pad / H)
        W_repeat = math.ceil(W_pad / W)
        x_repeat = x.repeat(1, H_repeat, W_repeat, 1)
        x_pad = x_repeat[:, :H_pad, :W_pad, :]
        return x_pad

    def pad_fit_win(self, x, win_size):
        N, H, W, C = x.shape
        W_ = math.ceil(W / win_size) * win_size
        H_ = math.ceil(H / win_size) * win_size
        padRight = W_ - W
        padBottom = H_ - H
        x_pad = self.Circular_pad2D(x, padRight, padBottom)
        return x_pad

    def self_attention(self, x):
        """
            x: (N, H, W, C)
            out: (N, H, W, C)
        """
        N, H, W, C = x.shape
        x_pad = self.pad_fit_win(x, self.win_size)
        _, H_, W_, _ = x_pad.shape
        x_window = window_partition(x_pad, self.win_size)
        x_window = x_window.view(-1, self.win_size * self.win_size, C)
        residual = self.dropout(self.selfAttn(x_window, x_window, x_window)[0])
        residual = residual.view(-1, self.win_size, self.win_size, C)
        residual = window_reverse(residual, self.win_size, H_, W_)
        out = x_pad + residual
        out = out[:, :H, :W, :]
        return out

    def cross_attention(self, query, keyVal):
        """
            query: (N, qH, qW, C)
            keyVal: (N, kH, kW, C)
            out: (N, qH, qW, C)
        """
        _, qH, qW, C = query.shape
        _, kH, kW, C = keyVal.shape
        query = self.pad_fit_win(query, self.win_size)
        _, qH_, qW_, C = query.shape
        query_win = window_partition(query, self.win_size)
        query_win = query_win.view(-1, self.win_size * self.win_size, C)
        kW_ = (math.ceil(kW / self.unfold_stride) - 1) * self.unfold_stride + self.win_size
        kH_ = (math.ceil(kH / self.unfold_stride) - 1) * self.unfold_stride + self.win_size
        padRight = kW_ - kW
        padBottom = kH_ - kH
        keyVal_pad = self.Circular_pad2D(keyVal, padRight, padBottom)
        keyVal = F.unfold(keyVal_pad.permute(0, 3, 1, 2), self.win_size, stride=self.unfold_stride)
        keyVal = keyVal.permute(0, 2, 1).reshape(-1, C, self.win_size * self.win_size).permute(0, 2, 1)
        residual = self.dropout(self.crossAttn(query_win, keyVal, keyVal)[0])
        residual = residual.view(-1, self.win_size, self.win_size, C)
        residual = window_reverse(residual, self.win_size, qH_, qW_)
        out = query + residual
        out = out[:, :qH, :qW, :]
        return out

    def forward(self, x):
        """ 
            x: input batches with N*C*H*W
        """
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.pad_fit_win(x, self.win_size)
        layerAttnList = []
        strideListLen = len(self.stride_list)
        for idx in range(strideListLen):
            x_attn = self.self_attention(x)
            x_attn = self.layerNormSelf(x_attn)
            layerAttnList.append(x_attn)
            if idx < strideListLen - 1:
                x = self.reduce(x_attn.permute(0, 3, 1, 2))
                x = x.permute(0, 2, 3, 1)
        KeyVal = layerAttnList[-1]
        for idx in range(strideListLen - 1, 0, -1):
            Query = layerAttnList[idx - 1]
            Query = self.cross_attention(Query, KeyVal)
            Query = self.layerNormCross(Query)
            KeyVal = Query
        Query = Query[:, :H, :W, :]
        q_residual = self.dropout(self.FFN(Query))
        x_new = self.layer_norm_out(Query + q_residual)
        outputs = x_new.permute(0, 3, 1, 2)
        return outputs


class BasicLayerEncoderBlock(nn.Module):

    def __init__(self, win_size, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.win_size = win_size
        self.down_factor = 2
        self.unfold_stride = int(self.win_size // self.down_factor)
        self.stride_list = [math.floor(win_size / self.down_factor ** idx) for idx in range(8) if win_size / self.down_factor ** idx >= 1]
        self.reduce = nn.Sequential(nn.AvgPool2d(self.down_factor, self.down_factor))
        self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.crossAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layerNormSelf = nn.LayerNorm(d_model)
        self.layerNormCross = nn.LayerNorm(d_model)
        self.FFN = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.layer_norm_out = nn.LayerNorm(d_model)

    def Circular_pad2D(self, x, pad_right, pad_bottom):
        """
            x: (N, H, W, C)
            x_pad: (N, H_pad, W_pad, C)
        """
        N, H, W, C = x.shape
        H_pad = H + pad_bottom
        W_pad = W + pad_right
        H_repeat = math.ceil(H_pad / H)
        W_repeat = math.ceil(W_pad / W)
        x_repeat = x.repeat(1, H_repeat, W_repeat, 1)
        x_pad = x_repeat[:, :H_pad, :W_pad, :]
        return x_pad

    def pad_fit_win(self, x, win_size):
        N, H, W, C = x.shape
        W_ = math.ceil(W / win_size) * win_size
        H_ = math.ceil(H / win_size) * win_size
        padRight = W_ - W
        padBottom = H_ - H
        x_pad = self.Circular_pad2D(x, padRight, padBottom)
        return x_pad

    def self_attention(self, x):
        """
            x: (N, H, W, C)
            out: (N, H, W, C)
        """
        N, H, W, C = x.shape
        x_pad = self.pad_fit_win(x, self.win_size)
        _, H_, W_, _ = x_pad.shape
        x_window = window_partition(x_pad, self.win_size)
        x_window = x_window.view(-1, self.win_size * self.win_size, C)
        residual = self.dropout(self.selfAttn(x_window, x_window, x_window)[0])
        residual = residual.view(-1, self.win_size, self.win_size, C)
        residual = window_reverse(residual, self.win_size, H_, W_)
        out = x_pad + residual
        out = out[:, :H, :W, :]
        return out

    def cross_attention(self, query, keyVal, query_win_size):
        """
            query: (N, qH, qW, C)
            keyVal: (N, kH, kW, C)
            out: (N, qH, qW, C)
        """
        _, qH, qW, C = query.shape
        query_win = window_partition(query, query_win_size)
        query_win = query_win.view(-1, query_win_size * query_win_size, C)
        keyWinSize = query_win_size // 2
        keyVal_win = window_partition(keyVal, keyWinSize)
        keyVal_win = keyVal_win.view(-1, keyWinSize * keyWinSize, C)
        residual = self.dropout(self.crossAttn(query_win, keyVal_win, keyVal_win)[0])
        residual = residual.view(-1, query_win_size, query_win_size, C)
        residual = window_reverse(residual, query_win_size, qH, qW)
        out = query + residual
        return out

    def forward(self, x):
        """ 
            x: input batches with N*C*H*W
        """
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.pad_fit_win(x, self.win_size)
        layerAttnList = []
        strideListLen = len(self.stride_list)
        for idx in range(strideListLen):
            x_attn = self.self_attention(x)
            x_attn = self.layerNormSelf(x_attn)
            layerAttnList.append(x_attn)
            if idx < strideListLen - 1:
                x = self.reduce(x_attn.permute(0, 3, 1, 2))
                x = x.permute(0, 2, 3, 1)
        KeyVal = layerAttnList[-1]
        for idx in range(strideListLen - 1, 0, -1):
            Query = layerAttnList[idx - 1]
            QueryWinSize = self.stride_list[idx - 1]
            Query = self.cross_attention(Query, KeyVal, QueryWinSize)
            Query = self.layerNormCross(Query)
            KeyVal = Query
        Query = Query[:, :H, :W, :]
        q_residual = self.dropout(self.FFN(Query))
        x_new = self.layer_norm_out(Query + q_residual)
        outputs = x_new.permute(0, 3, 1, 2)
        return outputs


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.max_len = 256
        self.d_model = d_model
        self._update_PE_table(self.max_len, self.d_model // 2)

    def _update_PE_table(self, max_len, d_model):
        self.PE_table = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        self.PE_table[:, 0::2] = torch.sin(pos / denominator)
        self.PE_table[:, 1::2] = torch.cos(pos / denominator)

    def forward(self, x):
        """ x: image batches with N*C*H*W """
        N, C, H, W = x.shape
        max_hw = max(H, W)
        if max_hw > self.max_len or self.d_model != C:
            self.max_len = max_hw
            self.d_model = C
            self._update_PE_table(self.max_len, self.d_model // 2)
        if self.PE_table.device != x.device:
            self.PE_table = self.PE_table
        h_pos_emb = self.PE_table[:H, :].unsqueeze(1).repeat(1, W, 1)
        w_pos_emb = self.PE_table[:W, :].unsqueeze(0).repeat(H, 1, 1)
        pos_emb = torch.cat([h_pos_emb, w_pos_emb], dim=-1).permute([2, 0, 1]).unsqueeze(0).repeat(N, 1, 1, 1)
        output = x + pos_emb
        return output


class TransformerEncoder(nn.Module):

    def __init__(self, anchor_dist, num_blocks, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.anchor_dist = anchor_dist
        blocks_list = []
        for idx in range(num_blocks):
            blocks_list.append(ReduceEncoderBlock(d_model, num_heads, d_ff, dropout))
        self.blocks = nn.Sequential(*blocks_list)
        self.posEmbedding = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        x_w_pos = self.posEmbedding(x)
        x_updated = self.blocks(x_w_pos)
        return x_updated


class RawInputTransEncoder(nn.Module):

    def __init__(self, anchor_dist, num_blocks, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.anchor_dist = anchor_dist
        self.linear = nn.Conv2d(3, d_model, 8, 8)
        blocks_list = []
        for idx in range(num_blocks):
            blocks_list.append(LayerEncoderBlock(anchor_dist, d_model, num_heads, d_ff, dropout))
        self.blocks = nn.Sequential(*blocks_list)
        self.posEmbedding = PositionalEncoding(d_model, dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.linear(x)
        x_w_pos = self.posEmbedding(x)
        x_updated = self.blocks(x_w_pos)
        if is_list:
            x_updated = torch.split(x_updated, [batch_dim, batch_dim], dim=0)
        return x_updated


class GlobalLocalBlock(nn.Module):

    def __init__(self, anchor_dist, d_model, num_heads, out_dim, dropout=0.0, stride=1):
        super().__init__()
        self.anchor_dist = anchor_dist
        self.half_anchor_dist = anchor_dist // 2
        self.d_model = d_model
        self.out_dim = out_dim
        self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.resBlock_1 = ResidualBlock(d_model, d_model, norm_fn='instance', stride=stride)
        self.change_channel = nn.Linear(d_model, out_dim)
        self.resBlock_2 = ResidualBlock(out_dim, out_dim, norm_fn='instance', stride=1)
        self.posEmbedding = PositionalEncoding(d_model, dropout)

    def forward(self, inputs):
        """ 
        inputs: batches with N*H*W*C
    """
        x = self.resBlock_1(inputs)
        x = self.posEmbedding(x)
        anchors = x[:, :, self.half_anchor_dist::self.anchor_dist, self.half_anchor_dist::self.anchor_dist].clone()
        N, C, H, W = x.shape
        x = x.reshape(N, C, H * W).transpose(-1, -2)
        anchors = anchors.reshape(N, C, anchors.shape[2] * anchors.shape[3]).transpose(-1, -2)
        anchors_new = self.dropout(self.selfAttn(anchors, x, x)[0])
        residual = self.dropout(self.selfAttn(x, anchors_new, anchors_new)[0])
        norm_1 = self.layer_norm_1(x + residual)
        norm_1 = self.change_channel(norm_1)
        norm_1 = norm_1.transpose(-1, -2).reshape(N, self.out_dim, H, W)
        outputs = self.resBlock_2(norm_1)
        return outputs


class GlobalLocalEncoder(nn.Module):

    def __init__(self, anchor_dist, output_dim, dropout=0.0):
        super().__init__()
        self.anchor_dist = anchor_dist
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = GlobalLocalBlock(self.anchor_dist, 64, 2, 96, dropout, stride=2)
        self.layer2 = GlobalLocalBlock(self.anchor_dist, 96, 3, 96, dropout, stride=1)
        self.layer3 = GlobalLocalBlock(self.anchor_dist // 2, 96, 4, 128, dropout, stride=2)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
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

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, use_shift_win=True, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=(0 if i % 2 == 0 else window_size // 2) if use_shift_win else 0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class SwinTransEncoder(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, use_shift_win=True, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, use_shift_win=use_shift_win, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self.apply(self._init_weights)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        x_out = outs[-1]
        if is_list:
            x_out = torch.split(x_out, [batch_dim, batch_dim], dim=0)
        return x_out

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransEncoder, self).train(mode)
        self._freeze_stages()


class NeighborWindowAttention(nn.Module):
    """ Patch-based OverLapping multi-head self-Attention (POLA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window (or patch).
        num_heads (int): Number of attention heads.
        neig_win_num (int): Number of neighbor windows. Default: 1
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, neig_win_num=1, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_proj=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_proj = use_proj
        self.n_win = 2 * neig_win_num + 1
        self.relative_position_bias_table = nn.Parameter(torch.zeros(((self.n_win + 1) * window_size[0] - 1) * ((self.n_win + 1) * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_h_neig = torch.arange(self.n_win * self.window_size[0])
        coords_w_neig = torch.arange(self.n_win * self.window_size[1])
        coords_neig = torch.stack(torch.meshgrid([coords_h_neig, coords_w_neig]))
        coords_flat = torch.flatten(coords, 1)
        coords_neig_flat = torch.flatten(coords_neig, 1)
        relative_coords = coords_flat[:, :, None] - coords_neig_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.n_win * self.window_size[0] - 1
        relative_coords[:, :, 1] += self.n_win * self.window_size[1] - 1
        relative_coords[:, :, 0] *= (self.n_win + 1) * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if self.use_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """ Forward function.
        Args:
            q: input queries with shape of (num_windows*B, N, C)
            k: input keys with shape of (num_windows*B, N, C)
            v: input values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N_q, C = q.shape
        N_kv = k.shape[1]
        dim_per_head = C // self.num_heads
        q = self.Wq(q).reshape(B_, N_q, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        k = self.Wk(k).reshape(B_, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        v = self.Wv(v).reshape(B_, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.n_win * self.window_size[0] * self.n_win * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_q, N_kv) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_q, N_kv)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)
        if self.use_proj:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class POLATransBlock(nn.Module):
    """ Transformer block with POLA.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window/patch size.
        neig_win_num (int): Number of overlapped windows
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, neig_win_num=1, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.neig_win_num = neig_win_num
        self.mlp_ratio = mlp_ratio
        self.n_win = 2 * neig_win_num + 1
        self.norm1 = norm_layer(dim)
        self.attn = NeighborWindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, neig_win_num=neig_win_num, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, attn_mask=None):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        x_win = window_partition(x, self.window_size)
        x_win = x_win.view(-1, self.window_size * self.window_size, C)
        pad_size = self.neig_win_num * self.window_size
        key_val = F.pad(x, (0, 0, pad_size, pad_size, pad_size, pad_size))
        key_val = F.unfold(key_val.permute(0, 3, 1, 2), self.n_win * self.window_size, stride=self.window_size)
        key_val = key_val.permute(0, 2, 1).reshape(-1, C, (self.n_win * self.window_size) ** 2).permute(0, 2, 1)
        attn_windows = self.attn(x_win, key_val, key_val, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixAxialPOLABlock(nn.Module):
    """ Transformer block with mixture of POLA, vertical and horizontal axis self-attentions
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads=8, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dim_per_head = dim // self.num_heads
        self.axis_head = 2
        self.local_head = self.num_heads - 2 * self.axis_head
        self.local_chl = self.local_head * self.dim_per_head
        self.axis_chl = self.axis_head * self.dim_per_head
        self.neig_win_num = 1
        self.n_win = 2 * self.neig_win_num + 1
        self.norm1 = norm_layer(dim)
        self.localAttn = NeighborWindowAttention(self.local_chl, window_size=to_2tuple(self.window_size), num_heads=self.local_head, neig_win_num=self.neig_win_num, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.vertiAttn = MultiHeadAttention(self.axis_chl, num_heads=self.axis_head, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_proj=False)
        self.horizAttn = MultiHeadAttention(self.axis_chl, num_heads=self.axis_head, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_proj=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, attn_mask=None):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x_local, x_horiz, x_verti = torch.split(x, [self.local_chl, self.axis_chl, self.axis_chl], dim=-1)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x_local = F.pad(x_local, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x_local.shape
        x_windows = window_partition(x_local, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, self.local_chl)
        pad_size = self.neig_win_num * self.window_size
        key_val = F.pad(x_local, (0, 0, pad_size, pad_size, pad_size, pad_size))
        key_val = F.unfold(key_val.permute(0, 3, 1, 2), self.n_win * self.window_size, stride=self.window_size)
        key_val = key_val.permute(0, 2, 1).reshape(-1, self.local_chl, (self.n_win * self.window_size) ** 2).permute(0, 2, 1)
        attn_windows = self.localAttn(x_windows, key_val, key_val, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.local_chl)
        x_local = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x_local = x_local[:, :H, :W, :].contiguous()
        x_horiz = x_horiz.view(-1, W, self.axis_chl)
        x_horiz = self.horizAttn(x_horiz, x_horiz, x_horiz)
        x_horiz = x_horiz.view(B, H, W, self.axis_chl)
        x_verti = x_verti.transpose(1, 2).reshape(-1, H, self.axis_chl)
        x_verti = self.vertiAttn(x_verti, x_verti, x_verti)
        x_verti = x_verti.view(B, W, H, self.axis_chl).transpose(1, 2)
        x = torch.cat([x_local, x_horiz, x_verti], dim=-1)
        x = x.view(B, H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicSwinUpdate(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depth (int): number of Swin Transformer blocks.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, embed_dim=96, depth=6, num_head=3, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=True, use_shift_win=True, use_checkpoint=False):
        super().__init__()
        self.num_feature = embed_dim
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.use_checkpoint = use_checkpoint
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=self.num_feature, num_heads=num_head, window_size=self.window_size, shift_size=(0 if i % 2 == 0 else self.window_size // 2) if use_shift_win else 0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(self.num_feature)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        x = self.norm(x)
        out = x.view(-1, H, W, self.num_feature).permute(0, 3, 1, 2).contiguous()
        if is_list:
            out = torch.split(out, [batch_dim, batch_dim], dim=0)
        return out


class POLAUpdate(nn.Module):
    """ POLA update for GMFlowNet.
        A PyTorch impl of : `Global Matching with Overlapping Attention for Optical Flow Estimation`  -
          https://arxiv.org/abs/2203.11335
    Args:
        embed_dim (int): Number of linear projection output channels. Default: 256.
        depths (int): Number of POLA blocks.
        num_heads (int): Number of attention head in each POLA block.
        window_size (int): Window/patch size. Default: 7.
        neig_win_num: Number of overlapped Windows/patches
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False):
        super().__init__()
        self.num_feature = embed_dim
        self.num_head = num_head
        self.win_size = window_size
        self.neig_win_num = neig_win_num
        self.use_checkpoint = use_checkpoint
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([POLATransBlock(dim=self.num_feature, num_heads=self.num_head, window_size=self.win_size, neig_win_num=self.neig_win_num, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(self.num_feature)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        img_mask = torch.zeros((1, H, W, 1), device=x.device)
        pad_r = (self.win_size - W % self.win_size) % self.win_size
        pad_b = (self.win_size - H % self.win_size) % self.win_size
        pad_extra = self.neig_win_num * self.win_size
        img_mask = F.pad(img_mask, (0, 0, pad_extra, pad_r + pad_extra, pad_extra, pad_b + pad_extra), mode='constant', value=float(-100.0))
        n_win = 2 * self.neig_win_num + 1
        mask_windows = F.unfold(img_mask.permute(0, 3, 1, 2), n_win * self.win_size, stride=self.win_size)
        mask_windows = mask_windows.permute(0, 2, 1).reshape(-1, (n_win * self.win_size) ** 2)
        attn_mask = mask_windows.unsqueeze(1).repeat(1, self.win_size * self.win_size, 1)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W, attn_mask)
            else:
                x = blk(x, H, W, attn_mask)
        x = self.norm(x)
        out = x.view(-1, H, W, self.num_feature).permute(0, 3, 1, 2).contiguous()
        if is_list:
            out = torch.split(out, [batch_dim, batch_dim], dim=0)
        return out


class MixSelfAttnUpdate(nn.Module):
    """ MixSelfAttnUpdate
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False):
        super().__init__()
        self.num_feature = embed_dim
        self.num_head = num_head
        self.win_size = window_size
        self.use_checkpoint = use_checkpoint
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([MixAxialPOLABlock(dim=self.num_feature, num_heads=self.num_head, window_size=self.win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(self.num_feature)
        self.apply(self._init_weights)
        self.x_list = list()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
                self.x_list.append(x.view(-1, H, W, self.num_feature).permute(0, 3, 1, 2).contiguous())
        x = self.norm(x)
        out = x.view(-1, H, W, self.num_feature).permute(0, 3, 1, 2).contiguous()
        if is_list:
            out = torch.split(out, [batch_dim, batch_dim], dim=0)
        return out


class MixAxialPOLAUpdate(nn.Module):
    """ Mixture attention (POLA and axial attentions) update for GMFlowNet.
        A PyTorch impl of : `Global Matching with Overlapping Attention for Optical Flow Estimation`  -
          https://arxiv.org/abs/2203.11335
    Args:
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depth (tuple[int]): Number of mix attention blocks.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        neig_win_num (int): Number of overlapped windows for POLA
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False):
        super().__init__()
        self.num_feature = embed_dim
        self.num_head = num_head
        self.win_size = window_size
        self.use_checkpoint = use_checkpoint
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([MixAxialPOLABlock(dim=self.num_feature, num_heads=self.num_head, window_size=self.win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(self.num_feature)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        x = self.norm(x)
        out = x.view(-1, H, W, self.num_feature).permute(0, 3, 1, 2).contiguous()
        if is_list:
            out = torch.split(out, [batch_dim, batch_dim], dim=0)
        return out


def drop_block_2d(x, drop_prob: float=0.1, block_size: int=7, gamma_scale: float=1.0, with_noise: bool=False, inplace: bool=False, batchwise: bool=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    w_i, h_i = torch.meshgrid(torch.arange(W), torch.arange(H))
    valid_block = (w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2) & ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W))
    if batchwise:
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = 2 - gamma - valid_block + uniform_noise >= 1
    block_mask = -F.max_pool2d(-block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: torch.Tensor, drop_prob: float=0.1, block_size: int=7, gamma_scale: float=1.0, with_noise: bool=False, inplace: bool=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    block_mask = torch.empty_like(x).bernoulli_(gamma)
    block_mask = F.max_pool2d(block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.empty_like(x).normal_()
        if inplace:
            x.mul_(1.0 - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-06)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(self, drop_prob: float=0.1, block_size: int=7, gamma_scale: float=1.0, with_noise: bool=False, inplace: bool=False, batchwise: bool=False, fast: bool=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace)
        else:
            return drop_block_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, normalize=True):
        super(PreActBlock, self).__init__()
        if normalize:
            self.bn1 = BatchNorm(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.relu(self.bn1(x)) if hasattr(self, 'bn1') else x
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out


class ResnetDecoder(nn.Module):

    def __init__(self, inplane, outplane):
        super(ResnetDecoder, self).__init__()
        self.block1 = PreActBlock(inplane, outplane, normalize=False)
        self.block2 = PreActBlock(outplane, outplane, normalize=True)

    def forward(self, x):
        x = self.block1(x)
        out = self.block2(x)
        return out


class HDADecoder(nn.Module):

    def __init__(self, inplane, outplane):
        super(HDADecoder, self).__init__()
        self.block1 = PreActBlock(inplane, outplane, normalize=False)
        self.block2 = PreActBlock(outplane, outplane, normalize=True)
        self.root = nn.Sequential(BatchNorm(outplane * 2), nn.ReLU(inplace=True), nn.Conv2d(outplane * 2, outplane, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(y1)
        out = self.root(torch.cat([y1, y2], 1))
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.bn1 = BatchNorm(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn2 = BatchNorm(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn3 = BatchNorm(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class Root(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):

    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1, dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), BatchNorm(out_channels))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):

    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, return_levels=False, pool_size=7, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False), BatchNorm(channels[0]), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)
        self.level6 = Tree(levels[6], block, channels[5], channels[6], 2, level_root=True, root_residual=residual_root)
        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation), BatchNorm(planes), nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(7):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):

    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(nn.Conv2d(c, out_dim, kernel_size=1, stride=1, bias=False), BatchNorm(out_dim), nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(out_dim, out_dim, f * 2, stride=f, padding=f // 2, output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
        for i in range(1, len(channels)):
            node = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, kernel_size=node_kernel, stride=1, padding=node_kernel // 2, bias=False), BatchNorm(out_dim), nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)
        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):

    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(3, channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        ms_feat = [layers[-1]]
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
            ms_feat.append(x)
        return ms_feat


class DLAUpEncoder(nn.Module):

    def __init__(self, planes):
        super(DLAUpEncoder, self).__init__()
        self.first_level = 1
        self.base = dla.dla34(planes)
        scales = [(2 ** i) for i in range(len(planes[self.first_level:]))]
        self.dla_up = DLAUp(planes[self.first_level:], scales=scales)

    def forward(self, x):
        x = self.base(x)
        y = self.dla_up(x[self.first_level:])
        return y[::-1]


class Context(nn.Module):

    def __init__(self, inplane, classes):
        super(Context, self).__init__()
        self.num_convs = 7
        ch = [inplane, 128, 128, 128, 128, 128, 128, 128]
        dilations = [1, 1, 2, 4, 8, 16, 1]
        for i in range(self.num_convs):
            setattr(self, 'dc_conv_{}'.format(i), nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=dilations[i], dilation=dilations[i], bias=False), BatchNorm(ch[i + 1]), nn.ReLU(inplace=True)))
        self.cls = nn.Conv2d(ch[-1], classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x
        for i in range(self.num_convs):
            dc_conv = getattr(self, 'dc_conv_' + str(i))
            out = dc_conv(out)
        out = self.cls(out)
        return out, None


class VGG(nn.Module):

    def __init__(self, block, planes):
        super(VGG, self).__init__()
        self.levels = len(planes)
        channels = [3] + planes
        for i in range(self.levels):
            setattr(self, 'block_{}'.format(i), block(channels[i], channels[i + 1]))
        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = []
        for i in range(self.levels):
            x = getattr(self, 'block_' + str(i))(x)
            out.append(x)
        return out


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.1, inplace=True))


def upsample_factor2(inputs, target_as):
    inputs = tf.interpolate(inputs, scale_factor=2, mode='nearest')
    _, _, h, w = target_as.size()
    if inputs.size(2) != h or inputs.size(3) != w:
        return tf.interpolate(inputs, [h, w], mode='bilinear', align_corners=False)
    else:
        return inputs


class OccUpsampleNetwork(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(OccUpsampleNetwork, self).__init__()
        self.feat_dim = 32
        self.init_conv = conv(ch_in, self.feat_dim)
        self.res_convs = nn.Sequential(conv(self.feat_dim, self.feat_dim), conv(self.feat_dim, self.feat_dim, isReLU=False))
        self.res_end_conv = conv(self.feat_dim, self.feat_dim)
        self.mul_const = 0.1
        self.out_convs = conv(self.feat_dim, ch_out)

    def forward(self, occ, x):
        occ = upsample_factor2(occ, x)
        x_in = torch.cat([occ, x], dim=1)
        x_init = self.init_conv(x_in)
        x_res = x_init
        x_res = x_res + self.res_convs(x_res) * self.mul_const
        x_res = x_res + self.res_convs(x_res) * self.mul_const
        x_res = x_res + self.res_convs(x_res) * self.mul_const
        x_init = x_init + self.res_end_conv(x_res)
        return self.out_convs(x_init) + occ


def subtract_mean(input):
    return input - input.mean(2).mean(2).unsqueeze(2).unsqueeze(2).expand_as(input)


class RefineFlow(nn.Module):

    def __init__(self, ch_in):
        super(RefineFlow, self).__init__()
        self.kernel_size = 3
        self.pad_size = 1
        self.pad_ftn = nn.ReplicationPad2d(self.pad_size)
        self.convs = nn.Sequential(conv(ch_in, 128, 3, 1, 1), conv(128, 128, 3, 1, 1), conv(128, 64, 3, 1, 1), conv(64, 64, 3, 1, 1), conv(64, 32, 3, 1, 1), conv(32, 32, 3, 1, 1), conv(32, self.kernel_size * self.kernel_size, 3, 1, 1))
        self.softmax_feat = nn.Softmax(dim=1)
        self.unfold_flow = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size))
        self.unfold_kernel = nn.Unfold(kernel_size=(1, 1))

    def forward(self, flow, diff_img, feature):
        b, _, h, w = flow.size()
        flow_m = subtract_mean(flow)
        norm2_img = torch.norm(diff_img, p=2, dim=1, keepdim=True)
        feat = self.convs(torch.cat([flow_m, norm2_img, feature], dim=1))
        feat_kernel = self.softmax_feat(-feat ** 2)
        flow_x = flow[:, 0].unsqueeze(1)
        flow_y = flow[:, 1].unsqueeze(1)
        flow_x_unfold = self.unfold_flow(self.pad_ftn(flow_x))
        flow_y_unfold = self.unfold_flow(self.pad_ftn(flow_y))
        feat_kernel_unfold = self.unfold_kernel(feat_kernel)
        flow_out_x = torch.sum(flow_x_unfold * feat_kernel_unfold, dim=1).unsqueeze(1).view(b, 1, h, w)
        flow_out_y = torch.sum(flow_y_unfold * feat_kernel_unfold, dim=1).unsqueeze(1).view(b, 1, h, w)
        return torch.cat([flow_out_x, flow_out_y], dim=1)


class RefineOcc(nn.Module):

    def __init__(self, ch_in):
        super(RefineOcc, self).__init__()
        self.kernel_size = 3
        self.pad_size = 1
        self.pad_ftn = nn.ReplicationPad2d(self.pad_size)
        self.convs = nn.Sequential(conv(ch_in, 128, 3, 1, 1), conv(128, 128, 3, 1, 1), conv(128, 64, 3, 1, 1), conv(64, 64, 3, 1, 1), conv(64, 32, 3, 1, 1), conv(32, 32, 3, 1, 1), conv(32, self.kernel_size * self.kernel_size, 3, 1, 1))
        self.softmax_feat = nn.Softmax(dim=1)
        self.unfold_occ = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size))
        self.unfold_kernel = nn.Unfold(kernel_size=(1, 1))

    def forward(self, occ, feat1, feat2):
        b, _, h, w = occ.size()
        feat = self.convs(torch.cat([occ, feat1, feat2], dim=1))
        feat_kernel = self.softmax_feat(-feat ** 2)
        occ_unfold = self.unfold_occ(self.pad_ftn(occ))
        feat_kernel_unfold = self.unfold_kernel(feat_kernel)
        occ_out = torch.sum(occ_unfold * feat_kernel_unfold, dim=1).unsqueeze(1).view(b, 1, h, w)
        return occ_out


def _downsample2d_as(inputs, target_as):
    _, _, h, w = target_as.size()
    return tf.adaptive_avg_pool2d(inputs, [h, w])


def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)


class MultiScaleEPE_FlowNet(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_FlowNet, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            outputs = [output_dict[key] for key in ['flow2', 'flow3', 'flow4', 'flow5', 'flow6']]
            target = self._args.div_flow * target_dict['flows'][:, 0]
            total_loss = 0
            for i, output_i in enumerate(outputs):
                target_i = _downsample2d_as(target, output_i)
                epe_i = _elementwise_epe(output_i, target_i)
                total_loss = total_loss + self._weights[i] * epe_i.sum() / self._batch_size
                loss_dict['epe%i' % (i + 2)] = epe_i.mean()
            loss_dict['loss'] = total_loss
        else:
            output = output_dict['flow1']
            target = target_dict['flows'][:, 0]
            epe = _elementwise_epe(output, target)
            loss_dict['epe'] = epe.mean()
        return loss_dict


class MultiScaleEPE_FlowNet_IRR(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_FlowNet_IRR, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            outputs_flo = [output_dict[key] for key in ['flow2', 'flow3', 'flow4', 'flow5', 'flow6']]
            target_f = self._args.div_flow * target_dict['flows'][:, 0]
            total_loss = 0
            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj, target_f_ii)
                    total_loss = total_loss + self._weights[ii] * epe_f_ii.sum()
                    loss_dict['epe%i' % (ii + 2)] = epe_f_ii.mean()
            loss_dict['loss'] = total_loss / self._batch_size / self._num_iters
        else:
            output = output_dict['flow1']
            target_f = target_dict['flows'][:, 0]
            epe_f = _elementwise_epe(target_f, output)
            loss_dict['epe'] = epe_f.mean()
        return loss_dict


class MultiScaleEPE_FlowNet_IRR_Bi(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_FlowNet_IRR_Bi, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            outputs_flo = [output_dict[key] for key in ['flow2', 'flow3', 'flow4', 'flow5', 'flow6']]
            target_f = self._args.div_flow * target_dict['flows'][:, 0]
            target_b = self._args.div_flow * target_dict['flows_b'][:, 0]
            total_loss = 0
            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0][0])
                target_b_ii = _downsample2d_as(target_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj[0], target_f_ii)
                    epe_b_ii = _elementwise_epe(output_ii_jj[1], target_b_ii)
                    total_loss = total_loss + self._weights[ii] * (epe_f_ii.sum() + epe_b_ii.sum())
                    loss_dict['epe%i' % (ii + 2)] = (epe_f_ii.mean() + epe_b_ii.mean()) / 2
            loss_dict['loss'] = total_loss / self._batch_size / self._num_iters / 2
        else:
            epe_f = _elementwise_epe(output_dict['flow1'], target_dict['flows'][:, 0])
            loss_dict['epe'] = epe_f.mean()
        return loss_dict


def fbeta_score(y_true, y_pred, beta, eps=1e-08):
    beta2 = beta ** 2
    y_pred = y_pred.float()
    y_true = y_true.float()
    true_positive = (y_pred * y_true).sum(dim=2).sum(dim=2)
    precision = true_positive / (y_pred.sum(dim=2).sum(dim=2) + eps)
    recall = true_positive / (y_true.sum(dim=2).sum(dim=2) + eps)
    return torch.mean(precision * recall / (precision * beta2 + recall + eps) * (1 + beta2))


def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 1)


def f1_score_bal_loss(y_pred, y_true):
    eps = 1e-08
    tp = -(y_true * torch.log(y_pred + eps)).sum(dim=2).sum(dim=2).sum(dim=1)
    fn = -((1 - y_true) * torch.log(1 - y_pred + eps)).sum(dim=2).sum(dim=2).sum(dim=1)
    denom_tp = y_true.sum(dim=2).sum(dim=2).sum(dim=1) + y_pred.sum(dim=2).sum(dim=2).sum(dim=1) + eps
    denom_fn = (1 - y_true).sum(dim=2).sum(dim=2).sum(dim=1) + (1 - y_pred).sum(dim=2).sum(dim=2).sum(dim=1) + eps
    return ((tp / denom_tp).sum() + (fn / denom_fn).sum()) * y_pred.size(2) * y_pred.size(3) * 0.5


class MultiScaleEPE_FlowNet_IRR_Occ(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_FlowNet_IRR_Occ, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters
        self.f1_score_bal_loss = f1_score_bal_loss
        self.occ_activ = nn.Sigmoid()

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            outputs_flo = [output_dict[key] for key in ['flow2', 'flow3', 'flow4', 'flow5', 'flow6']]
            outputs_occ = [output_dict[key] for key in ['occ2', 'occ3', 'occ4', 'occ5', 'occ6']]
            target = self._args.div_flow * target_dict['flows'][:, 0]
            target_occ = target_dict['occs'][:, 0]
            flow_loss = 0
            occ_loss = 0
            for ii, output_ii in enumerate(outputs_flo):
                target_ii = _downsample2d_as(target, output_ii[0])
                for jj, output_ii_jj in enumerate(output_ii):
                    flow_loss = flow_loss + self._weights[ii] * _elementwise_epe(output_ii_jj, target_ii).sum()
            for ii, output_ii in enumerate(outputs_occ):
                target_occ_f = _downsample2d_as(target_occ, output_ii[0])
                for jj, output_ii_jj in enumerate(output_ii):
                    occ_loss = occ_loss + self._weights[ii] * self.f1_score_bal_loss(self.occ_activ(output_ii_jj), target_occ_f)
            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1
            loss_dict['flow_loss'] = flow_loss / self._batch_size / self._num_iters
            loss_dict['occ_loss'] = occ_loss / self._batch_size / self._num_iters
            loss_dict['loss'] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size / self._num_iters
        else:
            loss_dict['epe'] = _elementwise_epe(output_dict['flow1'], target_dict['flows'][:, 0]).mean()
            loss_dict['F1'] = f1_score(target_dict['occs'][:, 0], torch.round(self.occ_activ(output_dict['occ1'])))
        return loss_dict


class MultiScaleEPE_FlowNet_IRR_Bi_Occ(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_FlowNet_IRR_Bi_Occ, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters
        self.f1_score_bal_loss = f1_score_bal_loss
        self.occ_activ = nn.Sigmoid()

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            outputs_flo = [output_dict[key] for key in ['flow2', 'flow3', 'flow4', 'flow5', 'flow6']]
            outputs_occ = [output_dict[key] for key in ['occ2', 'occ3', 'occ4', 'occ5', 'occ6']]
            target_f = self._args.div_flow * target_dict['flows'][:, 0]
            target_b = self._args.div_flow * target_dict['flows_b'][:, 0]
            target_occ_f = target_dict['occs'][:, 0]
            target_occ_b = target_dict['occs_b'][:, 0]
            flow_loss = 0
            occ_loss = 0
            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0][0])
                target_b_ii = _downsample2d_as(target_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj[0], target_f_ii)
                    epe_b_ii = _elementwise_epe(output_ii_jj[1], target_b_ii)
                    flow_loss = flow_loss + self._weights[ii] * (epe_f_ii.sum() + epe_b_ii.sum()) * 0.5
            for ii, output_ii in enumerate(outputs_occ):
                target_occ_f = _downsample2d_as(target_occ_f, output_ii[0][0])
                target_occ_b = _downsample2d_as(target_occ_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    output_occ_f = self.occ_activ(output_ii_jj[0])
                    output_occ_b = self.occ_activ(output_ii_jj[1])
                    bce_f_ii = self.f1_score_bal_loss(output_occ_f, target_occ_f)
                    bce_b_ii = self.f1_score_bal_loss(output_occ_b, target_occ_b)
                    occ_loss = occ_loss + self._weights[ii] * (bce_f_ii + bce_b_ii) * 0.5
            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1
            loss_dict['flow_loss'] = flow_loss / self._batch_size / self._num_iters
            loss_dict['occ_loss'] = occ_loss / self._batch_size / self._num_iters
            loss_dict['loss'] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size / self._num_iters
        else:
            loss_dict['epe'] = _elementwise_epe(output_dict['flow1'], target_dict['flows'][:, 0]).mean()
            loss_dict['F1'] = f1_score(target_dict['occs'][:, 0], torch.round(self.occ_activ(output_dict['occ1'])))
        return loss_dict


class MultiScaleEPE_FlowNet_IRR_Bi_Occ_upsample(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_FlowNet_IRR_Bi_Occ_upsample, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.0003125, 0.00125, 0.005, 0.01, 0.02, 0.08, 0.32]
        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            outputs_flo = [output_dict[key] for key in ['flow', 'flow1', 'flow2', 'flow3', 'flow4', 'flow5', 'flow6']]
            outputs_occ = [output_dict[key] for key in ['occ', 'occ1', 'occ2', 'occ3', 'occ4', 'occ5', 'occ6']]
            target_f = self._args.div_flow * target_dict['flows'][:, 0]
            target_b = self._args.div_flow * target_dict['flows_b'][:, 0]
            target_occ_f = target_dict['occs'][:, 0]
            target_occ_b = target_dict['occs_b'][:, 0]
            num_iters = len(outputs_flo[0])
            flow_loss = 0
            occ_loss = 0
            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0][0])
                target_b_ii = _downsample2d_as(target_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj[0], target_f_ii)
                    epe_b_ii = _elementwise_epe(output_ii_jj[1], target_b_ii)
                    flow_loss = flow_loss + self._weights[ii] * (epe_f_ii.sum() + epe_b_ii.sum()) * 0.5
            for ii, output_ii in enumerate(outputs_occ):
                target_occ_f = _downsample2d_as(target_occ_f, output_ii[0][0])
                target_occ_b = _downsample2d_as(target_occ_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    output_occ_f = self.occ_activ(output_ii_jj[0])
                    output_occ_b = self.occ_activ(output_ii_jj[1])
                    bce_f_ii = self.f1_score_bal_loss(output_occ_f, target_occ_f)
                    bce_b_ii = self.f1_score_bal_loss(output_occ_b, target_occ_b)
                    occ_loss = occ_loss + self._weights[ii] * (bce_f_ii + bce_b_ii) * 0.5
            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1
            loss_dict['flow_loss'] = flow_loss / self._batch_size / num_iters
            loss_dict['occ_loss'] = occ_loss / self._batch_size / num_iters
            loss_dict['loss'] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size / num_iters
        else:
            loss_dict['epe'] = _elementwise_epe(output_dict['flows'][:, 0], target_dict['flows'][:, 0]).mean()
            loss_dict['F1'] = f1_score(target_dict['occs'][:, 0], torch.round(self.occ_activ(output_dict['occ'])))
        return loss_dict


class MultiScaleEPE_PWC(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_PWC, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            outputs = output_dict['flow_preds']
            target = self._args.div_flow * target_dict['flows'][:, 0]
            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                loss_ii = _elementwise_epe(output_ii, _downsample2d_as(target, output_ii)).sum()
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict['loss'] = total_loss / self._batch_size
        else:
            epe = _elementwise_epe(output_dict['flows'][:, 0], target_dict['flows'][:, 0])
            loss_dict['epe'] = epe.mean()
        return loss_dict


class MultiScaleEPE_PWC_Bi(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_PWC_Bi, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            outputs = output_dict['flow_preds']
            target_f = self._args.div_flow * target_dict['flows'][:, 0]
            target_b = self._args.div_flow * target_dict['flows_b'][:, 0]
            total_loss = 0
            for i, output_i in enumerate(outputs):
                epe_i_f = _elementwise_epe(output_i[0], _downsample2d_as(target_f, output_i[0]))
                epe_i_b = _elementwise_epe(output_i[1], _downsample2d_as(target_b, output_i[1]))
                total_loss = total_loss + self._weights[i] * (epe_i_f.sum() + epe_i_b.sum())
            loss_dict['loss'] = total_loss / (2 * self._batch_size)
        else:
            epe = _elementwise_epe(output_dict['flows'][:, 0], target_dict['flows'][:, 0])
            loss_dict['epe'] = epe.mean()
        return loss_dict


class MultiScaleEPE_PWC_Occ(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_PWC_Occ, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]
        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']
            target_flo = self._args.div_flow * target_dict['flows'][:, 0]
            target_occ = target_dict['occs'][:, 0]
            flow_loss = 0
            occ_loss = 0
            for i, output_i in enumerate(output_flo):
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i, _downsample2d_as(target_flo, output_i)).sum()
            for i, output_i in enumerate(output_occ):
                output_occ = self.occ_activ(output_i)
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ, _downsample2d_as(target_occ, output_occ))
            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1
            loss_dict['flow_loss'] = flow_loss / self._batch_size
            loss_dict['occ_loss'] = occ_loss / self._batch_size
            loss_dict['loss'] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size
        else:
            loss_dict['epe'] = _elementwise_epe(output_dict['flows'][:, 0], target_dict['flows'][:, 0]).mean()
            loss_dict['F1'] = f1_score(target_dict['occs'][:, 0], torch.round(self.occ_activ(output_dict['occ'])))
        return loss_dict


class MultiScaleEPE_PWC_Bi_Occ(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_PWC_Bi_Occ, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]
        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']
            target_flo_f = self._args.div_flow * target_dict['flows'][:, 0]
            target_flo_b = self._args.div_flow * target_dict['flows_b'][:, 0]
            target_occ_f = target_dict['occs'][:, 0]
            target_occ_b = target_dict['occs_b'][:, 0]
            flow_loss = 0
            occ_loss = 0
            for i, output_i in enumerate(output_flo):
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i[0], _downsample2d_as(target_flo_f, output_i[0])).sum()
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i[1], _downsample2d_as(target_flo_b, output_i[1])).sum()
            for i, output_i in enumerate(output_occ):
                output_occ_f = self.occ_activ(output_i[0])
                output_occ_b = self.occ_activ(output_i[1])
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ_b, _downsample2d_as(target_occ_b, output_occ_b))
            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1
            loss_dict['flow_loss'] = flow_loss / (2 * self._batch_size)
            loss_dict['occ_loss'] = occ_loss / (2 * self._batch_size)
            loss_dict['loss'] = (flow_loss * f_l_w + occ_loss * o_l_w) / (2 * self._batch_size)
        else:
            loss_dict['epe'] = _elementwise_epe(output_dict['flows'][:, 0], target_dict['flows'][:, 0]).mean()
            loss_dict['F1'] = f1_score(target_dict['occs'][:, 0], torch.round(self.occ_activ(output_dict['occ'])))
        return loss_dict


class MultiScaleEPE_PWC_Bi_Occ_upsample(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_PWC_Bi_Occ_upsample, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]
        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']
            target_flo_f = self._args.div_flow * target_dict['flows'][:, 0]
            target_flo_b = self._args.div_flow * target_dict['flows_b'][:, 0]
            target_occ_f = target_dict['occs'][:, 0]
            target_occ_b = target_dict['occs_b'][:, 0]
            flow_loss = 0
            occ_loss = 0
            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj], _downsample2d_as(target_flo_f, output_ii[2 * jj])).sum()
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj + 1], _downsample2d_as(target_flo_b, output_ii[2 * jj + 1])).sum()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii)
            for ii, output_ii in enumerate(output_occ):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    output_occ_f = self.occ_activ(output_ii[2 * jj])
                    output_occ_b = self.occ_activ(output_ii[2 * jj + 1])
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_b, _downsample2d_as(target_occ_b, output_occ_b))
                occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii)
            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1
            loss_dict['flow_loss'] = flow_loss / self._batch_size
            loss_dict['occ_loss'] = occ_loss / self._batch_size
            loss_dict['loss'] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size
        else:
            loss_dict['epe'] = _elementwise_epe(output_dict['flow'], target_dict['flows'][:, 0]).mean()
            loss_dict['F1'] = f1_score(target_dict['occs'][:, 0], torch.round(self.occ_activ(output_dict['occ'])))
        return loss_dict


def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)


def occ_to_mask(occ, return_np=False):
    tens_out = nn.Sigmoid()(occ)
    if return_np:
        return np.round(tens_out.expand(-1, 3, -1, -1).data.cpu().numpy().transpose([0, 2, 3, 1])) * 255
    return tens_out.round()


class MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]
        self.occ_activ = nn.Sigmoid()
        self.occ_loss_bce = nn.BCELoss(reduction='sum')
        self.seploss = hasattr(args, 'seploss') and self._args.seploss or False
        if self.seploss:
            None
        self.loss_perc = hasattr(args, 'loss_perc') and args.loss_perc
        if self.loss_perc:
            from matplotlib.pyplot import hist
            self.perc = Percentile()
            None
            self.min_p = nn.Parameter(torch.Tensor([30]), requires_grad=False)
            self.max_p = nn.Parameter(torch.Tensor([97]), requires_grad=False)

    def _get_flow_loss(self, output_flo, target_flo_f):
        flow_loss = 0
        for ii, output_ii in enumerate(output_flo):
            loss_ii = 0
            for jj in range(0, len(output_ii) // 2):
                cur_epe = _elementwise_robust_epe_char(output_ii[2 * jj], _downsample2d_as(target_flo_f, output_ii[2 * jj]))
                if self.loss_perc:
                    tmin = torch.Tensor([self.perc(ten.flatten(), [self.min_p]) for ten in cur_epe])
                    tmax = torch.Tensor([self.perc(ten.flatten(), [self.max_p]) for ten in cur_epe])
                    cur_epe[cur_epe > tmax.view(-1, 1, 1, 1)] = 0.0
                    cur_epe[cur_epe < tmin.view(-1, 1, 1, 1)] = 0.0
                    tmin.detach()
                    tmax.detach()
                loss_ii = loss_ii + cur_epe.sum()
                output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
            flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii) * 2
        return flow_loss

    def _get_occ_loss(self, output_occ, target_occ_f):
        occ_loss = 0
        for ii, output_ii in enumerate(output_occ):
            loss_ii = 0
            for jj in range(0, len(output_ii) // 2):
                output_occ_f = self.occ_activ(output_ii[2 * jj])
                output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                loss_ii = loss_ii + self.occ_loss_bce(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
            occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii) * 2
        return occ_loss

    @staticmethod
    def mask_flow(output_flo, target_flo_f, output_occ, use_target=False, min_layer=0):
        for lix, layer_flow in enumerate(output_flo):
            if lix < min_layer:
                continue
            for fcix, flow_category in enumerate(layer_flow):
                if use_target:
                    occ_mask = _downsample2d_as(target_flo_f, output_flo[lix][fcix])
                else:
                    occ_mask = occ_to_mask(output_occ[lix][fcix])
                output_flo[lix][fcix] = (1 - occ_mask) * output_flo[lix][fcix]
        if use_target:
            target_flo_f = (1 - target_flo_f) * target_flo_f
        else:
            target_flo_f = (1 - occ_to_mask(output_occ[6][0])) * target_flo_f
        return output_flo, target_flo_f

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']
            target_flo_f = self._args.div_flow * target_dict['flows'][:, 0]
            target_occ_f = target_dict['occs'][:, 0]
            if self.seploss:
                output_flo, target_flo_f = self.mask_flow(output_flo, target_flo_f, output_occ)
            flow_loss = self._get_flow_loss(output_flo, target_flo_f)
            occ_loss = self._get_occ_loss(output_occ, target_occ_f)
            loss_dict['flow_loss'] = flow_loss / self._batch_size
            loss_dict['occ_loss'] = occ_loss / self._batch_size
            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1
            losses_mix = flow_loss * f_l_w + occ_loss * o_l_w
            loss_dict['loss'] = losses_mix / self._batch_size
        else:
            loss_dict['epe'] = _elementwise_epe(output_dict['flow'], target_dict['flows'][:, 0]).mean()
            loss_dict['F1'] = f1_score(target_dict['occs'][:, 0], torch.round(self.occ_activ(output_dict['occ'])))
        return loss_dict


def _upsample2d_as(inputs, target_as, mode='bilinear'):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)


class MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI(nn.Module):

    def __init__(self, args):
        super(MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.001, 0.001, 0.001, 0.002, 0.004, 0.004, 0.004]
        self.occ_activ = nn.Sigmoid()

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        valid_mask = target_dict['valids'][:, 0]
        b, _, h, w = target_dict['flows'][:, 0].size()
        if self.training:
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']
            target_flo_f = self._args.div_flow * target_dict['flows'][:, 0]
            flow_loss = 0
            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    valid_epe = _elementwise_robust_epe_char(_upsample2d_as(output_ii[2 * jj], target_flo_f), target_flo_f) * valid_mask
                    for bb in range(0, b):
                        valid_epe[bb, ...][valid_mask[bb, ...] == 0] = valid_epe[bb, ...][valid_mask[bb, ...] == 0].detach()
                        norm_const = h * w / valid_mask[bb, ...].sum()
                        loss_ii = loss_ii + valid_epe[bb, ...][valid_mask[bb, ...] != 0].sum() * norm_const
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii) * 2
            for ii, output_ii in enumerate(output_occ):
                for jj in range(0, len(output_ii) // 2):
                    output_ii[2 * jj] = output_ii[2 * jj].detach()
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
            loss_dict['flow_loss'] = flow_loss / self._batch_size
            loss_dict['loss'] = flow_loss / self._batch_size
        else:
            flow_gt_mag = torch.norm(target_dict['flows'][:, 0], p=2, dim=1, keepdim=True) + 1e-08
            flow_epe = _elementwise_epe(output_dict['flow'], target_dict['flows'][:, 0]) * valid_mask
            epe_per_image = flow_epe.view(b, -1).sum(1) / valid_mask.view(b, -1).sum(1)
            loss_dict['epe'] = epe_per_image.mean()
            outlier_epe = (flow_epe > 3).float() * (flow_epe / flow_gt_mag > 0.05).float() * valid_mask
            outlier_per_image = outlier_epe.view(b, -1).sum(1) / valid_mask.view(b, -1).sum(1)
            loss_dict['outlier'] = outlier_per_image.mean()
        return loss_dict


class FeatureExtractor(nn.Module):

    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()
        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(conv(ch_in, ch_out, stride=2), conv(ch_out, ch_out))
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)
        return feature_pyramid[::-1]


def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False)
    if x.is_cuda:
        grids_cuda = grids_cuda
    return grids_cuda


class WarpingLayer(nn.Module):

    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
        flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid, align_corners=True)
        mask = torch.ones(x.size(), requires_grad=False)
        if x.is_cuda:
            mask = mask
        mask = tf.grid_sample(mask, grid, align_corners=True)
        mask = (mask >= 1.0).float()
        return x_warp * mask


class OpticalFlowEstimator(nn.Module):

    def __init__(self, ch_in):
        super(OpticalFlowEstimator, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128), conv(128, 128), conv(128, 96), conv(96, 64), conv(64, 32))
        self.conv_last = conv(32, 2, isReLU=False)

    def forward(self, x):
        x_intm = self.convs(x)
        return x_intm, self.conv_last(x_intm)


class FlowEstimatorDense(nn.Module):

    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class OcclusionEstimator(nn.Module):

    def __init__(self, ch_in):
        super(OcclusionEstimator, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128), conv(128, 128), conv(128, 96), conv(96, 64), conv(64, 32))
        self.conv_last = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_intm = self.convs(x)
        return x_intm, self.conv_last(x_intm)


class OccEstimatorDense(nn.Module):

    def __init__(self, ch_in):
        super(OccEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 1, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class ContextNetwork(nn.Module):

    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128, 3, 1, 1), conv(128, 128, 3, 1, 2), conv(128, 128, 3, 1, 4), conv(128, 96, 3, 1, 8), conv(96, 64, 3, 1, 16), conv(64, 32, 3, 1, 1), conv(32, 2, isReLU=False))

    def forward(self, x):
        return self.convs(x)


class OccContextNetwork(nn.Module):

    def __init__(self, ch_in):
        super(OccContextNetwork, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128, 3, 1, 1), conv(128, 128, 3, 1, 2), conv(128, 128, 3, 1, 4), conv(128, 96, 3, 1, 8), conv(96, 64, 3, 1, 16), conv(64, 32, 3, 1, 1), conv(32, 1, isReLU=False))

    def forward(self, x):
        return self.convs(x)


class LearnableCorrBlock(torch.nn.Module):

    def __init__(self, dim, num_levels=4, radius=4):
        super(LearnableCorrBlock, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.dim = dim
        self.raw_P = torch.nn.Parameter(torch.eye(self.dim), requires_grad=True)
        self.raw_D = torch.nn.Parameter(torch.zeros(self.dim), requires_grad=True)
        self.register_buffer('eye', torch.eye(self.dim))

    def compute_cost_volume(self, fmap1, fmap2):
        self.raw_P_upper = torch.triu(self.raw_P)
        self.skew_P = (self.raw_P_upper - self.raw_P_upper.t()) / 2
        self.P = torch.matmul(self.eye + self.skew_P, torch.inverse(self.eye - self.skew_P))
        self.trans_D = torch.atan(self.raw_D) * 2 / math.pi
        self.D = torch.diag((1 + self.trans_D) / (1 - self.trans_D))
        self.W = torch.matmul(torch.matmul(self.P.t(), self.D), self.P)
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(torch.tensordot(fmap1, self.W, dims=[[1], [0]]), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr / torch.sqrt(torch.tensor(dim).float())
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.view(batch * h1 * w1, dim, h2, w2)
        corr_pyramid = []
        corr_pyramid.append(corr)
        for i in range(self.num_levels):
            if min(corr.shape[2:4]) > 2 * self.radius + 1:
                corr = F.avg_pool2d(corr, 2, stride=2)
            corr_pyramid.append(corr)
        return corr_pyramid

    def forward(self, corr_pyramid, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2)


class Matching(nn.Module):

    def __init__(self, level: int, num_levels: int=4, div_flow: float=20.0, use_s_version: bool=False) ->None:
        super(Matching, self).__init__()
        flow_kernel_size = [3, 3, 5, 5][level]
        self.mult = [(div_flow / 2 ** (num_levels - i + 1)) for i in range(num_levels)][level]
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        if level == 1 and not use_s_version:
            self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)
        else:
            self.up_flow = None
        if level < 2:
            self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=9, padding=0, stride=1, dilation_patch=1)
        else:
            self.corr = None
        self.flow_net = nn.Sequential(nn.Conv2d(81, 128, 3, 1, 1), self.leaky_relu, nn.Conv2d(128, 128, 3, 1, 1), self.leaky_relu, nn.Conv2d(128, 96, 3, 1, 1), self.leaky_relu, nn.Conv2d(96, 64, 3, 1, 1), self.leaky_relu, nn.Conv2d(64, 32, 3, 1, 1), self.leaky_relu, nn.Conv2d(32, 2, flow_kernel_size, 1, flow_kernel_size // 2))
        self.warp = WarpingLayer()

    def forward(self, feats: torch.Tensor, flow: Optional[torch.Tensor], corr: Optional[torch.Tensor]) ->torch.Tensor:
        if self.up_flow is not None:
            flow = self.up_flow(flow)
        if corr is None:
            warped_feat2 = feats[:, 1]
            if flow is not None:
                warped_feat2 = self.warp(feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0 / self.mult)
            corr = self.leaky_relu(self.corr(feats[:, 0], warped_feat2))
            corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
            corr = corr / feats.shape[2]
        new_flow = self.flow_net(corr)
        if flow is not None:
            new_flow = flow + new_flow
        return new_flow


class SubPixel(nn.Module):

    def __init__(self, level: int, num_levels: int=4, div_flow: float=20.0) ->None:
        super(SubPixel, self).__init__()
        inputs_dims = [386, 258, 194, 130][level]
        flow_kernel_size = [3, 3, 5, 5][level]
        self.mult = [(div_flow / 2 ** (num_levels - i + 1)) for i in range(num_levels)][level]
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.feat_net = nn.Sequential(nn.Conv2d(inputs_dims, 128, 3, 1, 1), self.leaky_relu, nn.Conv2d(128, 128, 3, 1, 1), self.leaky_relu, nn.Conv2d(128, 96, 3, 1, 1), self.leaky_relu, nn.Conv2d(96, 64, 3, 1, 1), self.leaky_relu, nn.Conv2d(64, 32, 3, 1, 1), self.leaky_relu)
        self.flow_net = nn.Conv2d(32, 2, flow_kernel_size, 1, flow_kernel_size // 2)
        self.warp = WarpingLayer()

    def forward(self, feats: torch.Tensor, flow: torch.Tensor) ->torch.Tensor:
        feat_warped = self.warp(feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0 / self.mult)
        x = torch.cat([feats[:, 0], feat_warped, flow], dim=1)
        x = self.feat_net(x)
        new_flow = self.flow_net(x)
        new_flow = flow + new_flow
        return new_flow, x


class Regularization(nn.Module):

    def __init__(self, level: int, num_levels: int=4, div_flow: float=20.0, use_s_version: bool=False) ->None:
        super(Regularization, self).__init__()
        inputs_dims = [195, 131, 99, 67][level]
        flow_kernel_size = [3, 3, 5, 5][level]
        conf_kernel_size = [3, 3, 5, None][level]
        self.mult = [(div_flow / 2 ** (num_levels - i + 1)) for i in range(num_levels)][level]
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        if level < 2:
            self.feat_conv = nn.Sequential()
        else:
            self.feat_conv = nn.Sequential(nn.Conv2d(inputs_dims - 3, 128, 1, 1, 0), self.leaky_relu)
            inputs_dims = 131
        self.feat_net = nn.Sequential(nn.Conv2d(inputs_dims, 128, 3, 1, 1), self.leaky_relu, nn.Conv2d(128, 128, 3, 1, 1), self.leaky_relu, nn.Conv2d(128, 64, 3, 1, 1), self.leaky_relu, nn.Conv2d(64, 64, 3, 1, 1), self.leaky_relu, nn.Conv2d(64, 32, 3, 1, 1), self.leaky_relu, nn.Conv2d(32, 32, 3, 1, 1), self.leaky_relu)
        if level < 2:
            self.dist = nn.Conv2d(32, flow_kernel_size ** 2, 3, 1, 1)
        else:
            self.dist = nn.Sequential(nn.Conv2d(32, flow_kernel_size ** 2, (flow_kernel_size, 1), 1, (flow_kernel_size // 2, 0)), nn.Conv2d(flow_kernel_size ** 2, flow_kernel_size ** 2, (1, flow_kernel_size), 1, (0, flow_kernel_size // 2)))
        self.unfold = nn.Unfold(flow_kernel_size, padding=flow_kernel_size // 2)
        if level == 0 and not use_s_version or level == 3:
            self.conf_pred = None
        else:
            self.conf_pred = nn.Sequential(nn.Conv2d(32, 1, conf_kernel_size, 1, conf_kernel_size // 2), nn.Sigmoid())
        self.warp = WarpingLayer()

    def forward(self, images: torch.Tensor, feats: torch.Tensor, flow: torch.Tensor) ->torch.Tensor:
        img2_warped = self.warp(images[:, 1], flow, images.shape[-2], images.shape[-1], 1.0 / self.mult)
        img_diff_norm = torch.norm(images[:, 0] - img2_warped, p=2, dim=1, keepdim=True)
        flow_mean = flow.view(*flow.shape[:2], -1).mean(dim=-1)[..., None, None]
        flow_nomean = flow - flow_mean
        feat = self.feat_conv(feats[:, 0])
        x = torch.cat([img_diff_norm, flow_nomean, feat], dim=1)
        x = self.feat_net(x)
        dist = self.dist(x)
        dist = dist.square().neg()
        dist = (dist - dist.max(dim=1, keepdim=True)[0]).exp()
        div = dist.sum(dim=1, keepdim=True)
        reshaped_flow_x = self.unfold(flow[:, :1])
        reshaped_flow_x = reshaped_flow_x.view(*reshaped_flow_x.shape[:2], *flow.shape[2:4])
        flow_smooth_x = (reshaped_flow_x * dist).sum(dim=1, keepdim=True) / div
        reshaped_flow_y = self.unfold(flow[:, 1:2])
        reshaped_flow_y = reshaped_flow_y.view(*reshaped_flow_y.shape[:2], *flow.shape[2:4])
        flow_smooth_y = (reshaped_flow_y * dist).sum(dim=1, keepdim=True) / div
        flow = torch.cat([flow_smooth_x, flow_smooth_y], dim=1)
        conf = None
        if self.conf_pred is not None:
            conf = self.conf_pred(x)
        return flow, conf, x


class PseudoSubpixel(nn.Module):

    def __init__(self) ->None:
        super(PseudoSubpixel, self).__init__()
        self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)
        self.flow_net = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.Conv2d(32, 2, 7, 1, 3))

    def forward(self, sub_feat: torch.Tensor, flow: torch.Tensor) ->torch.Tensor:
        return self.up_flow(flow) + self.flow_net(sub_feat)


class PseudoRegularization(nn.Module):

    def __init__(self) ->None:
        super(PseudoRegularization, self).__init__()
        self.feat_net = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.Conv2d(32, 49, (7, 1), 1, (3, 0)), nn.Conv2d(49, 49, (1, 7), 1, (0, 3)))
        self.unfold = nn.Unfold(7, padding=3)

    def forward(self, reg_feat: torch.Tensor, flow: torch.Tensor) ->torch.Tensor:
        dist = self.feat_net(reg_feat)
        dist = dist.square().neg()
        dist = (dist - dist.max(dim=1, keepdim=True)[0]).exp()
        div = dist.sum(dim=1, keepdim=True)
        reshaped_flow_x = self.unfold(flow[:, :1])
        reshaped_flow_x = reshaped_flow_x.view(*reshaped_flow_x.shape[:2], *flow.shape[2:4])
        flow_smooth_x = (reshaped_flow_x * dist).sum(dim=1, keepdim=True) / div
        reshaped_flow_y = self.unfold(flow[:, 1:2])
        reshaped_flow_y = reshaped_flow_y.view(*reshaped_flow_y.shape[:2], *flow.shape[2:4])
        flow_smooth_y = (reshaped_flow_y * dist).sum(dim=1, keepdim=True) / div
        flow = torch.cat([flow_smooth_x, flow_smooth_y], dim=1)
        return flow


class FlowFieldDeformation(nn.Module):

    def __init__(self, level: int) ->None:
        super(FlowFieldDeformation, self).__init__()
        patch_size = [None, 5, 7, 9][level]
        pred_kernel_size = [None, 3, 5, 5][level]
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.up_conf = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)
        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=patch_size, padding=0, stride=1, dilation_patch=2)
        self.feat_net = nn.Sequential(nn.Conv2d(patch_size ** 2 + 1, 128, 3, 1, 1), self.leaky_relu, nn.Conv2d(128, 64, 3, 1, 1), self.leaky_relu, nn.Conv2d(64, 32, 3, 1, 1), self.leaky_relu)
        self.disp_pred = nn.Conv2d(32, 2, pred_kernel_size, 1, pred_kernel_size // 2)
        self.conf_pred = nn.Sequential(nn.Conv2d(32, 1, pred_kernel_size, 1, pred_kernel_size // 2), nn.Sigmoid())
        self.warp = WarpingLayer()

    def forward(self, feats: torch.Tensor, flow: torch.Tensor, conf: torch.Tensor) ->torch.Tensor:
        conf = self.up_conf(conf)
        flow = self.up_flow(flow)
        self_corr = self.leaky_relu(self.corr(feats[:, 0], feats[:, 0]))
        self_corr = self_corr.view(self_corr.shape[0], -1, self_corr.shape[3], self_corr.shape[4])
        self_corr = self_corr / feats.shape[2]
        x = torch.cat([self_corr, conf], dim=1)
        x = self.feat_net(x)
        disp = self.disp_pred(x)
        flow = self.warp(flow, disp, flow.shape[-2], flow.shape[-1], 1.0)
        conf = self.conf_pred(x)
        return flow, conf


class CostVolumeModulation(nn.Module):

    def __init__(self, level: int, num_levels: int=4, div_flow: float=20.0) ->None:
        super().__init__()
        input_dims = [None, 210, 178, 146][level]
        self.mult = [(div_flow / 2 ** (num_levels - i + 1)) for i in range(num_levels)][level]
        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=9, padding=0, stride=1, dilation_patch=1)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.feat_net = nn.Sequential(nn.Conv2d(input_dims, 128, 3, 1, 1), self.leaky_relu, nn.Conv2d(128, 64, 3, 1, 1), self.leaky_relu)
        self.mod_scalar_net = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), self.leaky_relu, nn.Conv2d(32, 81, 1, 1, 0))
        self.mod_offset_net = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), self.leaky_relu, nn.Conv2d(32, 81, 1, 1, 0))
        self.warp = WarpingLayer()

    def forward(self, feats: torch.Tensor, flow: torch.Tensor, conf: torch.Tensor) ->torch.Tensor:
        warped_feat2 = self.warp(feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0 / self.mult)
        corr = self.leaky_relu(self.corr(feats[:, 0], warped_feat2))
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        corr = corr / feats.shape[2]
        x = torch.cat([feats[:, 0], corr, conf], dim=1)
        x = self.feat_net(x)
        mod_scalar = self.mod_scalar_net(x)
        mod_offset = self.mod_offset_net(x)
        corr = mod_scalar * corr + mod_offset
        return corr


class EpeLoss(nn.Module):

    def __init__(self, eps=0):
        super(EpeLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, label):
        loss = ((pred - label).pow(2).sum(1) + self.eps).sqrt()
        return loss.view(loss.shape[0], -1).mean(1)


class EpeLossWithMask(nn.Module):

    def __init__(self, eps=1e-08, q=None):
        super(EpeLossWithMask, self).__init__()
        self.eps = eps
        self.q = q

    def forward(self, pred, label, mask):
        if self.q is not None:
            loss = ((pred - label).abs().sum(1) + self.eps) ** self.q
        else:
            loss = ((pred - label).pow(2).sum(1) + self.eps).sqrt()
        loss = loss * mask.squeeze(1)
        loss = loss.view(loss.shape[0], -1).sum(1) / mask.view(mask.shape[0], -1).sum(1)
        return loss


def downsample_kernel2d(w, device):
    kernel = (w + 1 - torch.abs(w - torch.arange(w * 2 + 1, dtype=torch.float32, device=device))) / (2 * w + 1)
    kernel = kernel.repeat(w).view(w, -1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w * 2 + 1, w * 2 + 1)


def Downsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B * C, 1, H, W)
    kernel = downsample_kernel2d(factor // 2, img.device)
    upsamp_img = F.conv2d(batch_img, kernel, stride=factor, padding=factor // 2)
    upsamp_nom = F.conv2d(torch.ones_like(batch_img), kernel, stride=factor, padding=factor // 2)
    _, _, H_up, W_up = upsamp_img.shape
    upsamp_img = upsamp_img.view(B, C, H_up, W_up)
    upsamp_nom = upsamp_nom.view(B, C, H_up, W_up)
    return upsamp_img / upsamp_nom


def upsample_kernel2d(w, device):
    c = w // 2
    kernel = 1 - torch.abs(c - torch.arange(w, dtype=torch.float32, device=device)) / (c + 1)
    kernel = kernel.repeat(w).view(w, -1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w, w)


def Upsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B * C, 1, H, W)
    batch_img = F.pad(batch_img, [0, 1, 0, 1], mode='replicate')
    kernel = upsample_kernel2d(factor * 2 - 1, img.device)
    upsamp_img = F.conv_transpose2d(batch_img, kernel, stride=factor, padding=factor - 1)
    upsamp_img = upsamp_img[:, :, :-1, :-1]
    _, _, H_up, W_up = upsamp_img.shape
    return upsamp_img.view(B, C, H_up, W_up)


class MultiscaleEpe(nn.Module):

    def __init__(self, scales, weights, match, eps=1e-08, q=None):
        super(MultiscaleEpe, self).__init__()
        self.scales = scales
        self.weights = weights
        self.match = match
        self.eps = eps
        self.q = q

    def forward(self, output, target):
        flow = target['flows'][:, 0]
        mask = target['valids'][:, 0]
        predictions = output['flow_preds']
        losses = 0
        if self.match == 'upsampling':
            for p, w, s in zip(predictions, self.weights, self.scales):
                losses += EpeLossWithMask(eps=self.eps, q=self.q)(Upsample(p, s), flow, mask) * w
        elif self.match == 'downsampling':
            for p, w, s in zip(predictions, self.weights, self.scales):
                losses += EpeLossWithMask(eps=self.eps, q=self.q)(p, Downsample(flow, s), Downsample(mask, s)) * w
        else:
            raise NotImplementedError
        return losses


class NO_OP(nn.Module):

    def __init__(self, args=None):
        super(NO_OP, self).__init__()

    def forward(self, output_dict, target_dict):
        return {'flow_loss': -1, 'epe': -1, 'total_loss': torch.Tensor([0])}


class BasicEncoderQuarter(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoderQuarter, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class BasicUpdateBlockQuarter(nn.Module):

    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlockQuarter, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, input_dim=input_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(input_dim=hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 16 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class FlowAndOccEstimatorDense(nn.Module):

    def __init__(self, ch_in):
        super(FlowAndOccEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 3, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out[:, :2, :, :], x_out[:, 2, :, :].unsqueeze(1)


class FlowAndOccContextNetwork(nn.Module):

    def __init__(self, ch_in):
        super(FlowAndOccContextNetwork, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128, 3, 1, 1), conv(128, 128, 3, 1, 2), conv(128, 128, 3, 1, 4), conv(128, 96, 3, 1, 8), conv(96, 64, 3, 1, 16), conv(64, 32, 3, 1, 1), conv(32, 3, isReLU=False))

    def forward(self, x):
        x_out = self.convs(x)
        return x_out[:, :2, :, :], x_out[:, 2, :, :].unsqueeze(1)


def conv4d(data, filters, bias=None, permute_filters=True, use_half=False):
    """
    This is done by stacking results of multiple 3D convolutions, and is very slow.
    Taken from https://github.com/ignacio-rocco/ncnet
    """
    b, c, h, w, d, t = data.size()
    data = data.permute(2, 0, 1, 3, 4, 5).contiguous()
    if permute_filters:
        filters = filters.permute(2, 0, 1, 3, 4, 5).contiguous()
    c_out = filters.size(1)
    if use_half:
        output = Variable(torch.HalfTensor(h, b, c_out, w, d, t), requires_grad=data.requires_grad)
    else:
        output = Variable(torch.zeros(h, b, c_out, w, d, t), requires_grad=data.requires_grad)
    padding = filters.size(0) // 2
    if use_half:
        Z = Variable(torch.zeros(padding, b, c, w, d, t).half())
    else:
        Z = Variable(torch.zeros(padding, b, c, w, d, t))
    if data.is_cuda:
        Z = Z
        output = output
    data_padded = torch.cat((Z, data, Z), 0)
    for i in range(output.size(0)):
        output[i, :, :, :, :, :] = F.conv3d(data_padded[i + padding, :, :, :, :, :], filters[padding, :, :, :, :, :], bias=bias, stride=1, padding=padding)
        for p in range(1, padding + 1):
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(data_padded[i + padding - p, :, :, :, :, :], filters[padding - p, :, :, :, :, :], bias=None, stride=1, padding=padding)
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(data_padded[i + padding + p, :, :, :, :, :], filters[padding + p, :, :, :, :, :], bias=None, stride=1, padding=padding)
    output = output.permute(1, 2, 0, 3, 4, 5).contiguous()
    return output


class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True):
        stride = 1
        dilation = 1
        groups = 1
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        super(Conv4d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _quadruple(0), groups, bias)
        self.pre_permuted_filters = pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data = self.weight.data.permute(2, 0, 1, 3, 4, 5).contiguous()
        self.use_half = False

    def forward(self, input):
        out = conv4d(input, self.weight, bias=self.bias, permute_filters=not self.pre_permuted_filters, use_half=self.use_half)
        return out


class fullConv4d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True):
        super(fullConv4d, self).__init__()
        self.conv = Conv4d(in_channels, out_channels, kernel_size, bias=bias, pre_permuted_filters=pre_permuted_filters)
        self.isbias = bias
        if not self.isbias:
            self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, input):
        out = self.conv(input)
        if not self.isbias:
            b, c, u, v, h, w = out.shape
            out = self.bn(out.view(b, c, -1)).view(b, c, u, v, h, w)
        return out


class projfeat4d(torch.nn.Module):
    """
    Turn 3d projection into 2d projection
    """

    def __init__(self, in_planes, out_planes, stride, with_bn=True, groups=1):
        super(projfeat4d, self).__init__()
        self.with_bn = with_bn
        self.stride = stride
        self.conv1 = nn.Conv3d(in_planes, out_planes, 1, (stride, stride, 1), padding=0, bias=not with_bn, groups=groups)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        b, c, u, v, h, w = x.size()
        x = self.conv1(x.view(b, c, u, v, h * w))
        if self.with_bn:
            x = self.bn(x)
        _, c, u, v, _ = x.shape
        x = x.view(b, c, u, v, h, w)
        return x


class sepConv4d(torch.nn.Module):
    """
    Separable 4d convolution block as 2 3D convolutions
    """

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1), with_bn=True, ksize=3, full=True, groups=1):
        super(sepConv4d, self).__init__()
        bias = not with_bn
        self.isproj = False
        self.stride = stride[0]
        expand = 1
        if with_bn:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0, groups=groups), nn.BatchNorm2d(out_planes))
            if full:
                self.conv1 = nn.Sequential(nn.Conv3d(in_planes * expand, in_planes, (1, ksize, ksize), stride=(1, self.stride, self.stride), bias=bias, padding=(0, ksize // 2, ksize // 2), groups=groups), nn.BatchNorm3d(in_planes))
            else:
                self.conv1 = nn.Sequential(nn.Conv3d(in_planes * expand, in_planes, (1, ksize, ksize), stride=1, bias=bias, padding=(0, ksize // 2, ksize // 2), groups=groups), nn.BatchNorm3d(in_planes))
            self.conv2 = nn.Sequential(nn.Conv3d(in_planes, in_planes * expand, (ksize, ksize, 1), stride=(self.stride, self.stride, 1), bias=bias, padding=(ksize // 2, ksize // 2, 0), groups=groups), nn.BatchNorm3d(in_planes * expand))
        else:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0, groups=groups)
            if full:
                self.conv1 = nn.Conv3d(in_planes * expand, in_planes, (1, ksize, ksize), stride=(1, self.stride, self.stride), bias=bias, padding=(0, ksize // 2, ksize // 2), groups=groups)
            else:
                self.conv1 = nn.Conv3d(in_planes * expand, in_planes, (1, ksize, ksize), stride=1, bias=bias, padding=(0, ksize // 2, ksize // 2), groups=groups)
            self.conv2 = nn.Conv3d(in_planes, in_planes * expand, (ksize, ksize, 1), stride=(self.stride, self.stride, 1), bias=bias, padding=(ksize // 2, ksize // 2, 0), groups=groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, u, v, h, w = x.shape
        x = self.conv2(x.view(b, c, u, v, -1))
        b, c, u, v, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b, c, -1, h, w))
        b, c, _, h, w = x.shape
        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x


class sepConv4dBlock(torch.nn.Module):
    """
    Separable 4d convolution block as 2 2D convolutions and a projection
    layer
    """

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1), with_bn=True, full=True, groups=1):
        super(sepConv4dBlock, self).__init__()
        if in_planes == out_planes and stride == (1, 1, 1):
            self.downsample = None
        elif full:
            self.downsample = sepConv4d(in_planes, out_planes, stride, with_bn=with_bn, ksize=1, full=full, groups=groups)
        else:
            self.downsample = projfeat4d(in_planes, out_planes, stride[0], with_bn=with_bn, groups=groups)
        self.conv1 = sepConv4d(in_planes, out_planes, stride, with_bn=with_bn, full=full, groups=groups)
        self.conv2 = sepConv4d(out_planes, out_planes, (1, 1, 1), with_bn=with_bn, full=full, groups=groups)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        if self.downsample:
            x = self.downsample(x)
        out = self.relu2(x + self.conv2(out))
        return out


class butterfly4D(torch.nn.Module):
    """
    butterfly 4d
    """

    def __init__(self, fdima, fdimb, withbn=True, full=True, groups=1):
        super(butterfly4D, self).__init__()
        self.proj = nn.Sequential(projfeat4d(fdima, fdimb, 1, with_bn=withbn, groups=groups), nn.ReLU(inplace=True))
        self.conva1 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(2, 1, 1), full=full, groups=groups)
        self.conva2 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(2, 1, 1), full=full, groups=groups)
        self.convb3 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(1, 1, 1), full=full, groups=groups)
        self.convb2 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(1, 1, 1), full=full, groups=groups)
        self.convb1 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(1, 1, 1), full=full, groups=groups)

    def forward(self, x):
        out = self.proj(x)
        b, c, u, v, h, w = out.shape
        out1 = self.conva1(out)
        _, c1, u1, v1, h1, w1 = out1.shape
        out2 = self.conva2(out1)
        _, c2, u2, v2, h2, w2 = out2.shape
        out2 = self.convb3(out2)
        tout1 = F.interpolate(out2.view(b, c, u2, v2, -1), (u1, v1, h2 * w2), mode='trilinear', align_corners=False).view(b, c, u1, v1, h2, w2)
        tout1 = F.interpolate(tout1.view(b, c, -1, h2, w2), (u1 * v1, h1, w1), mode='trilinear', align_corners=False).view(b, c, u1, v1, h1, w1)
        out1 = tout1 + out1
        out1 = self.convb2(out1)
        tout = F.interpolate(out1.view(b, c, u1, v1, -1), (u, v, h1 * w1), mode='trilinear', align_corners=False).view(b, c, u, v, h1, w1)
        tout = F.interpolate(tout.view(b, c, -1, h1, w1), (u * v, h, w), mode='trilinear', align_corners=False).view(b, c, u, v, h, w)
        out = tout + out
        out = self.convb1(out)
        return out


class conv2DBatchNorm(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()
        bias = not with_bn
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=1)
        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()
        bias = not with_bn
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=1)
        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)), nn.LeakyReLU(0.1, inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.LeakyReLU(0.1, inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None, dilation=1, with_bn=True):
        super(residualBlock, self).__init__()
        if dilation > 1:
            padding = dilation
        else:
            padding = 1
        if with_bn:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3, stride, padding, dilation=dilation)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1)
        else:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3, stride, padding, dilation=dilation, with_bn=False)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, with_bn=False)
        self.downsample = downsample
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        out = self.convbnrelu1(x)
        out = self.convbn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class pyramidPooling(nn.Module):

    def __init__(self, in_channels, with_bn=True, levels=4):
        super(pyramidPooling, self).__init__()
        self.levels = levels
        self.paths = []
        for i in range(levels):
            self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, with_bn=with_bn))
        self.path_module_list = nn.ModuleList(self.paths)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h, w = x.shape[2:]
        k_sizes = []
        strides = []
        for pool_size in np.linspace(1, min(h, w) // 2, self.levels, dtype=int):
            k_sizes.append((int(h / pool_size), int(w / pool_size)))
            strides.append((int(h / pool_size), int(w / pool_size)))
        k_sizes = k_sizes[::-1]
        strides = strides[::-1]
        pp_sum = x
        for i, module in enumerate(self.path_module_list):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            pp_sum = pp_sum + 1.0 / self.levels * out
        pp_sum = self.relu(pp_sum / 2.0)
        return pp_sum


class pspnet(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """

    def __init__(self, is_proj=True, groups=1):
        super(pspnet, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16, padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16, padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32, padding=1, stride=1)
        self.res_block3 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block5 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block6 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block7 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False), conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128, padding=1, stride=1)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False), conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128, padding=1, stride=1)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False), conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1)
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False), conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32, padding=1, stride=1))
        self.iconv2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64, padding=1, stride=1)
        if self.is_proj:
            self.proj6 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj5 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj4 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1, n_filters=64 // groups, padding=0, stride=1)
            self.proj2 = conv2DBatchNormRelu(in_channels=64, k_size=1, n_filters=64 // groups, padding=0, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)
        pool1 = F.max_pool2d(conv1, 3, 2, 1)
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)
        conv6x = F.interpolate(conv6, [conv5.size()[2], conv5.size()[3]], mode='bilinear', align_corners=False)
        concat5 = torch.cat((conv5, self.upconv6[1](conv6x)), dim=1)
        conv5 = self.iconv5(concat5)
        conv5x = F.interpolate(conv5, [conv4.size()[2], conv4.size()[3]], mode='bilinear', align_corners=False)
        concat4 = torch.cat((conv4, self.upconv5[1](conv5x)), dim=1)
        conv4 = self.iconv4(concat4)
        conv4x = F.interpolate(conv4, [rconv3.size()[2], rconv3.size()[3]], mode='bilinear', align_corners=False)
        concat3 = torch.cat((rconv3, self.upconv4[1](conv4x)), dim=1)
        conv3 = self.iconv3(concat3)
        conv3x = F.interpolate(conv3, [pool1.size()[2], pool1.size()[3]], mode='bilinear', align_corners=False)
        concat2 = torch.cat((pool1, self.upconv3[1](conv3x)), dim=1)
        conv2 = self.iconv2(concat2)
        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
            proj2 = self.proj2(conv2)
            return proj6, proj5, proj4, proj3, proj2
        else:
            return conv6, conv5, conv4, conv3, conv2


class pspnet_s(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """

    def __init__(self, is_proj=True, groups=1):
        super(pspnet_s, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16, padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16, padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32, padding=1, stride=1)
        self.res_block3 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block5 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block6 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block7 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False), conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128, padding=1, stride=1)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False), conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128, padding=1, stride=1)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False), conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64, padding=1, stride=1)
        if self.is_proj:
            self.proj6 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj5 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj4 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1, n_filters=64 // groups, padding=0, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)
        pool1 = F.max_pool2d(conv1, 3, 2, 1)
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)
        conv6x = F.interpolate(conv6, [conv5.size()[2], conv5.size()[3]], mode='bilinear', align_corners=False)
        concat5 = torch.cat((conv5, self.upconv6[1](conv6x)), dim=1)
        conv5 = self.iconv5(concat5)
        conv5x = F.interpolate(conv5, [conv4.size()[2], conv4.size()[3]], mode='bilinear', align_corners=False)
        concat4 = torch.cat((conv4, self.upconv5[1](conv5x)), dim=1)
        conv4 = self.iconv4(concat4)
        conv4x = F.interpolate(conv4, [rconv3.size()[2], rconv3.size()[3]], mode='bilinear', align_corners=False)
        concat3 = torch.cat((rconv3, self.upconv4[1](conv4x)), dim=1)
        conv3 = self.iconv3(concat3)
        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
            return proj6, proj5, proj4, proj3
        else:
            return conv6, conv5, conv4, conv3


class WarpModule(nn.Module):
    """
    taken from https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
    """

    def __init__(self):
        super(WarpModule, self).__init__()
        self.create_grid([1, 1, 1])

    def create_grid(self, size):
        B, W, H = size
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        self.grid = torch.cat((xx, yy), 1).float()

    def forward(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        if B != self.grid.shape[0] or H != self.grid.shape[2] or W != self.grid.shape[3]:
            self.create_grid((B, W, H))
        vgrid = self.grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = (vgrid[:, :, :, 0].abs() < 1) * (vgrid[:, :, :, 1].abs() < 1) > 0
        return output * mask.unsqueeze(1).float(), mask


class VCNLoss(nn.Module):

    def __init__(self, args):
        super(VCNLoss, self).__init__()
        self.maxdisp = args.maxdisp
        self.fac = args.fac
        self.bs = args.train_batch_size
        self.warpx = WarpModule()

    def forward(self, preds, inputs):
        flowl0 = inputs['flows'][:, 0].clone()
        mask = (inputs['valids'][:, 0, 0] == 1) & (flowl0[:, 0].abs() < self.maxdisp) & (flowl0[:, 1].abs() < self.maxdisp // self.fac)
        mask = mask.detach().clone()
        im = inputs['images']
        im = im.view(im.shape[0] * im.shape[1], im.shape[2], im.shape[3], im.shape[4])
        im_warp, _ = self.warpx(im[self.bs:], flowl0[:, :2])
        occ_mask = (im_warp - im[:self.bs]).norm(dim=1) > 0.3
        m = 64.0
        loss = 0
        ws = [0.25, 0.25, 0.25, 0.5, 1.0]
        for i in range(len(ws)):
            loss += ws[i] * torch.norm(preds['flow_preds'][i] * m - flowl0[:, :2], 0, 1)[mask].mean()
            m /= 2
        im_size = flowl0.shape[2:4]
        m = 32.0
        up_flows = preds['up_flows_preds']
        for i in range(len(up_flows)):
            up_flows[i] = F.interpolate(up_flows[i], [im_size[0], im_size[1]], mode='bilinear', align_corners=False) * m
            m /= 2
        up_flows.insert(0, 0)
        oors = preds['oors_preds']
        for i in range(len(oors)):
            oors[i] = F.interpolate(oors[i][:, np.newaxis], [im_size[0], im_size[1]], mode='bilinear', align_corners=False)[:, 0]
        m = 64
        for i in range(len(oors)):
            loss += self.get_oor_loss(flowl0[:, :2] - up_flows[i], oors[i], m * preds['flow_reg_maxs_preds'][i], occ_mask)
            m /= 2
        return loss

    def get_oor_loss(self, flowl0, oor3, maxdisp, occ_mask):
        """ 
        return out-of-range loss
        """
        oor3_gt = (flowl0.abs() > maxdisp).detach()
        oor3_gt = ((oor3_gt.sum(1) > 0) + occ_mask > 0).float()
        weights = oor3_gt.sum().float() / (oor3_gt.shape[0] * oor3_gt.shape[1] * oor3_gt.shape[2])
        weights = oor3_gt * (1 - weights) + (1 - oor3_gt) * weights
        loss_oor3 = F.binary_cross_entropy_with_logits(oor3, oor3_gt, size_average=True, weight=weights)
        return loss_oor3


class flow_reg(nn.Module):
    """
    Soft winner-take-all that selects the most likely diplacement.
    Set ent=True to enable entropy output.
    Set maxdisp to adjust maximum allowed displacement towards one side.
        maxdisp=4 searches for a 9x9 region.
    Set fac to squeeze search window.
        maxdisp=4 and fac=2 gives search window of 9x5
    """

    def __init__(self, ent=False, maxdisp=int(4), fac=1):
        super(flow_reg, self).__init__()
        self.ent = ent
        self.md = maxdisp
        self.fac = fac
        self.truncated = True
        self.wsize = 3
        self.create_flow([1, 1, 1])
        self.pool3d = nn.MaxPool3d((self.wsize * 2 + 1, self.wsize * 2 + 1, 1), stride=1, padding=(self.wsize, self.wsize, 0))

    def create_flow(self, size):
        B, W, H = size
        flowrangey = range(-self.md, self.md + 1)
        flowrangex = range(-int(self.md // self.fac), int(self.md // self.fac) + 1)
        meshgrid = np.meshgrid(flowrangex, flowrangey)
        flowy = np.tile(np.reshape(meshgrid[0], [1, 2 * self.md + 1, 2 * int(self.md // self.fac) + 1, 1, 1]), (B, 1, 1, H, W))
        flowx = np.tile(np.reshape(meshgrid[1], [1, 2 * self.md + 1, 2 * int(self.md // self.fac) + 1, 1, 1]), (B, 1, 1, H, W))
        self.flowx = torch.Tensor(flowx)
        self.flowy = torch.Tensor(flowy)

    def forward(self, x):
        b, u, v, h, w = x.shape
        oldx = x
        if b != self.flowx.shape[0] or h != self.flowx.shape[3] or w != self.flowx.shape[4]:
            self.create_flow((b, w, h))
        if self.truncated:
            x = x.view(b, u * v, h, w)
            idx = x.argmax(1)[:, np.newaxis]
            if x.is_cuda:
                mask = Variable(torch.HalfTensor(b, u * v, h, w)).fill_(0)
            else:
                mask = Variable(torch.FloatTensor(b, u * v, h, w)).fill_(0)
            mask.scatter_(1, idx, 1)
            mask = mask.view(b, 1, u, v, -1)
            mask = self.pool3d(mask)[:, 0].view(b, u, v, h, w)
            ninf = x.clone().fill_(-np.inf).view(b, u, v, h, w)
            x = torch.where(mask.bool(), oldx, ninf)
        else:
            self.wsize = (np.sqrt(u * v) - 1) / 2
        b, u, v, h, w = x.shape
        x = F.softmax(x.view(b, -1, h, w), 1).view(b, u, v, h, w)
        outx = torch.sum(torch.sum(x * self.flowx, 1), 1, keepdim=True)
        outy = torch.sum(torch.sum(x * self.flowy, 1), 1, keepdim=True)
        if self.ent:
            local_entropy = (-x * torch.clamp(x, 1e-09, 1 - 1e-09).log()).sum(1).sum(1)[:, np.newaxis]
            if self.wsize == 0:
                local_entropy[:] = 1.0
            else:
                local_entropy /= np.log((self.wsize * 2 + 1) ** 2)
            x = F.softmax(oldx.view(b, -1, h, w), 1).view(b, u, v, h, w)
            global_entropy = (-x * torch.clamp(x, 1e-09, 1 - 1e-09).log()).sum(1).sum(1)[:, np.newaxis]
            global_entropy /= np.log(x.shape[1] * x.shape[2])
            return torch.cat([outx, outy], 1), torch.cat([local_entropy, global_entropy], 1)
        else:
            return torch.cat([outx, outy], 1), None


def iter_spatial_correlation_sample(input1: torch.Tensor, input2: torch.Tensor, kernel_size: Union[int, Tuple[int, int]]=1, patch_size: Union[int, Tuple[int, int]]=1, stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0, dilation: Union[int, Tuple[int, int]]=1, dilation_patch: Union[int, Tuple[int, int]]=1) ->torch.Tensor:
    """Apply spatial correlation sampling from input1 to input2 using iteration in PyTorch.

    This docstring is taken and adapted from the original package.

    Every parameter except input1 and input2 can be either single int or a pair of int. For more information about
    Spatial Correlation Sampling, see this page. https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/

    Parameters
    ----------
    input1 : torch.Tensor
        The origin feature map.
    input2 : torch.Tensor
        The target feature map.
    kernel_size : Union[int, Tuple[int, int]], default 1
        Total size of your correlation kernel, in pixels
    patch_size : Union[int, Tuple[int, int]], default 1
        Total size of your patch, determining how many different shifts will be applied.
    stride : Union[int, Tuple[int, int]], default 1
        Stride of the spatial sampler, will modify output height and width.
    padding : Union[int, Tuple[int, int]], default 0
        Padding applied to input1 and input2 before applying the correlation sampling, will modify output height and width.
    dilation : Union[int, Tuple[int, int]], default 1
        Similar to dilation in convolution.
    dilation_patch : Union[int, Tuple[int, int]], default 1
        Step for every shift in patch.

    Returns
    -------
    torch.Tensor
        Result of correlation sampling.

    Raises
    ------
    NotImplementedError
        If kernel_size != 1.
    NotImplementedError
        If dilation != 1.
    """
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    dilation_patch = (dilation_patch, dilation_patch) if isinstance(dilation_patch, int) else dilation_patch
    if kernel_size[0] != 1 or kernel_size[1] != 1:
        raise NotImplementedError('Only kernel_size=1 is supported.')
    if dilation[0] != 1 or dilation[1] != 1:
        raise NotImplementedError('Only dilation=1 is supported.')
    if patch_size[0] % 2 == 0 or patch_size[1] % 2 == 0:
        raise NotImplementedError('Only odd patch sizes are supperted.')
    if max(padding) > 0:
        input1 = F.pad(input1, (padding[1], padding[1], padding[0], padding[0]))
        input2 = F.pad(input2, (padding[1], padding[1], padding[0], padding[0]))
    max_displacement = dilation_patch[0] * (patch_size[0] - 1) // 2, dilation_patch[1] * (patch_size[1] - 1) // 2
    input2 = F.pad(input2, (max_displacement[1], max_displacement[1], max_displacement[0], max_displacement[0]))
    b, _, h, w = input1.shape
    input1 = input1[:, :, ::stride[0], ::stride[1]]
    sh, sw = input1.shape[2:4]
    corr = torch.zeros(b, patch_size[0], patch_size[1], sh, sw)
    for i in range(0, 2 * max_displacement[0] + 1, dilation_patch[0]):
        for j in range(0, 2 * max_displacement[1] + 1, dilation_patch[1]):
            p2 = input2[:, :, i:i + h, j:j + w]
            p2 = p2[:, :, ::stride[0], ::stride[1]]
            corr[:, i // dilation_patch[0], j // dilation_patch[1]] = (input1 * p2).sum(dim=1)
    return corr


class IterSpatialCorrelationSampler(nn.Module):
    """Apply spatial correlation sampling from two inputs using iteration in PyTorch."""

    def __init__(self, kernel_size: Union[int, Tuple[int, int]]=1, patch_size: Union[int, Tuple[int, int]]=1, stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0, dilation: Union[int, Tuple[int, int]]=1, dilation_patch: Union[int, Tuple[int, int]]=1) ->None:
        """Initialize IterSpatialCorrelationSampler.

        Parameters
        ----------
        kernel_size : Union[int, Tuple[int, int]], default 1
            Total size of your correlation kernel, in pixels
        patch_size : Union[int, Tuple[int, int]], default 1
            Total size of your patch, determining how many different shifts will be applied.
        stride : Union[int, Tuple[int, int]], default 1
            Stride of the spatial sampler, will modify output height and width.
        padding : Union[int, Tuple[int, int]], default 0
            Padding applied to input1 and input2 before applying the correlation sampling, will modify output height and width.
        dilation : Union[int, Tuple[int, int]], default 1
            Similar to dilation in convolution.
        dilation_patch : Union[int, Tuple[int, int]], default 1
            Step for every shift in patch.
        """
        super(IterSpatialCorrelationSampler, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) ->torch.Tensor:
        """Compute the correlation sampling from input1 to input2.

        Parameters
        ----------
        input1 : torch.Tensor
            The origin feature map.
        input2 : torch.Tensor
            The target feature map.

        Returns
        -------
        torch.Tensor
            Result of correlation sampling.
        """
        return iter_spatial_correlation_sample(input1=input1, input2=input2, kernel_size=self.kernel_size, patch_size=self.patch_size, stride=self.stride, padding=self.padding, dilation=self.dilation, dilation_patch=self.dilation_patch)

