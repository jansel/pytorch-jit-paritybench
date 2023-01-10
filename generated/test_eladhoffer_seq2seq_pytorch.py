import sys
_module = sys.modules[__name__]
del sys
eval = _module
main = _module
seq2seq = _module
datasets = _module
coco_caption = _module
coco_captions = _module
concept_captions = _module
iwslt = _module
multi_language = _module
open_subtitles = _module
text = _module
vision = _module
wmt = _module
models = _module
bytenet = _module
conv = _module
img2seq = _module
modules = _module
attention = _module
conv = _module
linear = _module
prevasive_densenet = _module
prevasive_resnet = _module
recurrent = _module
state = _module
transformer_blocks = _module
vision_encoders = _module
weight_drop = _module
weight_norm = _module
prevasive = _module
recurrent = _module
seq2seq_base = _module
seq2seq_generic = _module
transformer = _module
test_batch_loader = _module
test_tokenizer = _module
tools = _module
beam_search = _module
config = _module
inference = _module
tokenizer = _module
trainer = _module
setup = _module
translate = _module

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


import logging


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.distributed as dist


from torch.utils.data.distributed import DistributedSampler


import string


from random import randrange


from collections import OrderedDict


import torch


from torch.nn.utils.rnn import pack_padded_sequence


import torchvision.datasets as dset


import torchvision.transforms as transforms


from torchvision.datasets.folder import default_loader


from copy import copy


from copy import deepcopy


from torch.utils.data import Dataset


import warnings


import torch.nn.functional as F


import math


from torch.nn.utils.rnn import PackedSequence


from torchvision.models import resnet


from torchvision.models import densenet


from torchvision.models import vgg


from torchvision.models import alexnet


from torchvision.models import squeezenet


from torch.nn import Parameter


from functools import wraps


from torch.nn.utils.rnn import pad_packed_sequence as unpack


from torch.nn.utils.rnn import pack_padded_sequence as pack


from torch.nn.parallel import data_parallel


from torch.nn.functional import log_softmax


from random import shuffle


from math import floor


from torch.nn.functional import adaptive_avg_pool2d


from collections import Counter


import time


from itertools import chain


from itertools import cycle


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


from torch.nn.utils import clip_grad_norm_


import numpy as np


class MaskedConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, bias=True, causal=True):
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = (kernel_size - 1) * dilation // 2
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        output = super(MaskedConv1d, self).forward(inputs)
        return output[:, :, :inputs.size(2)]


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, interm_channels=None, out_channels=None, kernel_size=3, dilation=1, causal=True):
        super(ResidualBlock, self).__init__()
        out_channels = out_channels or in_channels
        interm_channels = interm_channels or in_channels // 2
        self.layernorm1 = nn.LayerNorm(in_channels)
        self.layernorm2 = nn.LayerNorm(interm_channels)
        self.layernorm3 = nn.LayerNorm(interm_channels)
        self.conv1 = nn.Conv1d(in_channels, interm_channels, 1)
        self.conv2 = MaskedConv1d(interm_channels, interm_channels, kernel_size, dilation=dilation, causal=causal)
        self.conv3 = nn.Conv1d(interm_channels, out_channels, 1)
        self.relu = nn.ReLU(True)

    def forward(self, inputs):
        out = self.layernorm1(inputs)
        out = self.conv1(self.relu(out))
        out = self.layernorm2(out)
        out = self.conv2(self.relu(out))
        out = self.layernorm3(out)
        out = self.conv3(self.relu(out))
        out += inputs
        return out


class ByteNet(nn.Sequential):

    def __init__(self, num_channels=512, num_sets=6, dilation_rates=[1, 2, 4, 8, 16], kernel_size=3, block=ResidualBlock, causal=True):
        super(ByteNet, self).__init__()
        for s in range(num_sets):
            for r in dilation_rates:
                self.add_module('block%s_%s' % (s, r), block(num_channels, kernel_size=kernel_size, dilation=r, causal=causal))


class GatedConv1d(MaskedConv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, bias=True, causal=True):
        super(GatedConv1d, self).__init__(in_channels, 2 * out_channels, kernel_size, dilation, groups, bias, causal)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        output = super(GatedConv1d, self).forward(inputs)
        mask, output = output.chunk(2, 1)
        mask = self.sigmoid(mask)
        return output * mask


class StackedConv(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=4, bias=True, dropout=0, causal=True):
        super(StackedConv, self).__init__()
        self.convs = nn.ModuleList()
        size = input_size
        for l in range(num_layers):
            self.convs.append(GatedConv1d(size, hidden_size, 1, bias=bias, causal=False))
            self.convs.append(nn.BatchNorm1d(hidden_size))
            self.convs.append(MaskedConv1d(hidden_size, hidden_size, kernel_size, bias=bias, groups=hidden_size, causal=causal))
            self.convs.append(nn.BatchNorm1d(hidden_size))
            size = hidden_size

    def forward(self, x):
        res = None
        for conv in self.convs:
            x = conv(x)
            if res is not None:
                x = x + res
            res = x
        return x


class ConvEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, kernel_size=3, num_layers=4, bias=True, dropout=0, causal=False):
        super(ConvEncoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.convs = StackedConv(hidden_size, hidden_size, kernel_size, num_layers, bias, causal=causal)

    def forward(self, inputs):
        x = self.embedder(inputs)
        x = x.transpose(1, 2)
        x = self.convs(x)
        return x


class ConvDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, kernel_size=3, num_layers=4, bias=True, dropout=0, causal=True):
        super(ConvDecoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.convs = StackedConv(2 * hidden_size, hidden_size, kernel_size, num_layers, bias, causal=causal)

    def forward(self, inputs, state):
        x = self.embedder(inputs)
        x = x.transpose(1, 2)
        state = F.adaptive_avg_pool1d(state, x.size(2))
        x = torch.cat([x, state], 1)
        x = self.convs(x)
        x = x.transpose(1, 2)
        x = x.contiguous().view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)
        return x


class AttentionLayer(nn.Module):
    """
    Attention layer according to https://arxiv.org/abs/1409.0473.

    Params:
      num_units: Number of units used in the attention layer
    """

    def __init__(self, query_size, key_size, value_size=None, mode='bahdanau', normalize=False, dropout=0, batch_first=False, weight_norm=False, bias=True, query_transform=True, output_transform=True, output_nonlinearity='tanh', output_size=None):
        super(AttentionLayer, self).__init__()
        assert mode == 'bahdanau' or mode == 'dot_prod'
        value_size = value_size or key_size
        self.mode = mode
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.normalize = normalize
        wn_func = wn if weight_norm else lambda x: x
        if mode == 'bahdanau':
            self.linear_att = nn.Linear(key_size, 1, bias=bias)
            if normalize:
                self.linear_att = nn.utils.weight_norm(self.linear_att)
        elif normalize:
            self.scale = nn.Parameter(torch.Tensor([1]))
        if output_transform:
            output_size = output_size or query_size
            self.linear_out = wn_func(nn.Linear(query_size + value_size, output_size, bias=bias))
            self.output_size = output_size
        else:
            self.output_size = value_size
        if query_transform:
            self.linear_q = wn_func(nn.Linear(query_size, key_size, bias=bias))
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.output_nonlinearity = output_nonlinearity
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask
        if mask is not None and not self.batch_first:
            self.mask = self.mask.t()

    def calc_score(self, att_query, att_keys):
        """
        att_query is: b x t_q x n
        att_keys is b x t_k x n
        return b x t_q x t_k scores
        """
        b, t_k, n = list(att_keys.size())
        t_q = att_query.size(1)
        if self.mode == 'bahdanau':
            att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
            att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
            sum_qk = att_query + att_keys
            sum_qk = sum_qk.view(b * t_k * t_q, n)
            out = self.linear_att(F.tanh(sum_qk)).view(b, t_q, t_k)
        elif self.mode == 'dot_prod':
            out = torch.bmm(att_query, att_keys.transpose(1, 2))
            if hasattr(self, 'scale'):
                out = out * self.scale
        return out

    def forward(self, query, keys, values=None):
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if values is not None:
                values = values.transpose(0, 1)
            if query.dim() == 3:
                query = query.transpose(0, 1)
        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False
        values = keys if values is None else values
        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)
        if hasattr(self, 'linear_q'):
            att_query = self.linear_q(query)
        else:
            att_query = query
        scores = self.calc_score(att_query, keys)
        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
            scores.masked_fill_(mask, -1000000000000.0)
        scores_normalized = F.softmax(scores, dim=2)
        scores_normalized = self.dropout(scores_normalized)
        context = torch.bmm(scores_normalized, values)
        if hasattr(self, 'linear_out'):
            context = self.linear_out(torch.cat([query, context], 2))
            if self.output_nonlinearity == 'tanh':
                context = F.tanh(context)
            elif self.output_nonlinearity == 'relu':
                context = F.relu(context, inplace=True)
        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)
        return context, scores_normalized


class OrderAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, dropout=0, causal=False):
        super(OrderAttention, self).__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.mask_q = None
        self.mask_k = None

    def set_mask_q(self, masked_tq):
        self.mask_q = masked_tq

    def set_mask_k(self, masked_tk):
        self.mask_k = masked_tk

    def forward(self, q, k):
        b_q, t_q, dim_q = list(q.size())
        b_k, t_k, dim_k = list(k.size())
        assert dim_q == dim_k
        b = b_q
        qk = torch.bmm(q, k.transpose(1, 2))
        qk = qk / dim_k ** 0.5
        mask = None
        with torch.no_grad():
            if self.causal and t_q > 1:
                causal_mask = q.data.new(t_q, t_k).byte().fill_(1).triu_(1)
                mask = causal_mask.unsqueeze(0).expand(b, t_q, t_k)
            if self.mask_k is not None:
                mask_k = self.mask_k.unsqueeze(1).expand(b, t_q, t_k)
                mask = mask_k if mask is None else mask | mask_k
            if self.mask_q is not None:
                mask_q = self.mask_q.unsqueeze(2).expand(b, t_q, t_k)
                mask = mask_q if mask is None else mask | mask_q
        if mask is not None:
            qk.masked_fill_(mask, float('-inf'))
        return F.softmax(qk, dim=2, dtype=torch.float32 if qk.dtype == torch.float16 else qk.dtype)


class SDPAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, dropout=0, causal=False, gumbel=False):
        super(SDPAttention, self).__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.mask_q = None
        self.mask_k = None
        self.gumbel = gumbel

    def set_mask_q(self, masked_tq):
        self.mask_q = masked_tq

    def set_mask_k(self, masked_tk):
        self.mask_k = masked_tk

    def forward(self, q, k, v):
        b_q, t_q, dim_q = list(q.size())
        b_k, t_k, dim_k = list(k.size())
        b_v, t_v, dim_v = list(v.size())
        assert b_q == b_k and b_k == b_v
        assert dim_q == dim_k
        assert t_k == t_v
        b = b_q
        qk = torch.bmm(q, k.transpose(1, 2))
        qk = qk / dim_k ** 0.5
        mask = None
        with torch.no_grad():
            if self.causal and t_q > 1:
                causal_mask = q.data.new(t_q, t_k).byte().fill_(1).triu_(1)
                mask = causal_mask.unsqueeze(0).expand(b, t_q, t_k)
            if self.mask_k is not None:
                mask_k = self.mask_k.unsqueeze(1).expand(b, t_q, t_k)
                mask = mask_k if mask is None else mask | mask_k
            if self.mask_q is not None:
                mask_q = self.mask_q.unsqueeze(2).expand(b, t_q, t_k)
                mask = mask_q if mask is None else mask | mask_q
        if mask is not None:
            qk.masked_fill_(mask, -1000000000000.0)
        if self.gumbel:
            sm_qk = F.gumbel_softmax(qk, dim=2, hard=True)
        else:
            sm_qk = F.softmax(qk, dim=2, dtype=torch.float32 if qk.dtype == torch.float16 else qk.dtype)
        sm_qk = self.dropout(sm_qk)
        return torch.bmm(sm_qk, v), sm_qk


def _sum_tensor_scalar(tensor, scalar, expand_size):
    if scalar is not None:
        scalar = scalar.expand(expand_size).contiguous()
    else:
        return tensor
    if tensor is None:
        return scalar
    return tensor + scalar


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, groups=1, multiplier=False, pre_bias=False, post_bias=False):
        if in_features % groups != 0:
            raise ValueError('in_features must be divisible by groups')
        if out_features % groups != 0:
            raise ValueError('out_features must be divisible by groups')
        self.groups = groups
        super(Linear, self).__init__(in_features, out_features // self.groups, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.full((out_features,), 0))
        else:
            self.register_parameter('bias', None)
        if pre_bias:
            self.pre_bias = nn.Parameter(torch.tensor([0.0]))
        else:
            self.register_parameter('pre_bias', None)
        if post_bias:
            self.post_bias = nn.Parameter(torch.tensor([0.0]))
        else:
            self.register_parameter('post_bias', None)
        if multiplier:
            self.multiplier = nn.Parameter(torch.tensor([1.0]))
        else:
            self.register_parameter('multiplier', None)

    def forward(self, x):
        if self.pre_bias is not None:
            x = x + self.pre_bias
        weight = self.weight if self.multiplier is None else self.weight * self.multiplier
        bias = _sum_tensor_scalar(self.bias, self.post_bias, self.out_features)
        if self.groups == 1:
            out = F.linear(x, weight, bias)
        else:
            x_g = x.chunk(self.groups, dim=-1)
            w_g = weight.chunk(self.groups, dim=-1)
            out = torch.cat([F.linear(x_g[i], w_g[i]) for i in range(self.groups)], -1)
            if bias is not None:
                out += bias
        return out


class MultiHeadAttentionV2(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, input_size, output_size, num_heads, weight_norm=False, groups=1, dropout=0, causal=False, add_bias_kv=False):
        super(MultiHeadAttentionV2, self).__init__()
        assert input_size % num_heads == 0
        wn_func = wn if weight_norm else lambda x: x
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.linear_q = wn_func(Linear(input_size, input_size, bias=False, groups=groups))
        self.linear_k = wn_func(Linear(input_size, input_size, bias=add_bias_kv, groups=groups))
        self.linear_v = wn_func(Linear(input_size, input_size, bias=add_bias_kv, groups=groups))
        self.linear_out = wn_func(Linear(input_size, output_size, groups=groups))
        self.sdp_attention = SDPAttention(dropout=dropout, causal=causal)

    def set_mask_q(self, masked_tq):
        self.sdp_attention.set_mask_q(masked_tq)

    def set_mask_k(self, masked_tk):
        self.sdp_attention.set_mask_k(masked_tk)

    def forward(self, q, k, v):
        b_q, t_q, dim_q = list(q.size())
        b_k, t_k, dim_k = list(k.size())
        b_v, t_v, dim_v = list(v.size())
        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)
        qw = qw.chunk(self.num_heads, 2)
        kw = kw.chunk(self.num_heads, 2)
        vw = vw.chunk(self.num_heads, 2)
        output = []
        attention_scores = []
        for i in range(self.num_heads):
            out_h, score = self.sdp_attention(qw[i], kw[i], vw[i])
            output.append(out_h)
            attention_scores.append(score)
        output = torch.cat(output, 2)
        return self.linear_out(output), attention_scores


class MultiHeadAttention(nn.MultiheadAttention):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, input_size, output_size, num_heads, dropout=0, causal=False, bias=True, add_bias_kv=False, add_zero_attn=False, batch_first=True, groups=None, weight_norm=None):
        super(MultiHeadAttention, self).__init__(input_size, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        assert input_size % num_heads == 0
        assert input_size == output_size
        self.causal = causal
        self.batch_first = batch_first

    def set_mask_q(self, masked_tq):
        self.mask_q = masked_tq

    def set_mask_k(self, masked_tk):
        self.mask_k = masked_tk

    def forward(self, query, key, value, need_weights=False, static_kv=False):
        key_padding_mask = attn_mask = None
        time_dim = 1 if self.batch_first else 0
        t_q = query.size(time_dim)
        t_k = key.size(time_dim)
        with torch.no_grad():
            if self.causal and t_q > 1:
                attn_mask = torch.full((t_q, t_k), float('-inf'), device=query.device, dtype=query.dtype).triu_(1)
            key_padding_mask = self.mask_k
        if self.batch_first:
            qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
            kv_same = key.data_ptr() == value.data_ptr()
            key = key.transpose(0, 1)
            if kv_same:
                value = key
            else:
                value = value.transpose(0, 1)
            if qkv_same:
                query = key
            else:
                query = query.transpose(0, 1)
        elif key_padding_mask is not None:
            key_padding_mask.t()
        attn_output, attn_output_weights = super(MultiHeadAttention, self).forward(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=need_weights)
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_output_weights


class MaskedConv2d(nn.Conv2d):
    """masked over 3rd dimension (2nd spatial dimension)"""

    def __init__(self, *kargs, **kwargs):
        super(MaskedConv2d, self).__init__(*kargs, **kwargs)
        self.masked_dim = 1

        def pad_needed(causal, size, stride, pad, dilation):
            if not causal:
                return 0
            else:
                return (size - 1) * dilation
        add_padding = pad_needed(self.masked_dim == 0, self.kernel_size[0], self.stride[0], self.padding[0], self.dilation[0]), pad_needed(self.masked_dim == 1, self.kernel_size[1], self.stride[1], self.padding[1], self.dilation[1])
        self.padding = self.padding[0] + add_padding[0] // 2, self.padding[1] + add_padding[1] // 2
        self.remove_padding = add_padding

    def forward(self, inputs):
        output = super(MaskedConv2d, self).forward(inputs)
        return output[:, :, :output.size(2) - self.remove_padding[0], :output.size(3) - self.remove_padding[1]]


class TimeNorm2d(nn.InstanceNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(TimeNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, inputs):
        B, C, T1, T2 = inputs.shape
        x = inputs.transpose(0, 3)
        y = super(TimeNorm2d, self).forward(x)
        y = y.transpose(0, 3)
        return y


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=True)
        self.conv2 = MaskedConv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate

    def forward(self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)
        x = self.relu(x)
        new_features, _ = self.conv2(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([inputs, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=True))


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=(3, 3), stride=1, expansion=4, downsample=None, groups=1, residual_block=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, 3, padding=1, stride=(stride, 1), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x, cache=None):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out, cache = self.conv2(out, cache)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        if self.residual_block is not None:
            residual = self.residual_block(residual)
        out += residual
        out = self.relu(out)
        return out


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, MaskedConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, input_size, output_size, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseNet, self).__init__()
        num_init_features = input_size
        self.features = nn.Sequential()
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.conv2 = nn.Conv2d(num_features, output_size, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        init_model(self)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, input_size, output_size, inplanes=128, strided=True, block=Bottleneck, residual_block=None, layers=[1, 1, 1, 1], width=[128, 128, 128, 128], expansion=2, groups=[1, 1, 1, 1]):
        super(ResNet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = MaskedConv2d(input_size, self.inplanes, kernel_size=7, stride=(2, 1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        for i in range(len(layers)):
            if strided:
                stride = 1 if i == 0 else 2
            else:
                stride = 1
            setattr(self, 'layer%s' % str(i + 1), self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion, stride=stride, residual_block=residual_block, groups=groups[i]))
        self.conv2 = nn.Conv2d(self.inplanes, output_size, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU(inplace=True)
        init_model(self)

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, out_planes, kernel_size=1, stride=(stride, 1), bias=True), nn.BatchNorm2d(planes * expansion))
        if residual_block is not None:
            residual_block = residual_block(out_planes)
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, expansion=expansion, downsample=downsample, groups=groups, residual_block=residual_block))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups, residual_block=residual_block))
        return nn.Sequential(*layers)

    def forward(self, x):
        x, _ = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class StackedRecurrent(nn.Sequential):

    def __init__(self, dropout=0, residual=False):
        super(StackedRecurrent, self).__init__()
        self.residual = residual
        self.dropout = dropout

    def forward(self, inputs, hidden=None):
        hidden = hidden or tuple([None] * len(self))
        next_hidden = []
        for i, module in enumerate(self._modules.values()):
            output, h = module(inputs, hidden[i])
            next_hidden.append(h)
            if self.residual and inputs.size(-1) == output.size(-1):
                inputs = output + inputs
            else:
                inputs = output
            if isinstance(inputs, PackedSequence):
                data = nn.functional.dropout(inputs.data, self.dropout, self.training)
                inputs = PackedSequence(data, inputs.batch_sizes)
            else:
                inputs = nn.functional.dropout(inputs, self.dropout, self.training)
        return output, tuple(next_hidden)


class ConcatRecurrent(nn.Sequential):

    def forward(self, inputs, hidden=None):
        hidden = hidden or tuple([None] * len(self))
        next_hidden = []
        outputs = []
        for i, module in enumerate(self._modules.values()):
            curr_output, h = module(inputs, hidden[i])
            outputs.append(curr_output)
            next_hidden.append(h)
        output = torch.cat(outputs, -1)
        return output, tuple(next_hidden)


class StackedCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, bias=True, rnn_cell=nn.LSTMCell, residual=False, weight_norm=False):
        super(StackedCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.residual = residual
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            rnn = rnn_cell(input_size, hidden_size, bias=bias)
            if weight_norm:
                rnn = wn(rnn_cell)
            self.layers.append(rnn)
            input_size = hidden_size

    def forward(self, inputs, hidden):

        def select_layer(h_state, i):
            if isinstance(h_state, tuple):
                return tuple([select_layer(s, i) for s in h_state])
            else:
                return h_state[i]
        next_hidden = []
        for i, layer in enumerate(self.layers):
            next_hidden_i = layer(inputs, select_layer(hidden, i))
            output = next_hidden_i[0] if isinstance(next_hidden_i, tuple) else next_hidden_i
            if i + 1 < self.num_layers:
                output = self.dropout(output)
            if self.residual and inputs.size(-1) == output.size(-1):
                inputs = output + inputs
            else:
                inputs = output
            next_hidden.append(next_hidden_i)
        if isinstance(hidden, tuple):
            next_hidden = tuple([torch.stack(h) for h in zip(*next_hidden)])
        else:
            next_hidden = torch.stack(next_hidden)
        return inputs, next_hidden


class StackedsAttentionCell(StackedCell):

    def __init__(self, input_size, hidden_size, attention_layer, num_layers=1, dropout=0, bias=True, rnn_cell=nn.LSTMCell, residual=False, weight_norm=False):
        super(StackedsAttentionCell, self).__init__(input_size, hidden_size, num_layers, dropout, bias, rnn_cell, residual)
        self.attention = attention_layer

    def forward(self, input_with_context, hidden, get_attention=False):
        inputs, context = input_with_context
        if isinstance(context, tuple):
            context_keys, context_values = context
        else:
            context_keys = context_values = context
        hidden_cell, hidden_attention = hidden
        inputs = torch.cat([inputs, hidden_attention], inputs.dim() - 1)
        output_cell, hidden_cell = super(StackedsAttentionCell, self).forward(inputs, hidden_cell)
        output, score = self.attention(output_cell, context_keys, context_values)
        if get_attention:
            return output, (hidden_cell, output), score
        else:
            del score
            return output, (hidden_cell, output)


class ZoneOutCell(nn.Module):

    def __init__(self, cell, zoneout_prob=0):
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_prob = zoneout_prob

    def forward(self, inputs, hidden):

        def zoneout(h, next_h, prob):
            if isinstance(h, tuple):
                num_h = len(h)
                if not isinstance(prob, tuple):
                    prob = tuple([prob] * num_h)
                return tuple([zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)])
            mask = h.new_tensor(h.size()).bernoulli_(prob)
            return mask * next_h + (1 - mask) * h
        next_hidden = self.cell(inputs, hidden)
        next_hidden = zoneout(hidden, next_hidden, self.zoneout_prob)
        return next_hidden


class TimeRecurrentCell(nn.Module):

    def __init__(self, cell, batch_first=False, lstm=True, with_attention=False, reverse=False):
        super(TimeRecurrentCell, self).__init__()
        self.cell = cell
        self.lstm = lstm
        self.reverse = reverse
        self.batch_first = batch_first
        self.with_attention = with_attention

    def forward(self, inputs, hidden=None, context=None, mask_attention=None, get_attention=False):
        if self.with_attention and mask_attention is not None:
            self.cell.attention.set_mask(mask_attention)
        hidden_size = self.cell.hidden_size
        batch_dim = 0 if self.batch_first else 1
        time_dim = 1 if self.batch_first else 0
        batch_size = inputs.size(batch_dim)
        if hidden is None:
            num_layers = getattr(self.cell, 'num_layers', 1)
            zero = inputs.data.new(1).zero_()
            h0 = zero.view(1, 1, 1).expand(num_layers, batch_size, hidden_size)
            hidden = h0
            if self.lstm:
                hidden = hidden, h0
        if self.with_attention and (not isinstance(hidden, tuple) or self.lstm and not isinstance(hidden[0], tuple)):
            zero = inputs.data.new(1).zero_()
            attn_size = self.cell.attention.output_size
            a0 = zero.view(1, 1).expand(batch_size, attn_size)
            hidden = hidden, a0
        outputs = []
        attentions = []
        inputs_time = inputs.split(1, time_dim)
        if self.reverse:
            inputs_time.reverse()
        for input_t in inputs_time:
            input_t = input_t.squeeze(time_dim)
            if self.with_attention:
                input_t = input_t, context
            if self.with_attention and get_attention:
                output_t, hidden, attn = self.cell(input_t, hidden, get_attention=True)
                attentions += [attn]
            else:
                output_t, hidden = self.cell(input_t, hidden)
            outputs += [output_t]
        if self.reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, time_dim)
        if get_attention:
            attentions = torch.stack(attentions, time_dim)
            return outputs, hidden, attentions
        else:
            return outputs, hidden


def wrap_stacked_recurrent(recurrent_func, num_layers=1, residual=False, weight_norm=False):

    def f(*kargs, **kwargs):
        module = StackedRecurrent(residual)
        for i in range(num_layers):
            rnn = recurrent_func(*kargs, **kwargs)
            if weight_norm:
                rnn = wn(rnn)
            module.add_module(str(i), rnn)
        return module
    return f


def wrap_zoneout_cell(cell_func, zoneout_prob=0):

    def f(*kargs, **kwargs):
        return ZoneOutCell(cell_func(*kargs, **kwargs), zoneout_prob)
    return f


def Recurrent(mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, weight_norm=False, weight_drop=0, bidirectional=False, residual=False, zoneout=None, attention_layer=None, forget_bias=None):
    params = dict(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    need_to_wrap = attention_layer is not None or zoneout is not None or mode not in ['LSTM', 'GRU', 'RNN', 'iRNN']
    wn_func = wn if weight_norm else lambda x: x
    if need_to_wrap:
        if mode == 'LSTM':
            rnn_cell = nn.LSTMCell
        elif mode == 'GRU':
            rnn_cell = nn.GRUCell
        else:
            raise Exception('Mode {} is unsupported yet'.format(mode))
        cell = rnn_cell
        if zoneout is not None:
            cell = wrap_zoneout_cell(cell, zoneout)
        if bidirectional:
            bi_module = ConcatRecurrent()
            bi_module.add_module('0', TimeRecurrentCell(wn_func(cell(input_size, hidden_size)), batch_first=batch_first, lstm=mode == 'LSTM', with_attention=attention_layer is not None))
            bi_module.add_module('0.reversed', TimeRecurrentCell(wn_func(cell(input_size, hidden_size)), batch_first=batch_first, lstm=mode == 'LSTM', reverse=True, with_attention=attention_layer is not None))
            module = StackedRecurrent(residual)
            for i in range(num_layers):
                module.add_module(str(i), bi_module)
        else:
            if attention_layer is None:
                cell = StackedCell(rnn_cell=cell, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, residual=residual, weight_norm=weight_norm, dropout=dropout)
            else:
                cell = StackedsAttentionCell(rnn_cell=cell, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, residual=residual, weight_norm=weight_norm, dropout=dropout, attention_layer=attention_layer)
            module = TimeRecurrentCell(cell, batch_first=batch_first, lstm=mode == 'LSTM', with_attention=attention_layer is not None)
    else:
        if mode == 'LSTM':
            rnn = nn.LSTM
        elif mode == 'GRU':
            rnn = nn.GRU
        elif mode == 'RNN':
            rnn = nn.RNN
            params['nonlinearity'] = 'tanh'
        elif mode == 'iRNN':
            rnn = nn.RNN
            params['nonlinearity'] = 'relu'
        else:
            raise Exception('Unknown mode: {}'.format(mode))
        if residual:
            rnn = wrap_stacked_recurrent(rnn, num_layers=num_layers, weight_norm=weight_norm, residual=True)
            params['num_layers'] = 1
        if params.get('num_layers', 0) == 1:
            params.pop('dropout', None)
        module = wn_func(rnn(**params))
    if mode == 'LSTM' and forget_bias is not None:
        for n, p in module.named_parameters():
            if 'bias_hh' in n or 'bias_ih' in n:
                forget_bias_params = p.data.chunk(4)[1]
                forget_bias_params.fill_(forget_bias / 2)
    if mode == 'iRNN':
        for n, p in module.named_parameters():
            if 'weight_hh' in n:
                p.detach().copy_(torch.eye(*p.shape))
    return module


class RecurrentAttention(nn.Module):

    def __init__(self, input_size, context_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, concat_attention=True, num_pre_attention_layers=None, mode='LSTM', residual=False, weight_norm=False, attention=None, forget_bias=None):
        super(RecurrentAttention, self).__init__()
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.concat_attention = concat_attention
        self.batch_first = batch_first
        if isinstance(context_size, tuple):
            context_key_size, context_value_size = context_size
        else:
            context_key_size = context_value_size = context_size
        attention = attention or {}
        attention['key_size'] = context_key_size
        attention['value_size'] = context_value_size
        attention['query_size'] = hidden_size
        attention['batch_first'] = batch_first
        attention['weight_norm'] = weight_norm
        self.attn = AttentionLayer(**attention)
        num_pre_attention_layers = num_pre_attention_layers or num_layers
        if concat_attention and num_pre_attention_layers > 0:
            input_size = input_size + self.attn.output_size
            embedd_attn = self.attn
            del self.attn
        else:
            embedd_attn = None
        self.rnn_att = Recurrent(mode, input_size, hidden_size, num_layers=num_pre_attention_layers, bias=bias, dropout=dropout, forget_bias=forget_bias, residual=residual, weight_norm=weight_norm, attention_layer=embedd_attn)
        if num_layers > num_pre_attention_layers:
            self.rnn_no_att = Recurrent(mode, hidden_size, hidden_size, num_layers=num_layers - num_pre_attention_layers, bias=bias, batch_first=batch_first, residual=residual, weight_norm=weight_norm, dropout=dropout, forget_bias=forget_bias)

    def forward(self, inputs, context, hidden=None, mask_attention=None, get_attention=False):
        if isinstance(context, tuple):
            context_keys, context_values = context
        else:
            context_keys = context_values = context
        if hasattr(self, 'rnn_no_att'):
            if hidden is None:
                hidden = [None] * 2
            hidden, hidden_2 = hidden
        if not self.concat_attention:
            outputs, hidden = self.rnn_att(inputs, hidden)
            self.attn.set_mask(mask_attention)
            outputs, attentions = self.attn(outputs, context_keys, context_values)
        else:
            out = self.rnn_att(inputs, hidden, context, mask_attention=mask_attention, get_attention=get_attention)
            if get_attention:
                outputs, hidden, attentions = out
            else:
                outputs, hidden = out
        if hasattr(self, 'rnn_no_att'):
            outputs, hidden_2 = self.rnn_no_att(outputs, hidden_2)
            hidden = hidden, hidden_2
        if get_attention:
            return outputs, hidden, attentions
        else:
            return outputs, hidden


def positional_embedding(position_or_length, channels, min_timescale=1.0, max_timescale=10000.0, offset=0, device=None):
    assert channels % 2 == 0
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1.0)
    if torch.is_tensor(position_or_length):
        position = position_or_length + offset
        device = device or position_or_length.device
    else:
        position = torch.arange(offset, offset + position_or_length, device=device, dtype=torch.long)
    inv_timescales = torch.arange(0, num_timescales, device=device, dtype=torch.float)
    inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
    dims = position.dim() - 1
    inv_timescales = inv_timescales.view(*([1] * dims), -1)
    scaled_time = position.float().unsqueeze(-1) * inv_timescales
    signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)
    return signal


class PositionalEmbedding(nn.Module):

    def __init__(self, channels, min_timescale=1.0, max_timescale=10000.0):
        super(PositionalEmbedding, self).__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.channels = channels

    def forward(self, x):
        position = x if x.dim() == 1 else x.contiguous().view(-1)
        emb = positional_embedding(position, self.channels, device=x.device, min_timescale=self.min_timescale, max_timescale=self.max_timescale)
        return emb.view(*x.shape, -1)


class AverageNetwork(nn.Module):
    """ Average Attention Network Block from https://arxiv.org/abs/1805.00631 
    """

    def __init__(self, input_size, inner_linear, inner_groups=1, layer_norm=True, weight_norm=False, dropout=0, batch_first=True):
        super(AverageNetwork, self).__init__()
        wn_func = wn if weight_norm else lambda x: x
        self.input_size = input_size
        self.time_step = 0
        self.batch_dim, self.time_dim = (0, 1) if batch_first else (1, 0)
        self.gates = nn.Sequential(wn_func(nn.Linear(2 * input_size, 2 * input_size)), nn.Sigmoid())
        if layer_norm:
            self.lnorm = nn.LayerNorm(input_size)
        self.fc = nn.Sequential(wn_func(Linear(input_size, inner_linear, groups=inner_groups)), nn.ReLU(inplace=True), nn.Dropout(dropout), wn_func(Linear(inner_linear, input_size, groups=inner_groups)))

    def forward(self, x, state=None):
        if state is None:
            self.time_step = 0
        num_steps = torch.arange(self.time_step + 1, x.size(self.time_dim) + self.time_step + 1, device=x.device, dtype=x.dtype).view(-1, 1)
        if state is None:
            avg_attn = x.cumsum(self.time_dim) / num_steps
        else:
            past_num_steps = self.time_step
            avg_attn = (self.time_step * state.unsqueeze(self.time_dim) + x.cumsum(self.time_dim)) / num_steps
        state = avg_attn.select(self.time_dim, -1)
        g = self.fc(avg_attn)
        gate_values = self.gates(torch.cat((x, g), dim=-1))
        input_gate, forget_gate = gate_values.chunk(2, dim=-1)
        output = (input_gate + 1) * x + forget_gate * g
        if hasattr(self, 'lnorm'):
            output = self.lnorm(output)
        self.time_step += x.size(self.time_dim)
        return output, state


class EncoderBlock(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, inner_linear=2048, inner_groups=1, batch_first=True, layer_norm=True, weight_norm=False, dropout=0):
        super(EncoderBlock, self).__init__()
        wn_func = wn if weight_norm else lambda x: x
        if layer_norm:
            self.lnorm1 = nn.LayerNorm(hidden_size)
            self.lnorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.attention = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout=dropout, causal=False, batch_first=batch_first, groups=inner_groups, weight_norm=weight_norm)
        self.fc = nn.Sequential(wn_func(Linear(hidden_size, inner_linear, groups=inner_groups)), nn.ReLU(inplace=True), nn.Dropout(dropout), wn_func(Linear(inner_linear, hidden_size, groups=inner_groups)))

    def set_mask(self, mask):
        self.attention.set_mask_q(mask)
        self.attention.set_mask_k(mask)

    def forward(self, inputs):
        x = inputs
        res = x
        x, _ = self.attention(x, x, x)
        x = self.dropout(x).add_(res)
        x = self.lnorm1(x) if hasattr(self, 'lnorm1') else x
        res = x
        x = self.fc(x)
        x = self.dropout(x).add_(res)
        x = self.lnorm2(x) if hasattr(self, 'lnorm2') else x
        return x


class EncoderBlockPreNorm(EncoderBlock):

    def __init__(self, *kargs, **kwargs):
        super(EncoderBlockPreNorm, self).__init__(*kargs, **kwargs)

    def forward(self, inputs):
        x = inputs
        res = x
        x = self.lnorm1(x) if hasattr(self, 'lnorm1') else x
        x, _ = self.attention(x, x, x)
        x = self.dropout(x).add_(res)
        res = x
        x = self.lnorm2(x) if hasattr(self, 'lnorm2') else x
        x = self.fc(x)
        x = self.dropout(x).add_(res)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, inner_linear=2048, inner_groups=1, batch_first=True, layer_norm=True, weight_norm=False, dropout=0, stateful=None, state_dim=None, causal=True):
        super(DecoderBlock, self).__init__()
        wn_func = wn if weight_norm else lambda x: x
        if layer_norm:
            self.lnorm1 = nn.LayerNorm(hidden_size)
            self.lnorm2 = nn.LayerNorm(hidden_size)
            self.lnorm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.weight_norm = weight_norm
        self.stateful = stateful
        self.batch_first = batch_first
        self.attention = MultiHeadAttention(hidden_size, hidden_size, num_heads, batch_first=batch_first, dropout=dropout, causal=False, groups=inner_groups, weight_norm=weight_norm)
        if stateful is not None:
            residual = False
            stateful_hidden = hidden_size
            if state_dim is not None:
                stateful_hidden = state_dim
            if stateful_hidden != hidden_size:
                self.state_proj = nn.Linear(stateful_hidden, hidden_size)
            if stateful.endswith('_res'):
                stateful = stateful.replace('_res', '')
                residual = True
            if stateful in ['RNN', 'iRNN', 'LSTM', 'GRU']:
                self.state_block = Recurrent(stateful, hidden_size, stateful_hidden, dropout=dropout, residual=residual, batch_first=batch_first)
            else:
                self.state_block = AverageNetwork(hidden_size, hidden_size, layer_norm=layer_norm, weight_norm=weight_norm, batch_first=batch_first)
        else:
            self.masked_attention = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout=dropout, batch_first=batch_first, causal=causal, groups=inner_groups, weight_norm=weight_norm)
        self.fc = nn.Sequential(wn_func(Linear(hidden_size, inner_linear, groups=inner_groups)), nn.ReLU(inplace=True), nn.Dropout(dropout), wn_func(Linear(inner_linear, hidden_size, groups=inner_groups)))

    def set_mask(self, mask, context_mask):
        self.attention.set_mask_q(mask)
        self.attention.set_mask_k(context_mask)
        if hasattr(self, 'masked_attention'):
            self.masked_attention.set_mask_q(mask)
            self.masked_attention.set_mask_k(mask)

    def forward(self, inputs, context, state=None):
        x = inputs
        res = x
        if self.stateful:
            x, state = self.state_block(x, state)
        else:
            if state is None:
                x_past = x
                mask_past = self.masked_attention.mask_k
            else:
                time_dim = 1 if self.batch_first else 0
                x_past, mask_past = state
                x_past = torch.cat((x_past, x), time_dim)
                mask_past = torch.cat((mask_past, self.masked_attention.mask_k), time_dim)
                self.masked_attention.set_mask_k(mask_past)
            x, _ = self.masked_attention(x, x_past, x_past)
            state = x_past, mask_past
        if hasattr(self, 'state_proj'):
            x = self.state_proj(x)
        x = self.dropout(x).add(res)
        x = self.lnorm1(x) if hasattr(self, 'lnorm1') else x
        res = x
        x, attn_enc = self.attention(x, context, context)
        x = self.dropout(x).add_(res)
        x = self.lnorm2(x) if hasattr(self, 'lnorm2') else x
        res = x
        x = self.fc(x)
        x = self.dropout(x).add_(res)
        x = self.lnorm3(x) if hasattr(self, 'lnorm3') else x
        return x, attn_enc, state


class DecoderBlockPreNorm(DecoderBlock):

    def __init__(self, *kargs, **kwargs):
        super(DecoderBlockPreNorm, self).__init__(*kargs, **kwargs)

    def forward(self, inputs, context, state=None):
        x = inputs
        res = x
        x = self.lnorm1(x) if hasattr(self, 'lnorm1') else x
        if self.stateful:
            x, state = self.state_block(x, state)
        else:
            if state is None:
                x_past = x
            else:
                x_past = torch.cat((state, x), 1)
            x, _ = self.masked_attention(x, x_past, x_past)
            state = x_past
        if hasattr(self, 'state_proj'):
            x = self.state_proj(x)
        x = self.dropout(x).add(res)
        res = x
        x = self.lnorm2(x) if hasattr(self, 'lnorm2') else x
        x, attn_enc = self.attention(x, context, context)
        x = self.dropout(x).add_(res)
        res = x
        x = self.lnorm3(x) if hasattr(self, 'lnorm3') else x
        x = self.fc(x)
        x = self.dropout(x).add_(res)
        return x, attn_enc, state


class CharWordEmbedder(nn.Module):

    def __init__(self, num_chars, embedding_size, output_size, num_heads=8, padding_idx=0):
        super(CharWordEmbedder, self).__init__()
        self.num_chars = num_chars
        self.char_embedding = nn.Embedding(num_chars, embedding_size, padding_idx=padding_idx)
        self.attn = MultiHeadAttention(embedding_size, output_size, num_heads)
        self.padding_idx = padding_idx

    def forward(self, x):
        """ input is of size BxTwxTc --> BxTw
        """
        B, Tw, Tc = x.shape
        x = x.flatten(0, 1)
        x = self.char_embedding(x)
        mask = x.eq(self.padding_idx)
        self.attn.set_mask_k(mask)
        self.attn.set_mask_q(mask)
        x = self.attn(x)
        x = x.sum(1)
        x = x.view(B, Tw, x.size(-1))
        return x


class CNNEncoderBase(nn.Module):
    """docstring for CNNEncoder."""

    def __init__(self, model, context_size, context_transform=None, context_nonlinearity=None, spatial_context=True, finetune=True):
        super(CNNEncoderBase, self).__init__()
        self.model = model
        self.finetune = finetune
        self.batch_first = True
        self.toggle_grad()
        self.spatial_context = spatial_context
        if context_transform is None:
            self.context_size = context_size
        else:
            if self.spatial_context:
                self.context_transform = nn.Conv2d(context_size, context_transform, 1)
            else:
                self.context_transform = nn.Linear(context_size, context_transform)
            if context_nonlinearity is not None:
                self.context_nonlinearity = F.__dict__[context_nonlinearity]
            self.context_size = context_transform

    def toggle_grad(self):
        for p in self.model.parameters():
            p.requires_grad = self.finetune

    def named_parameters(self, *kargs, **kwargs):
        if self.finetune:
            return super(CNNEncoderBase, self).named_parameters(*kargs, **kwargs)
        elif hasattr(self, 'context_transform'):
            return self.context_transform.named_parameters(*kargs, **kwargs)
        else:
            return set()

    def state_dict(self, *kargs, **kwargs):
        if self.finetune:
            return super(CNNEncoderBase, self).state_dict(*kargs, **kwargs)
        elif hasattr(self, 'context_transform'):
            return self.context_transform.state_dict(*kargs, **kwargs)
        else:
            return {}

    def load_state_dict(self, state_dict, *kargs, **kwargs):
        if self.finetune:
            return super(CNNEncoderBase, self).load_state_dict(state_dict, *kargs, **kwargs)
        elif hasattr(self, 'context_transform'):
            return self.context_transform.load_state_dict(state_dict, *kargs, **kwargs)
        else:
            return


class ResNetEncoder(CNNEncoderBase):

    def __init__(self, model='resnet50', pretrained=True, **kwargs):
        model = resnet.__dict__[model](pretrained=pretrained)
        super(ResNetEncoder, self).__init__(model, context_size=model.fc.in_features, **kwargs)
        del self.model.fc

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
        if not self.spatial_context:
            x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), x.size(1))
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x


class DenseNetEncoder(CNNEncoderBase):

    def __init__(self, model='densenet121', pretrained=True, **kwargs):
        model = densenet.__dict__[model](pretrained=pretrained)
        super(DenseNetEncoder, self).__init__(model, context_size=model.classifier.in_features, **kwargs)
        del self.model.classifier

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            features = self.model.features(x)
            x = F.relu(features, inplace=True)
        if not self.spatial_context:
            x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), x.size(1))
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x


class VGGEncoder(CNNEncoderBase):

    def __init__(self, model='vgg16', pretrained=True, **kwargs):
        model = vgg.__dict__[model](pretrained=pretrained)
        super(VGGEncoder, self).__init__(model, context_size=model.classifier.in_features, **kwargs)
        del self.model.classifier

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            x = self.features(x)
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x


class AlexNetEncoder(CNNEncoderBase):

    def __init__(self, model='alexnet', pretrained=True, **kwargs):
        model = alexnet.__dict__[model](pretrained=pretrained)
        super(AlexNetEncoder, self).__init__(model, context_size=model.classifier.in_features, **kwargs)
        del self.model.classifier

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            x = self.features(x)
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x


class SqueezeNetEncoder(CNNEncoderBase):

    def __init__(self, model='squeezenet1_1', pretrained=True, **kwargs):
        model = squeezenet.__dict__[model](pretrained=pretrained)
        super(SqueezeNetEncoder, self).__init__(model, context_size=model.classifier.in_features, **kwargs)
        del self.model.classifier

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            x = self.features(x)
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x


class WeightDrop(torch.nn.Module):

    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def _dummy(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self._dummy
        for name_w in self.weights:
            None
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


def _dummy(*args, **kwargs):
    return


def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


class WeightNorm(torch.nn.Module):

    def __init__(self, weights, dim):
        super(WeightNorm, self).__init__()
        self.weights = weights
        self.dim = dim

    def compute_weight(self, module, name):
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')
        return v * (g / _norm(v, self.dim))

    @staticmethod
    def apply(module, weights, dim):
        if issubclass(type(module), torch.nn.RNNBase):
            module.flatten_parameters = _dummy
        if weights is None:
            weights = [w for w in module._parameters.keys() if 'weight' in w]
        fn = WeightNorm(weights, dim)
        for name in weights:
            if hasattr(module, name):
                logging.debug('Applying weight norm to {} - {}'.format(str(module), name))
                weight = getattr(module, name)
                del module._parameters[name]
                module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
                module.register_parameter(name + '_v', Parameter(weight.data))
                setattr(module, name, fn.compute_weight(module, name))
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        for name in self.weights:
            weight = self.compute_weight(module)
            delattr(module, name)
            del module._parameters[name + '_g']
            del module._parameters[name + '_v']
            module.register_parameter(name, Parameter(weight.data))

    def __call__(self, module, inputs):
        for name in self.weights:
            setattr(module, name, self.compute_weight(module, name))


PAD, UNK, BOS, EOS = [0, 1, 2, 3]


def is_empty(x):
    if x is None:
        return True
    if isinstance(x, tuple) or isinstance(x, list):
        return all([is_empty(x_i) for x_i in x])
    if isinstance(x, State):
        return all([is_empty(getattr(x, s)) for s in State.__slots__ if s is not 'batch_first'])
    return False


class State(object):
    __slots__ = ['batch_first', 'hidden', 'inputs', 'outputs', 'context', 'attention', 'attention_score', 'mask']

    def __init__(self, hidden=None, inputs=None, outputs=None, context=None, attention=None, attention_score=None, mask=None, batch_first=False):
        self.hidden = hidden
        self.outputs = outputs
        self.inputs = inputs
        self.context = context
        self.attention = attention
        self.mask = mask
        self.batch_first = batch_first
        self.attention_score = attention_score

    def __select_state(self, state, i, type_state='hidden'):
        if isinstance(state, tuple):
            return tuple(self.__select_state(s, i, type_state) for s in state)
        elif torch.is_tensor(state):
            if type_state == 'hidden':
                batch_dim = 0 if state.dim() < 3 else 1
            else:
                batch_dim = 0 if self.batch_first else 1
            if state.size(batch_dim) > i:
                return state.narrow(batch_dim, i, 1)
            else:
                return None
        else:
            return state

    def __merge_states(self, state_list, type_state='hidden'):
        if state_list is None:
            return None
        if isinstance(state_list[0], State):
            return State().from_list(state_list)
        if isinstance(state_list[0], tuple):
            return tuple([self.__merge_states(s, type_state) for s in zip(*state_list)])
        elif torch.is_tensor(state_list[0]):
            if type_state == 'hidden':
                batch_dim = 0 if state_list[0].dim() < 3 else 1
            else:
                batch_dim = 0 if self.batch_first else 1
            return torch.cat(state_list, batch_dim)
        else:
            assert state_list[1:] == state_list[:-1]
            return state_list[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            state_list = [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
            return State().from_list(state_list)
        else:
            item = State()
            for s in self.__slots__:
                value = getattr(self, s, None)
                if isinstance(value, State):
                    selected_value = value[index]
                else:
                    selected_value = self.__select_state(value, index, s)
                setattr(item, s, selected_value)
            return item

    def as_list(self):
        i = 0
        out_list = []
        item = self.__getitem__(i)
        while not is_empty(item):
            out_list.append(item)
            i += 1
            item = self.__getitem__(i)
        return out_list

    def from_list(self, state_list):
        for s in self.__slots__:
            values = [getattr(item, s, None) for item in state_list]
            setattr(self, s, self.__merge_states(values, s))
        return self


class PrevasiveEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, encoding='embedding', mask_symbol=PAD, dropout=0):
        super(PrevasiveEncoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.hidden_size = hidden_size
        self.batch_first = True
        self.mask_symbol = mask_symbol
        self.embedder = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.hidden_size = embedding_size
        self.encoding = encoding

    def forward(self, inputs, hidden=None):
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        x.add_(positional_embedding(x))
        x = self.dropout(x)
        return State(outputs=x, mask=padding_mask, batch_first=True)


def merge_time(x1, x2, x3=None, mode='cat'):
    B1, C1, T1 = x1.shape
    B2, C2, T2 = x2.shape
    assert B1 == B2
    if mode == 'cat':
        shape1 = list(x1.shape)
        shape1.insert(3, T2)
        shape2 = list(x2.shape)
        shape2.insert(2, T1)
        return torch.cat((x1.unsqueeze(3).expand(shape1), x2.unsqueeze(2).expand(shape2)), dim=1)
    elif mode == 'sum':
        assert C1 == C2
        return x1.unsqueeze(3) + x2.unsqueeze(2)
    elif mode == 'film':
        assert x3 is not None
        assert C1 == C2
        assert x3.shape == x2.shape
        return x1.unsqueeze(3) * x2.unsqueeze(2) + x3.unsqueeze(2)


class PrevasiveDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, context_size=None, dropout=0, convnet='resnet', merge_time='cat', stateful=False, mask_symbol=PAD, tie_embedding=True):
        super(PrevasiveDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        context_size = context_size or hidden_size
        self.batch_first = True
        self.mask_symbol = mask_symbol
        self.embedder = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.merge_time = merge_time
        if self.merge_time == 'cat':
            embed_input = embedding_size + context_size
        else:
            embed_input = embedding_size
        if convnet == 'resnet':
            self.main_block = ResNet(embed_input, output_size=embedding_size)
        elif convnet == 'densenet':
            self.main_block = DenseNet(embed_input, output_size=embedding_size, block_config=(6, 6, 6, 8), growth_rate=40)
        self.pool = nn.AdaptiveMaxPool2d((1, None))
        self.classifier = nn.Linear(embedding_size, vocab_size)
        if tie_embedding:
            self.embedder.weight = self.classifier.weight

    def forward(self, inputs, state, get_attention=False):
        context = state.context
        time_step = 0
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        x.add_(positional_embedding(x, offset=time_step))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        y = context.outputs.transpose(1, 2)
        x = merge_time(y, x, mode=self.merge_time)
        x = self.main_block(x)
        x = self.pool(x)
        x = x.squeeze(2).transpose(1, 2)
        x = self.classifier(x)
        return x, state


class HiddenTransform(nn.Module):
    """docstring for [object Object]."""

    def __init__(self, input_shape, output_shape, activation='tanh', bias=True, batch_first=False):
        super(HiddenTransform, self).__init__()
        self.batch_first = batch_first
        self.activation = nn.Tanh() if activation == 'tanh' else None
        if not isinstance(input_shape, tuple):
            input_shape = input_shape,
        if not isinstance(output_shape, tuple):
            output_shape = output_shape,
        assert len(input_shape) == len(output_shape)
        self.module_list = nn.ModuleList()
        for i in range(len(input_shape)):
            self.module_list.append(nn.Linear(input_shape[i], output_shape[i], bias=bias))

    def forward(self, hidden):
        hidden_in = hidden if isinstance(hidden, tuple) else (hidden,)
        hidden_out = []
        for i, h_in in enumerate(hidden_in):
            if not self.batch_first:
                h_in = h_in.transpose(0, 1)
            h_in = h_in.contiguous().view(h_in.size(0), -1)
            h_out = self.module_list[i](h_in)
            if self.activation is not None:
                h_out = self.activation(h_out)
            h_out = h_out.unsqueeze(1 if self.batch_first else 0)
            hidden_out.append(h_out)
        if isinstance(hidden, tuple):
            return tuple(hidden_out)
        else:
            return hidden_out[0]


class RecurrentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, embedding_size=None, num_layers=1, bias=True, batch_first=False, dropout=0, embedding_dropout=0, forget_bias=None, context_transform=None, context_transform_bias=True, hidden_transform=None, hidden_transform_bias=True, bidirectional=True, adapt_bidirectional_size=False, num_bidirectional=None, mode='LSTM', pack_inputs=True, residual=False, weight_norm=False):
        super(RecurrentEncoder, self).__init__()
        self.pack_inputs = pack_inputs
        self.layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        embedding_size = embedding_size or hidden_size
        num_bidirectional = num_bidirectional or num_layers
        self.embedder = nn.Embedding(vocab_size, embedding_size, sparse=False, padding_idx=PAD)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)
        if adapt_bidirectional_size and bidirectional and num_bidirectional > 0:
            assert hidden_size % 2 == 0
            hidden_size = hidden_size // 2
        self.hidden_size = hidden_size
        self.context_size = 2 * hidden_size if bidirectional else hidden_size
        if context_transform is not None:
            self.context_transform = nn.Linear(self.context_size, context_transform, bias=context_transform_bias)
            self.context_size = context_transform, self.context_size
            if weight_norm:
                self.context_transform = wn(self.context_transform)
        if num_bidirectional is not None and num_bidirectional < num_layers:
            assert hidden_transform is None, 'hidden transform can be used only for single bidi encoder for now'
            self.rnn = StackedRecurrent(dropout=dropout, residual=residual)
            self.rnn.add_module('bidirectional', Recurrent(mode, embedding_size, hidden_size, num_layers=num_bidirectional, bias=bias, batch_first=batch_first, residual=residual, weight_norm=weight_norm, forget_bias=forget_bias, dropout=dropout, bidirectional=True))
            self.rnn.add_module('unidirectional', Recurrent(mode, hidden_size * 2, hidden_size * 2, num_layers=num_layers - num_bidirectional, batch_first=batch_first, residual=residual, weight_norm=weight_norm, forget_bias=forget_bias, bias=bias, dropout=dropout, bidirectional=False))
        else:
            self.rnn = Recurrent(mode, embedding_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, residual=residual, weight_norm=weight_norm, dropout=dropout, bidirectional=bidirectional)
            if hidden_transform is not None:
                hidden_size = hidden_size * num_layers
                if bidirectional:
                    hidden_size += hidden_size
                if mode == 'LSTM':
                    hidden_size = hidden_size, hidden_size
                self.hidden_transform = HiddenTransform(hidden_size, hidden_transform, bias=hidden_transform_bias, batch_first=batch_first)

    def forward(self, inputs, hidden=None):
        if isinstance(inputs, PackedSequence):
            emb = PackedSequence(self.embedding_dropout(self.embedder(inputs.data)), inputs.batch_sizes)
            bsizes = inputs.batch_sizes
            max_batch = int(bsizes[0])
            time_dim = 1 if self.batch_first else 0
            range_batch = torch.arange(0, max_batch, dtype=bsizes.dtype, device=bsizes.device)
            range_batch = range_batch.unsqueeze(time_dim)
            bsizes = bsizes.unsqueeze(1 - time_dim)
            padding_mask = (bsizes - range_batch).le(0)
        else:
            padding_mask = inputs.eq(PAD)
            emb = self.embedding_dropout(self.embedder(inputs))
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(inputs, PackedSequence):
            outputs = unpack(outputs)[0]
        outputs = self.dropout(outputs)
        if hasattr(self, 'context_transform'):
            context = self.context_transform(outputs)
        else:
            context = None
        if hasattr(self, 'hidden_transform'):
            hidden_t = self.hidden_transform(hidden_t)
        state = State(outputs=outputs, hidden=hidden_t, context=context, mask=padding_mask, batch_first=self.batch_first)
        return state


class RecurrentDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, hidden_size=128, embedding_size=None, num_layers=1, bias=True, forget_bias=None, batch_first=False, dropout=0, embedding_dropout=0, mode='LSTM', residual=False, weight_norm=False, tie_embedding=True):
        super(RecurrentDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.embedder = nn.Embedding(vocab_size, embedding_size, sparse=False, padding_idx=PAD)
        self.rnn = Recurrent(mode, embedding_size, self.hidden_size, num_layers=num_layers, bias=bias, forget_bias=forget_bias, batch_first=batch_first, residual=residual, weight_norm=weight_norm, dropout=dropout, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight

    def forward(self, inputs, state):
        context, hidden = state.context, state.hidden
        if isinstance(inputs, PackedSequence):
            emb = PackedSequence(self.embedding_dropout(self.embedder(inputs.data)), inputs.batch_size)
        else:
            emb = self.embedding_dropout(self.embedder(inputs))
        x, hidden_t = self.rnn(emb, hidden)
        if isinstance(inputs, PackedSequence):
            x = unpack(x)[0]
        x = self.dropout(x)
        x = self.classifier(x)
        return x, State(hidden=hidden_t, context=context, batch_first=self.batch_first)


class RecurrentAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, hidden_size=128, embedding_size=None, num_layers=1, bias=True, forget_bias=None, batch_first=False, bias_classifier=True, dropout=0, embedding_dropout=0, tie_embedding=False, residual=False, mode='LSTM', weight_norm=False, attention=None, concat_attention=True, num_pre_attention_layers=None):
        super(RecurrentAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        attention = attention or {}
        self.layers = num_layers
        self.batch_first = batch_first
        self.embedder = nn.Embedding(vocab_size, embedding_size, sparse=False, padding_idx=PAD)
        self.rnn = RecurrentAttention(embedding_size, context_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, mode=mode, forget_bias=forget_bias, residual=residual, weight_norm=weight_norm, attention=attention, concat_attention=concat_attention, num_pre_attention_layers=num_pre_attention_layers)
        self.classifier = nn.Linear(hidden_size, vocab_size, bias=bias_classifier)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight
        self.hidden_size = hidden_size

    def forward(self, inputs, state, get_attention=False):
        context, hidden = state.context, state.hidden
        if context.context is not None:
            attn_input = context.context, context.outputs
        else:
            attn_input = context.outputs
        emb = self.embedding_dropout(self.embedder(inputs))
        if get_attention:
            x, hidden, attentions = self.rnn(emb, attn_input, state.hidden, mask_attention=context.mask, get_attention=get_attention)
        else:
            x, hidden = self.rnn(emb, attn_input, state.hidden, mask_attention=context.mask)
        x = self.dropout(x)
        x = self.classifier(x)
        new_state = State(hidden=hidden, context=context, batch_first=self.batch_first)
        if get_attention:
            new_state.attention_score = attentions
        return x, new_state


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, output, state, logprob, score, attention=None):
        """Initializes the Sequence.

        Args:
          output: List of word ids in the sequence.
          state: Model state after generating the previous word.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
        """
        self.output = output
        self.state = state
        self.logprob = logprob
        self.score = score
        self.attention = attention

    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self, decode_step, eos_id=EOS, beam_size=3, max_sequence_length=50, get_attention=False, length_normalization_factor=0.0, length_normalization_const=5.0, device_ids=None):
        """Initializes the generator.

        Args:
          deocde_step: function, with inputs: (input, state) and outputs len(vocab) values
          eos_id: the token number symobling the end of sequence
          beam_size: Beam size to use when generating sequences.
          max_sequence_length: The maximum sequence length before stopping the search.
          length_normalization_factor: If != 0, a number x such that sequences are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of sequences depending on their lengths. For example, if
            x > 0 then longer sequences will be favored.
            alpha in: https://arxiv.org/abs/1609.08144
          length_normalization_const: 5 in https://arxiv.org/abs/1609.08144
        """
        self.decode_step = decode_step
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.length_normalization_factor = length_normalization_factor
        self.length_normalization_const = length_normalization_const
        self.get_attention = get_attention
        self.device_ids = device_ids

    def beam_search(self, initial_input, initial_state=None):
        """Runs beam search sequence generation on a single image.

        Args:
          initial_input: An initial input for the model -
                         list of batch size holding the first input for every entry.
          initial_state (optional): An initial state for the model -
                         list of batch size holding the current state for every entry.

        Returns:
          A list of batch size, each the most likely sequence from the possible beam_size candidates.
        """
        batch_size = len(initial_input)
        partial_sequences = [TopN(self.beam_size) for _ in range(batch_size)]
        complete_sequences = [TopN(self.beam_size) for _ in range(batch_size)]
        words, logprobs, new_state = self.decode_step(initial_input, initial_state, k=self.beam_size, feed_all_timesteps=True, get_attention=self.get_attention)
        for b in range(batch_size):
            for k in range(self.beam_size):
                seq = Sequence(output=initial_input[b] + [words[b][k]], state=new_state[b], logprob=logprobs[b][k], score=logprobs[b][k], attention=None if not self.get_attention else [new_state[b].attention_score])
                partial_sequences[b].push(seq)
        for _ in range(self.max_sequence_length - 1):
            partial_sequences_list = [p.extract() for p in partial_sequences]
            for p in partial_sequences:
                p.reset()
            flattened_partial = [s for sub_partial in partial_sequences_list for s in sub_partial]
            input_feed = [c.output for c in flattened_partial]
            state_feed = [c.state for c in flattened_partial]
            if len(input_feed) == 0:
                break
            words, logprobs, new_states = self.decode_step(input_feed, state_feed, k=self.beam_size + 1, get_attention=self.get_attention, device_ids=self.device_ids)
            idx = 0
            for b in range(batch_size):
                for partial in partial_sequences_list[b]:
                    state = new_states[idx]
                    if self.get_attention:
                        attention = partial.attention + [new_states[idx].attention_score]
                    else:
                        attention = None
                    k = 0
                    num_hyp = 0
                    while num_hyp < self.beam_size:
                        w = words[idx][k]
                        output = partial.output + [w]
                        logprob = partial.logprob + logprobs[idx][k]
                        score = logprob
                        k += 1
                        num_hyp += 1
                        if w.item() == self.eos_id:
                            if self.length_normalization_factor > 0:
                                L = self.length_normalization_const
                                length_penalty = (L + len(output)) / (L + 1)
                                score /= length_penalty ** self.length_normalization_factor
                            beam = Sequence(output, state, logprob, score, attention)
                            complete_sequences[b].push(beam)
                            num_hyp -= 1
                        else:
                            beam = Sequence(output, state, logprob, score, attention)
                            partial_sequences[b].push(beam)
                    idx += 1
        for b in range(batch_size):
            if not complete_sequences[b].size():
                complete_sequences[b] = partial_sequences[b]
        seqs = [complete.extract(sort=True)[0] for complete in complete_sequences]
        return seqs


def _limit_lengths(seqs, max_length=None, max_tokens=None):
    max_length = max_length or float('inf')
    lengths = [min(s.nelement(), max_length) for s in seqs]
    if max_tokens is not None:
        num_tokens = sum(lengths)
        if num_tokens > max_tokens:
            max_length = int(floor(num_tokens / len(seqs)))
            lengths = [min(length, max_length) for length in lengths]
    return lengths


def batch_sequences(seqs, max_length=None, max_tokens=None, fixed_length=None, batch_first=False, pad_value=PAD, sort=False, pack=False, augment=False, device=None, dtype=torch.long):
    """
    seqs: a list of Tensors to be batched together
    max_length: maximum sequence length permitted
    max_tokens: maximum number of tokens in batch permitted

    """
    batch_dim, time_dim = (0, 1) if batch_first else (1, 0)
    if fixed_length is not None:
        fixed_length = max_length = min(max_length, fixed_length)
    if len(seqs) == 1 and not fixed_length:
        lengths = _limit_lengths(seqs, max_length, max_tokens)
        seq_tensor = seqs[0].view(-1)[:lengths[0]]
        seq_tensor = seq_tensor.unsqueeze(batch_dim)
    else:
        if sort:
            seqs.sort(key=len, reverse=True)
        lengths = _limit_lengths(seqs, max_length, max_tokens)
        batch_length = max(lengths) if fixed_length is None else fixed_length
        tensor_size = (len(seqs), batch_length) if batch_first else (batch_length, len(seqs))
        seq_tensor = torch.full(tensor_size, pad_value, dtype=dtype, device=device)
        for i, seq in enumerate(seqs):
            start_seq = 0
            end_seq = lengths[i]
            if augment and end_seq < seq.nelement():
                delta = randrange(seq.nelement() - end_seq + 1)
                start_seq += delta
                end_seq += delta
            seq_tensor.narrow(time_dim, 0, lengths[i]).select(batch_dim, i).copy_(seq[start_seq:end_seq])
    if pack:
        seq_tensor = pack_padded_sequence(seq_tensor, lengths, batch_first=batch_first)
        if device is not None:
            seq_tensor = PackedSequence(seq_tensor.data, seq_tensor.batch_sizes)
    return seq_tensor, lengths


class Seq2Seq(nn.Module):

    def __init__(self, encoder=None, decoder=None, bridge=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if bridge is not None:
            self.bridge = bridge

    def bridge(self, context):
        return State(context=context, batch_first=getattr(self.decoder, 'batch_first', context.batch_first))

    def encode(self, inputs, hidden=None, device_ids=None):
        if isinstance(device_ids, tuple):
            return data_parallel(self.encoder, (inputs, hidden), device_ids=device_ids, dim=0 if self.encoder.batch_first else 1)
        else:
            return self.encoder(inputs, hidden)

    def decode(self, *kargs, **kwargs):
        device_ids = kwargs.pop('device_ids', None)
        if isinstance(device_ids, tuple):
            return data_parallel(self.decoder, *kargs, **kwargs, device_ids=device_ids, dim=0 if self.decoder.batch_first else 1)
        else:
            return self.decoder(*kargs, **kwargs)

    def forward(self, input_encoder, input_decoder, encoder_hidden=None, device_ids=None):
        if not isinstance(device_ids, dict):
            device_ids = {'encoder': device_ids, 'decoder': device_ids}
        context = self.encode(input_encoder, encoder_hidden, device_ids=device_ids.get('encoder', None))
        if hasattr(self, 'bridge'):
            state = self.bridge(context)
        output, state = self.decode(input_decoder, state, device_ids=device_ids.get('decoder', None))
        return output

    def _decode_step(self, input_list, state_list, args_dict={}, k=1, feed_all_timesteps=False, keep_all_timesteps=False, time_offset=0, time_multiply=1, apply_lsm=True, remove_unknown=False, get_attention=False, device_ids=None):
        view_shape = (-1, 1) if self.decoder.batch_first else (1, -1)
        time_dim = 1 if self.decoder.batch_first else 0
        device = next(self.decoder.parameters()).device
        if feed_all_timesteps:
            inputs = [torch.tensor(inp, device=device, dtype=torch.long) for inp in input_list]
            inputs = batch_sequences(inputs, device=device, batch_first=self.decoder.batch_first)[0]
        else:
            last_tokens = [inputs[-1] for inputs in input_list]
            inputs = torch.stack(last_tokens).view(*view_shape)
        states = State().from_list(state_list)
        decode_inputs = dict(get_attention=get_attention, device_ids=device_ids, **args_dict)
        if time_multiply > 1:
            decode_inputs['time_multiply'] = time_multiply
        logits, new_states = self.decode(inputs, states, **decode_inputs)
        if not keep_all_timesteps:
            logits = logits.select(time_dim, -1).contiguous()
        if remove_unknown:
            logits[:, UNK].fill_(-float('inf'))
        if apply_lsm:
            logprobs = log_softmax(logits, dim=-1)
        else:
            logprobs = logits
        logprobs, words = logprobs.topk(k, dim=-1)
        new_states_list = [new_states[i] for i in range(len(input_list))]
        return words, logprobs, new_states_list

    def generate(self, input_encoder, input_decoder, beam_size=None, max_sequence_length=None, length_normalization_factor=0, get_attention=False, device_ids=None, autoregressive=True):
        if not isinstance(device_ids, dict):
            device_ids = {'encoder': device_ids, 'decoder': device_ids}
        context = self.encode(input_encoder, device_ids=device_ids.get('encoder', None))
        if hasattr(self, 'bridge'):
            state = self.bridge(context)
        state_list = state.as_list()
        params = dict(decode_step=self._decode_step, beam_size=beam_size, max_sequence_length=max_sequence_length, get_attention=get_attention, length_normalization_factor=length_normalization_factor, device_ids=device_ids.get('encoder', None))
        if autoregressive:
            generator = SequenceGenerator(**params)
        else:
            generator = PermutedSequenceGenerator(**params)
        return generator.beam_search(input_decoder, state_list)


def index_select_2d(x, order):
    idxs = order.add(torch.arange(order.size(0), dtype=torch.long, device=order.device).view(-1, 1) * order.size(1))
    out_sz = x.shape
    out_sz = order.size(0), order.size(1), *out_sz[2:]
    return x.flatten(0, 1).index_select(0, idxs.contiguous().view(-1)).view(out_sz)


@torch.jit.script
def _reorder(order):
    B, T = order.shape
    reorder_list = []
    for j in range(T):
        reorder_list.append(order.eq(j).nonzero()[:, -1])
    return torch.stack(reorder_list, dim=-1)


def rand_order(T, block_size=None, block_ratio=0.25, out=None):
    if block_size is None:
        block_size = max(int(round(T * block_ratio)), 1)
    if block_size == 1:
        return torch.randperm(T, out=out)
    else:
        if out is None:
            out = torch.empty((T,), dtype=torch.long)
        order = list(range(T))
        offset = randrange(T)
        order = torch.tensor(order[offset:] + order[:offset])
        order = list(order.split(block_size))
        shuffle(order)
        order = torch.cat(order)
        out.copy_(order)
    return out


def permuted_order(inputs, padding_idx=PAD, eos_idx=EOS, batch_first=True):
    time_dim, batch_dim = (1, 0) if batch_first else (0, 1)
    B, T = inputs.size(batch_dim), inputs.size(time_dim)
    order = torch.arange(-1, T, dtype=torch.long, device=inputs.device)
    order = order.view(1, -1).expand(B, T + 1).contiguous()
    max_time = inputs.ne(padding_idx).sum(time_dim) - 1
    for i in range(B):
        t = int(max_time[i])
        scope = order[i].narrow(0, 1, t)
        rand_order(t, out=scope)
    order.add_(1)
    reorder = _reorder(order.narrow(time_dim, 1, T) - 1)
    if not batch_first:
        order = order.t()
        reorder = reorder.t()
    return order, reorder


def repeat(x, N, dim=0):
    if x is None:
        return None
    sz = list(x.shape)
    expand_sz = list(x.shape)
    sz.insert(dim, 1)
    expand_sz.insert(dim, N)
    x = x.view(*sz)
    x = x.expand(*expand_sz)
    x = x.contiguous()
    return x


class TransformerAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6, num_heads=8, batch_first=True, dropout=0, inner_linear=2048, inner_groups=1, prenormalized=False, stateful=None, state_dim=None, mask_symbol=PAD, tie_embedding=True, layer_norm=True, weight_norm=False, embedder=None, classifier=True, permuted=False, learned_condition=False, max_length=512, **kwargs):
        super(TransformerAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        if embedding_size != hidden_size:
            self.input_projection = nn.Parameter(torch.empty(embedding_size, hidden_size))
            nn.init.kaiming_uniform_(self.input_projection, a=math.sqrt(5))
        self.batch_first = batch_first
        self.mask_symbol = mask_symbol
        self.embedder = embedder or nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.stateful = stateful
        self.permuted = permuted
        block_args = dict(hidden_size=hidden_size, num_heads=num_heads, inner_linear=inner_linear, inner_groups=inner_groups, layer_norm=layer_norm, weight_norm=weight_norm, dropout=dropout, batch_first=batch_first, stateful=stateful, state_dim=state_dim)
        if permuted:
            if learned_condition:
                self.conditioned_pos = nn.Embedding(max_length, embedding_size)
            else:
                self.conditioned_pos = PositionalEmbedding(embedding_size, min_timescale=10000.0, max_timescale=100000000.0)
        if prenormalized:
            block = DecoderBlockPreNorm
        else:
            block = DecoderBlock
        self.blocks = nn.ModuleList([block(**block_args) for _ in range(num_layers)])
        if layer_norm and prenormalized:
            self.lnorm = nn.LayerNorm(hidden_size)
        if classifier:
            self.classifier = nn.Linear(embedding_size, vocab_size)
            if tie_embedding:
                self.embedder.weight = self.classifier.weight
            if embedding_size != hidden_size:
                if tie_embedding:
                    self.output_projection = self.input_projection
                else:
                    self.output_projection = nn.Parameter(torch.empty(embedding_size, hidden_size))
                    nn.init.kaiming_uniform_(self.output_projection, a=math.sqrt(5))

    def forward(self, inputs, state, time_multiply=1, get_attention=False, causal=None, input_order=None, output_order=None, output_reorder=None):
        context = state.context
        time_step = 0
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        if self.stateful:
            block_state = state.hidden
            if block_state is None:
                self.time_step = 0
            time_step = self.time_step
        else:
            block_state = state.inputs
            time_step = 0 if block_state is None else block_state[0][0].size(time_dim)
        if block_state is None:
            block_state = [None] * len(self.blocks)
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        if hasattr(self, 'input_projection'):
            x = x @ self.input_projection
        pos_embedding = positional_embedding(x.size(time_dim), x.size(-1), offset=time_step, device=x.device).unsqueeze(batch_dim)
        x.add_(pos_embedding)
        if self.permuted:
            if self.training:
                output_order, output_reorder = permuted_order(inputs, batch_first=self.batch_first)
                pos_target = output_order.narrow(time_dim, 1, x.size(time_dim))
                pos_input = output_order.narrow(time_dim, 0, x.size(time_dim))
                x = index_select_2d(x, pos_input)
                cond_embedding = self.conditioned_pos(pos_target)
            else:
                pos_target = torch.arange(x.size(time_dim) * time_multiply, device=x.device) + time_step + 1
                cond_embedding = self.conditioned_pos(pos_target).unsqueeze(batch_dim)
                output_reorder = None
            if time_multiply > 1:
                padding_mask = repeat(padding_mask, time_multiply).flatten(0, 1)
                x = repeat(x, time_multiply).flatten(0, 1)
                cond_embedding = repeat(cond_embedding.squeeze(batch_dim), inputs.size(batch_dim), dim=1).transpose(0, 1).contiguous().view_as(x)
                context.mask = repeat(context.mask, time_multiply).flatten(0, 1)
                context.outputs = repeat(context.outputs, time_multiply).flatten(0, 1)
            x = x.add(cond_embedding)
        x = self.dropout(x)
        attention_scores = []
        updated_state = []
        for i, block in enumerate(self.blocks):
            if causal is not None:
                block.masked_attention.causal = causal
            block.set_mask(padding_mask, context.mask)
            x, attn_enc, block_s = block(x, context.outputs, block_state[i])
            updated_state.append(block_s)
            if get_attention:
                attention_scores.append(attn_enc)
            else:
                del attn_enc
        if hasattr(self, 'lnorm'):
            x = self.lnorm(x)
        if output_reorder is not None:
            x = index_select_2d(x, output_reorder)
        if hasattr(self, 'output_projection'):
            x = x @ self.output_projection.t()
        if self.classifier is not None:
            x = self.classifier(x)
        if self.stateful:
            state.hidden = tuple(updated_state)
            self.time_step += 1
        else:
            state.inputs = tuple(updated_state)
        if get_attention:
            state.attention_score = attention_scores
        if time_multiply > 1:
            x = x.view(time_multiply, inputs.size(batch_dim), -1)
            x = x.transpose(0, 1)
        return x, state


class TransformerAttentionEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6, num_heads=8, inner_linear=2048, inner_groups=1, prenormalized=False, mask_symbol=PAD, batch_first=True, layer_norm=True, weight_norm=False, dropout=0, embedder=None):
        super(TransformerAttentionEncoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        if embedding_size != hidden_size:
            self.input_projection = nn.Parameter(torch.empty(embedding_size, hidden_size))
            nn.init.kaiming_uniform_(self.input_projection, a=math.sqrt(5))
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.mask_symbol = mask_symbol
        self.embedder = embedder or nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        if prenormalized:
            block = EncoderBlockPreNorm
        else:
            block = EncoderBlock
        self.blocks = nn.ModuleList([block(hidden_size, num_heads=num_heads, inner_linear=inner_linear, inner_groups=inner_groups, layer_norm=layer_norm, weight_norm=weight_norm, batch_first=batch_first, dropout=dropout) for _ in range(num_layers)])
        if layer_norm and prenormalized:
            self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, hidden=None):
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        if hasattr(self, 'input_projection'):
            x = x @ self.input_projection
        pos_embedding = positional_embedding(x.size(time_dim), x.size(-1), device=x.device)
        x.add_(pos_embedding.unsqueeze(batch_dim))
        x = self.dropout(x)
        for block in self.blocks:
            block.set_mask(padding_mask)
            x = block(x)
        if hasattr(self, 'lnorm'):
            x = self.lnorm(x)
        return State(outputs=x, mask=padding_mask, batch_first=self.batch_first)


class HybridSeq2Seq(Seq2Seq):

    def __init__(self, vocab_size, tie_embedding=False, transfer_hidden=False, encoder=None, decoder=None):
        super(HybridSeq2Seq, self).__init__()
        encoder = encoder or {}
        decoder = decoder or {}
        encoder.setdefault('vocab_size', vocab_size)
        encoder_type = encoder.pop('type', 'recurrent')
        decoder_type = decoder.pop('type', 'recurrent')
        if encoder_type == 'recurrent':
            self.encoder = RecurrentEncoder(**encoder)
        elif encoder_type == 'transformer':
            self.encoder = TransformerAttentionEncoder(**encoder)
        decoder['context_size'] = self.encoder.hidden_size
        if decoder_type == 'recurrent':
            self.decoder = RecurrentAttentionDecoder(**decoder)
        elif decoder_type == 'transformer':
            self.decoder = TransformerAttentionDecoder(**encoder)
        self.transfer_hidden = transfer_hidden
        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.embedder.weight

    def generate(self, input_list, state_list, k=1, feed_all_timesteps=True, get_attention=False):
        if isinstance(self.decoder, TransformerAttentionDecoder):
            feed_all_timesteps = True
        else:
            feed_all_timesteps = False
        return super(HybridSeq2Seq, self).generate(input_list, state_list, k=k, feed_all_timesteps=feed_all_timesteps, get_attention=get_attention)


class Transformer(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6, num_heads=8, inner_linear=2048, inner_groups=1, dropout=0.1, prenormalized=False, tie_embedding=True, encoder=None, decoder=None, layer_norm=True, weight_norm=False, batch_first=True, stateful=None):
        super(Transformer, self).__init__()
        embedding_size = embedding_size or hidden_size
        encoder = encoder or {}
        decoder = decoder or {}
        encoder = deepcopy(encoder)
        decoder = deepcopy(decoder)
        encoder.setdefault('embedding_size', embedding_size)
        encoder.setdefault('hidden_size', hidden_size)
        encoder.setdefault('num_layers', num_layers)
        encoder.setdefault('num_heads', num_heads)
        encoder.setdefault('vocab_size', vocab_size)
        encoder.setdefault('layer_norm', layer_norm)
        encoder.setdefault('weight_norm', weight_norm)
        encoder.setdefault('dropout', dropout)
        encoder.setdefault('inner_linear', inner_linear)
        encoder.setdefault('inner_groups', inner_groups)
        encoder.setdefault('prenormalized', prenormalized)
        encoder.setdefault('batch_first', batch_first)
        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('num_layers', num_layers)
        decoder.setdefault('num_heads', num_heads)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('layer_norm', layer_norm)
        decoder.setdefault('weight_norm', weight_norm)
        decoder.setdefault('dropout', dropout)
        decoder.setdefault('inner_linear', inner_linear)
        decoder.setdefault('inner_groups', inner_groups)
        decoder.setdefault('batch_first', batch_first)
        decoder.setdefault('prenormalized', prenormalized)
        decoder.setdefault('stateful', stateful)
        if isinstance(vocab_size, tuple):
            embedder = CharWordEmbedder(vocab_size[1], embedding_size, hidden_size)
            encoder.setdefault('embedder', embedder)
            decoder.setdefault('embedder', embedder)
            decoder['classifier'] = False
        self.batch_first = batch_first
        self.encoder = TransformerAttentionEncoder(**encoder)
        self.decoder = TransformerAttentionDecoder(**decoder)
        if tie_embedding and not isinstance(vocab_size, tuple):
            assert self.encoder.embedder.weight.shape == self.decoder.classifier.weight.shape
            self.encoder.embedder.weight = self.decoder.classifier.weight
            if embedding_size != hidden_size:
                self.encoder.input_projection = self.decoder.input_projection


class AddLossModule(nn.Module):
    """adds a loss to module for easy parallelization"""

    def __init__(self, module, criterion, ignore_index=PAD):
        super(AddLossModule, self).__init__()
        self.module = module
        self.criterion = criterion
        self.ignore_index = ignore_index

    def forward(self, module_inputs, target):
        output = self.module(*module_inputs)
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        output = nn.functional.log_softmax(output, -1)
        loss = self.criterion(output, target).view(1, 1)
        nll = nn.functional.nll_loss(output, target, ignore_index=self.ignore_index, reduction='sum')
        _, argmax = output.max(-1)
        invalid_targets = target.eq(self.ignore_index)
        accuracy = argmax.eq(target).masked_fill_(invalid_targets, 0).long().sum()
        return loss, nll, accuracy.view(1, 1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionLayer,
     lambda: ([], {'query_size': 4, 'key_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (DenseNetEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (GatedConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (HiddenTransform,
     lambda: ([], {'input_shape': 4, 'output_shape': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (MaskedConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MaskedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (OrderAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PositionalEmbedding,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNetEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SDPAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (StackedConv,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TimeNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_eladhoffer_seq2seq_pytorch(_paritybench_base):
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

