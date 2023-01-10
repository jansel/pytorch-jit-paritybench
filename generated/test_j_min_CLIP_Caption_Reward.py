import sys
_module = sys.modules[__name__]
del sys
captioning = _module
data = _module
dataloader = _module
pth_loader = _module
pth_loader_FineCapEval = _module
AoAModel = _module
AttEnsemble = _module
AttModel = _module
BertCapModel = _module
CaptionModel = _module
FCModel = _module
M2Transformer = _module
ShowTellModel = _module
TransformerModel = _module
models = _module
cachedTransformer = _module
utils = _module
loss_wrapper = _module
losses = _module
clipscore = _module
config = _module
dist_utils = _module
div_utils = _module
eval_multi = _module
eval_utils = _module
misc = _module
opts = _module
resnet = _module
resnet_utils = _module
rewards = _module
utils = _module
clip = _module
clip = _module
model = _module
simple_tokenizer = _module
predict = _module
caption_data = _module
clip_model = _module
param = _module
pth_loader = _module
text_utils = _module
train_pl = _module
build_bpe_subword_nmt = _module
clip_prepro_feats = _module
clipscore_prepro_feats = _module
dump_to_h5df = _module
dump_to_lmdb = _module
make_bu_data = _module
prepro_feats = _module
prepro_labels = _module
prepro_ngrams = _module
prepro_reference_json = _module
clip_prepro_feats = _module
clipscore_prepro_feats = _module
prepro_labels = _module
setup = _module
eval = _module
eval_clip_retrieval = _module
eval_finecapeval = _module
finecapeval_inference = _module
train_pl = _module

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


import numpy.random as npr


import random


from functools import partial


import torch


import torch.utils.data as data


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import *


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import copy


import math


from random import shuffle


from random import seed


import string


import torchvision.models as models


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torch import nn


import functools


import logging


import torch.distributed as dist


import time


import collections


import torch.optim as optim


import torchvision.models.resnet


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from collections import OrderedDict


import re


import warnings


from typing import Union


from typing import List


from typing import Tuple


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import Sampler


import torchvision


import torchvision.transforms as T


from torch.utils.data.distributed import DistributedSampler


from collections import defaultdict


from torchvision import transforms as trn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedDotAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        self.d_k = d_model * scale // h
        self.h = h
        self.project_k_v = project_k_v
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)
        self.output_layer = nn.Linear(d_model * scale, d_model)
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x
        if self.use_aoa or not use_output_layer:
            del self.output_layer
            self.output_layer = lambda x: x
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            mask = mask.unsqueeze(1)
        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)
        nbatches = query.size(0)
        query = self.norm(query)
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query_, key_, value_, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if self.use_aoa:
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        x = self.output_layer(x)
        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class AoA_Refiner_Layer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1 + self.use_ff)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class AoA_Refiner_Core(nn.Module):

    def __init__(self, opt):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=1, scale=opt.multi_head_scale, do_aoa=opt.refine_aoa, norm_q=0, dropout_aoa=getattr(opt, 'dropout_aoa', 0.3))
        layer = AoA_Refiner_Layer(opt.rnn_size, attn, PositionwiseFeedForward(opt.rnn_size, 2048, 0.1) if opt.use_ff else None, 0.1)
        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Attention(nn.Module):

    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)
        weight = F.softmax(dot, dim=1)
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size)
            weight = weight / weight.sum(1, keepdim=True)
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)
        return att_res


class AoA_Decoder_Core(nn.Module):

    def __init__(self, opt):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = opt.multi_head_scale
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.decoder_type = getattr(opt, 'decoder_type', 'AoA')
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)
        self.out_drop = nn.Dropout(self.drop_prob_lm)
        if self.decoder_type == 'AoA':
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'LSTM':
            self.att2ctx = nn.LSTMCell(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size)
        else:
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size), nn.ReLU())
        if opt.use_multi_head == 2:
            self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0, scale=opt.multi_head_scale, use_output_layer=0, do_aoa=0, norm_q=1)
        else:
            self.attention = Attention(opt)
        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        h_att, c_att = self.att_lstm(torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        if self.use_multi_head == 2:
            att = self.attention(h_att, p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model), p_att_feats.narrow(2, self.multi_head_scale * self.d_model, self.multi_head_scale * self.d_model), att_masks)
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        ctx_input = torch.cat([att, h_att], 1)
        if self.decoder_type == 'LSTM':
            output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
            state = torch.stack((h_att, output)), torch.stack((c_att, c_logic))
        else:
            output = self.att2ctx(ctx_input)
            state = torch.stack((h_att, output)), torch.stack((c_att, state[1][1]))
        if self.out_res:
            output = output + h_att
        output = self.out_drop(output)
        return output, state


class AdaAtt_lstm(nn.Module):

    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.use_maxout = use_maxout
        self.w2h = nn.Linear(self.input_encoding_size, (4 + (use_maxout == True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4 + (use_maxout == True)) * self.rnn_size)
        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4 + (use_maxout == True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4 + (use_maxout == True)) * self.rnn_size) for _ in range(self.num_layers)])
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, xt, img_fc, state):
        hs = []
        cs = []
        for L in range(self.num_layers):
            prev_h = state[0][L]
            prev_c = state[1][L]
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L - 1](x)
            all_input_sums = i2h + self.h2h[L](prev_h)
            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            if not self.use_maxout:
                in_transform = torch.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size), in_transform.narrow(1, self.rnn_size, self.rnn_size))
            next_c = forget_gate * prev_c + in_gate * in_transform
            tanh_nex_c = torch.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers - 1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h + self.r_h2h(prev_h)
                fake_region = torch.sigmoid(n5) * tanh_nex_c
            cs.append(next_c)
            hs.append(next_h)
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)
        state = torch.cat([_.unsqueeze(0) for _ in hs], 0), torch.cat([_.unsqueeze(0) for _ in cs], 0)
        return top_h, fake_region, state


class AdaAtt_attention(nn.Module):

    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size
        self.fr_linear = nn.Sequential(nn.Linear(self.rnn_size, self.input_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.ho_linear = nn.Sequential(nn.Linear(self.rnn_size, self.input_encoding_size), nn.Tanh(), nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, att_masks=None):
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)
        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)
        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))
        img_all = torch.cat([fake_region.view(-1, 1, self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1, 1, self.input_encoding_size), conv_feat_embed], 1)
        hA = torch.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA, self.drop_prob_lm, self.training)
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim=1)
        if att_masks is not None:
            att_masks = att_masks.view(-1, att_size)
            PI = PI * torch.cat([att_masks[:, :1], att_masks], 1)
            PI = PI / PI.sum(1, keepdim=True)
        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)
        atten_out = visAttdim + h_out_linear
        h = torch.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h


class AdaAttCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats, att_masks)
        return atten_out, state


class UpDownCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(UpDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        lang_lstm_input = torch.cat([att, h_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang])
        return output, state


class LSTMCore(nn.Module):

    def __init__(self, opt):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, state):
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = torch.max(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size), all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)
        output = self.dropout(next_h)
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        return output, state


class StackAttCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(StackAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)
        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt)
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        h_0, state_0 = self.lstm0(torch.cat([xt, fc_feats], 1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0, att_res_1], 1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([h_1, att_res_2], 1), [state[0][2:3], state[1][2:3]])
        return h_2, [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]


class DenseAttCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(DenseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)
        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt)
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.fusion1 = nn.Sequential(nn.Linear(opt.rnn_size * 2, opt.rnn_size), nn.ReLU(), nn.Dropout(opt.drop_prob_lm))
        self.fusion2 = nn.Sequential(nn.Linear(opt.rnn_size * 3, opt.rnn_size), nn.ReLU(), nn.Dropout(opt.drop_prob_lm))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        h_0, state_0 = self.lstm0(torch.cat([xt, fc_feats], 1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0, att_res_1], 1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([self.fusion1(torch.cat([h_0, h_1], 1)), att_res_2], 1), [state[0][2:3], state[1][2:3]])
        return self.fusion2(torch.cat([h_0, h_1, h_2], 1)), [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]


class Att2in2Core(nn.Module):

    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + self.a2c(att_res)
        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size), in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)
        output = self.dropout(next_h)
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        return output, state


class Att2inCore(Att2in2Core):

    def __init__(self, opt):
        super(Att2inCore, self).__init__(opt)
        del self.a2c
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)


class Att2all2Core(nn.Module):

    def __init__(self, opt):
        super(Att2all2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.a2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1]) + self.a2h(att_res)
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size), in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)
        output = self.dropout(next_h)
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        return output, state


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, past=past)


class CaptionModel(nn.Module):

    def __init__(self):
        super(CaptionModel, self).__init__()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        def add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]
            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time]
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1), change.new_ones(batch_size, 1))
                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda
                else:
                    logprobs = logprobs - self.repeat_tensor(bdash, change) * diversity_lambda
            return logprobs, unaug_logprobs

        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)
            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]
            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:, :beam_size], ix[:, :beam_size]
            beam_ix = ix // vocab_size
            selected_ix = ix % vocab_size
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(-1)
            if t > 0:
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) == beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(beam_seq_logprobs))
            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + logprobs.reshape(batch_size, -1).gather(1, ix)
            assert (beam_logprobs_sum == ys).all()
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1, beam_ix.unsqueeze(-1).expand(-1, -1, vocab_size))
            assert (_tmp_beam_logprobs == beam_logprobs).all()
            beam_seq_logprobs = torch.cat([beam_seq_logprobs, beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)
            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
                new_state[_ix] = state[_ix][:, state_ix]
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size
        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, self.vocab_size + 1) for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash) for _ in range(group_size)]
        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]
        args = list(args)
        args = model_utils.split_tensors(group_size, args)
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)]
        else:
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]
        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.seq_length + divm - 1:
                    logprobs = logprobs_table[divm]
                    if decoding_constraint and t - divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, t - divm - 1].reshape(-1, 1), float('-inf'))
                    if remove_bad_endings and t - divm > 0:
                        logprobs[torch.from_numpy(np.isin(beam_seq_table[divm][:, :, t - divm - 1].cpu().numpy(), self.bad_endings_ix)).reshape(-1), 0] = float('-inf')
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobs.size(1) - 1)] == 'UNK':
                        logprobs[:, logprobs.size(1) - 1] = logprobs[:, logprobs.size(1) - 1] - 1000
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash)
                    beam_seq_table[divm], beam_seq_logprobs_table[divm], beam_logprobs_sum_table[divm], state_table[divm] = beam_step(logprobs, unaug_logprobs, bdash, t - divm, beam_seq_table[divm], beam_seq_logprobs_table[divm], beam_logprobs_sum_table[divm], state_table[divm])
                    for b in range(batch_size):
                        is_end = beam_seq_table[divm][b, :, t - divm] == self.eos_idx
                        assert beam_seq_table[divm].shape[-1] == t - divm + 1
                        if t == self.seq_length + divm - 1:
                            is_end.fill_(1)
                        for vix in range(bdash):
                            if is_end[vix]:
                                final_beam = {'seq': beam_seq_table[divm][b, vix].clone(), 'logps': beam_seq_logprobs_table[divm][b, vix].clone(), 'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(), 'p': beam_logprobs_sum_table[divm][b, vix].item()}
                                final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000
                    it = beam_seq_table[divm][:, :, t - divm].reshape(-1)
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it, *(args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)
        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)] for b in range(batch_size)]
        done_beams = [sum(_, []) for _ in done_beams_table]
        return done_beams

    def old_beam_search(self, init_state, init_logprobs, *args, **kwargs):

        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):
                for q in range(rows):
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': unaug_logprobsf[q]})
            candidates = sorted(candidates, key=lambda x: -x['p'])
            new_state = [_.clone() for _ in state]
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]
                beam_seq[t, vix] = v['c']
                beam_seq_logprobs[t, vix] = v['r']
                beam_logprobs_sum[vix] = v['p']
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash, self.vocab_size + 1).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]
        done_beams_table = [[] for _ in range(group_size)]
        state_table = list(zip(*[_.chunk(group_size, 1) for _ in init_state]))
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        args = list(args)
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[(_.chunk(group_size) if _ is not None else [None] * group_size) for _ in args_] for args_ in args]
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)]
        else:
            args = [(_.chunk(group_size) if _ is not None else [None] * group_size) for _ in args]
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]
        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.seq_length + divm - 1:
                    logprobsf = logprobs_table[divm]
                    if decoding_constraint and t - divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t - divm - 1].unsqueeze(1), float('-inf'))
                    if remove_bad_endings and t - divm > 0:
                        logprobsf[torch.from_numpy(np.isin(beam_seq_table[divm][t - divm - 1].cpu().numpy(), self.bad_endings_ix)), 0] = float('-inf')
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobsf.size(1) - 1)] == 'UNK':
                        logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000
                    unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash)
                    beam_seq_table[divm], beam_seq_logprobs_table[divm], beam_logprobs_sum_table[divm], state_table[divm], candidates_divm = beam_step(logprobsf, unaug_logprobsf, bdash, t - divm, beam_seq_table[divm], beam_seq_logprobs_table[divm], beam_logprobs_sum_table[divm], state_table[divm])
                    for vix in range(bdash):
                        if beam_seq_table[divm][t - divm, vix] == self.eos_idx or t == self.seq_length + divm - 1:
                            final_beam = {'seq': beam_seq_table[divm][:, vix].clone(), 'logps': beam_seq_logprobs_table[divm][:, vix].clone(), 'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(), 'p': beam_logprobs_sum_table[divm][vix].item()}
                            final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
                            done_beams_table[divm].append(final_beam)
                            beam_logprobs_sum_table[divm][vix] = -1000
                    it = beam_seq_table[divm][t - divm]
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it, *(args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = sum(done_beams_table, [])
        return done_beams

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        elif sample_method == 'gumbel':

            def sample_gumbel(shape, eps=1e-20):
                U = torch.rand(shape)
                return -torch.log(-torch.log(U + eps) + eps)

            def gumbel_softmax_sample(logits, temperature):
                y = logits + sample_gumbel(logits.size())
                return F.log_softmax(y / temperature, dim=-1)
            _logprobs = gumbel_softmax_sample(logprobs, temperature)
            _, it = torch.max(_logprobs.data, 1)
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        else:
            logprobs = logprobs / temperature
            if sample_method.startswith('top'):
                top_num = float(sample_method[3:])
                if 0 < top_num < 1:
                    probs = F.softmax(logprobs, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
                    sorted_probs = sorted_probs * mask
                    sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                    logprobs.scatter_(1, sorted_indices, sorted_probs.log())
                else:
                    the_k = int(top_num)
                    tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                    topk, indices = torch.topk(logprobs, the_k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprobs = tmp
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        return it, sampleLogprobs

    def decode_sequence(self, seq):
        return utils.decode_sequence(self.vocab, seq)


class FCModel(CaptionModel):

    def __init__(self, opt):
        super(FCModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.ss_prob = 0.0
        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = LSTMCore(opt)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = self.logit.weight
        if self.rnn_type == 'lstm':
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size), weight.new_zeros(self.num_layers, bsz, self.rnn_size)
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size * seq_per_img)
        outputs = []
        if seq_per_img > 1:
            fc_feats = utils.repeat_tensors(seq_per_img, fc_feats)
        for i in range(seq.size(1) + 1):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size * seq_per_img).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i - 1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i - 1].data.clone()
                        prob_prev = torch.exp(outputs[-1].data)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = seq[:, i - 1].clone()
                if i >= 2 and seq[:, i - 1].sum() == 0:
                    break
                xt = self.embed(it)
            output, state = self.core(xt, state)
            output = F.log_softmax(self.logit(output), dim=1)
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        xt = self.embed(it)
        output, state = self.core(xt, state)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size, self.vocab_size + 1)
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k + 1]).expand(beam_size, self.input_encoding_size)
                elif t == 1:
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(it)
                output, state = self.core(xt, state)
                logprobs = F.log_softmax(self.logit(output), dim=1)
            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, opt)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)
            output, state = self.core(xt, state)
            logprobs = F.log_softmax(self.logit(output), dim=1)
            if t == self.seq_length + 1:
                break
            if sample_method == 'greedy':
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)
                it = it.view(-1).long()
            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished & (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it
                seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break
        return seq, seqLogprobs


class ShowTellModel(CaptionModel):

    def __init__(self, opt):
        super(ShowTellModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.ss_prob = 0.0
        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = self.logit.weight
        if self.rnn_type == 'lstm':
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size), weight.new_zeros(self.num_layers, bsz, self.rnn_size)
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size * seq_per_img)
        outputs = []
        if seq_per_img > 1:
            fc_feats = utils.repeat_tensors(seq_per_img, fc_feats)
        for i in range(seq.size(1) + 1):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size * seq_per_img).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i - 1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i - 1].data.clone()
                        prob_prev = torch.exp(outputs[-1].data)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = seq[:, i - 1].clone()
                if i >= 2 and seq[:, i - 1].data.sum() == 0:
                    break
                xt = self.embed(it)
            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        xt = self.embed(it)
        output, state = self.core(xt.unsqueeze(0), state)
        logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)
        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k + 1]).expand(beam_size, self.input_encoding_size)
                elif t == 1:
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(it)
                output, state = self.core(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)
            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self.sample_beam(fc_feats, att_feats, opt)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)
            output, state = self.core(xt.unsqueeze(0), state)
            logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)
            if t == self.seq_length + 1:
                break
            if sample_method == 'greedy':
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)
                it = it.view(-1).long()
            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished & (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it
                seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break
        return seq, seqLogprobs


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask, layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        """Follow Figure 1 (right) for connections."""
        m = memory
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        """Implements Figure 2"""
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not layer_past.shape[2] == key.shape[1] > 1:
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])
        query, key, value = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for x in [query, key, value]]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def contrastive_loss(logits: torch.Tensor, dim: int) ->torch.Tensor:
    neg_ce = torch.diag(nn.functional.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def clip_loss_fn(similarity: torch.Tensor) ->torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0


class CLIPScore(nn.Module):

    def __init__(self, clipscore_w=2.5, image_size=224, mode='clip_s', use_grammar=False, joint_out=False):
        super(CLIPScore, self).__init__()
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model.eval()
        self.clipscore_w = clipscore_w
        self.image_transform = self._transform(image_size)
        self.mode = mode
        assert mode in ['clip_s', 'refclip_s']
        self.use_grammar = use_grammar
        self.joint_out = joint_out
        if self.use_grammar and self.joint_out is False:
            self.grammar_score_head = nn.Sequential(nn.Linear(self.clip_model.text_embed_dim, self.clip_model.projection_dim, bias=False), nn.ReLU(), nn.Linear(self.clip_model.projection_dim, 2, bias=False))

    def _transform(self, n_px):
        return Compose([Resize(n_px, interpolation=Image.BICUBIC), CenterCrop(n_px), lambda image: image.convert('RGB'), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    def load_image(self, image_path):
        image = Image.open(image_path)
        return image

    def image_extract(self, image):
        if isinstance(image, str):
            image = self.load_image(image)
        if not isinstance(image, torch.Tensor):
            image = self.image_transform(image)
        img_tensor = image.view(-1, 3, 224, 224)
        device = next(self.clip_model.parameters()).device
        img_tensor = img_tensor
        clip_model = self.clip_model
        img_feat = clip_model.vision_model(img_tensor).pooler_output
        img_feat = clip_model.visual_projection(img_feat)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        return img_feat

    def text_extract(self, text, prompt='A photo depicts', proj_norm=True):
        if isinstance(text, str):
            text_batch = [' '.join([prompt, text])]
        elif isinstance(text, list):
            text_batch = [' '.join([prompt, txt]) for txt in text]
        if isinstance(text, tuple) and isinstance(text[0], torch.Tensor):
            input_ids, attention_mask = text
        else:
            input_text = text_batch
            tokenized = self.tokenizer(input_text, return_tensors='pt', padding=True)
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
        clip_model = self.clip_model
        device = next(self.clip_model.parameters()).device
        input_ids = input_ids
        attention_mask = attention_mask
        text_feat = clip_model.text_model(input_ids, attention_mask).pooler_output
        if proj_norm:
            text_feat = clip_model.text_projection(text_feat)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    def calc_clip_s(self, img_feat, text_feat):
        return self.clipscore_w * torch.relu((img_feat * text_feat).sum(dim=-1))

    def calc_refclip_s(self, img_feat=None, text_feat=None, ref_text_feat=None, ref_text_mask=None, clip_s=None):
        if clip_s is None:
            clip_s = self.calc_clip_s(img_feat, text_feat)
        B, dim = img_feat.size()
        ref_text_feat = ref_text_feat.view(B, -1, dim)
        K = ref_text_feat.size(1)
        text_feat = text_feat.view(B, 1, dim).expand(-1, K, -1)
        assert ref_text_feat.size() == text_feat.size(), (ref_text_feat.size(), text_feat.size())
        ref_score = self.calc_clip_s(text_feat, ref_text_feat)
        if ref_text_mask is not None:
            if not isinstance(ref_text_mask, torch.Tensor):
                ref_text_mask = torch.tensor(ref_text_mask, dtype=ref_score.dtype, device=ref_score.device)
            ref_score = ref_score.view(B, K) * ref_text_mask.view(B, K)
        ref_score = ref_score.view(B, K).max(dim=1).values
        assert clip_s.size() == (B,)
        assert clip_s.size() == ref_score.size()
        refclip_s = 2 / (1 / clip_s + 1 / ref_score)
        return refclip_s

    def train_step(self, images=None, text=None, img_feat=None, text_feat=None, neg_text=None, neg_text_feat=None, prompt='A photo depicts', **kwargs):
        if img_feat is None:
            img_feat = self.image_extract(images)
        img_feat = img_feat.view(-1, 512)
        B = img_feat.size(0)
        if self.joint_out:
            pos_text_feat = self.text_extract(text, prompt=prompt, proj_norm=False).view(B, 512)
            neg_text_feat = self.text_extract(neg_text, prompt=prompt, proj_norm=False).view(-1, 512)
            neg_B = neg_text_feat.size(0)
            text_feat = torch.cat([pos_text_feat, neg_text_feat], dim=0)
            text_cont_feat = self.clip_model.text_projection(text_feat)
            text_cont_feat = text_cont_feat / text_cont_feat.norm(dim=-1, keepdim=True)
            text_cont_feat = text_cont_feat.view(B + neg_B, 512)
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_text = torch.matmul(text_cont_feat, img_feat.t()) * logit_scale
            caption_loss = -torch.diag(nn.functional.log_softmax(logits_per_text, dim=0)[:B]).mean()
            image_loss = -torch.diag(nn.functional.log_softmax(logits_per_text[:B], dim=1)).mean()
            clip_loss = (caption_loss + image_loss) / 2.0
            out = {'clip_loss': clip_loss, 'img_feat': img_feat, 'text_feat': text_cont_feat[:B].detach()}
            return out
        else:
            if text_feat is None:
                text_feat = self.text_extract(text, prompt=prompt, proj_norm=False)
            text_cont_feat = self.clip_model.text_projection(text_feat)
            text_cont_feat = text_cont_feat / text_cont_feat.norm(dim=-1, keepdim=True)
            text_cont_feat = text_cont_feat.view(B, 512)
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_text = torch.matmul(text_cont_feat, img_feat.t()) * logit_scale
            clip_loss = clip_loss_fn(logits_per_text)
            pos_text_feat = text_feat.view(B, 512)
            neg_text_feat = self.text_extract(neg_text, prompt=prompt, proj_norm=False).view(B, 512)
            grammar_text_feat = torch.cat([pos_text_feat, neg_text_feat], dim=0)
            grammar_text_logit = self.grammar_score_head(grammar_text_feat)
            grammar_labels = torch.LongTensor([1] * B + [0] * B).view(2 * B)
            grammar_loss = torch.nn.functional.cross_entropy(grammar_text_logit, grammar_labels)
            grammar_pred = grammar_text_logit.argmax(dim=1, keepdim=False)
            grammar_pos_pred = grammar_pred[:B]
            grammar_neg_pred = grammar_pred[B:]
            out = {'clip_loss': clip_loss, 'grammar_loss': grammar_loss, 'img_feat': img_feat, 'text_feat': text_cont_feat, 'neg_text_feat': neg_text_feat, 'grammar_pos_pred': grammar_pos_pred, 'grammar_neg_pred': grammar_neg_pred}
            return out

    def train_step_old(self, images=None, text=None, img_feat=None, text_feat=None, neg_text=None, neg_text_feat=None, prompt='A photo depicts', **kwargs):
        if img_feat is None:
            img_feat = self.image_extract(images)
        img_feat = img_feat.view(-1, 512)
        B = img_feat.size(0)
        if text_feat is None:
            text_feat = self.text_extract(text, prompt=prompt, proj_norm=False)
            text_cont_feat = self.clip_model.text_projection(text_feat)
            text_cont_feat = text_cont_feat / text_cont_feat.norm(dim=-1, keepdim=True)
        text_cont_feat = text_cont_feat.view(B, 512)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_cont_feat, img_feat.t()) * logit_scale
        clip_loss = clip_loss_fn(logits_per_text)
        pos_text_feat = text_feat.view(B, 512)
        neg_text_feat = self.text_extract(neg_text, prompt=prompt, proj_norm=False).view(B, 512)
        grammar_text_feat = torch.cat([pos_text_feat, neg_text_feat], dim=0)
        grammar_text_logit = self.grammar_score_head(grammar_text_feat)
        grammar_labels = torch.LongTensor([1] * B + [0] * B).view(2 * B)
        grammar_loss = torch.nn.functional.cross_entropy(grammar_text_logit, grammar_labels)
        grammar_pred = grammar_text_logit.argmax(dim=1, keepdim=False)
        grammar_pos_pred = grammar_pred[:B]
        grammar_neg_pred = grammar_pred[B:]
        out = {'clip_loss': clip_loss, 'grammar_loss': grammar_loss, 'img_feat': img_feat, 'text_feat': text_cont_feat, 'neg_text_feat': neg_text_feat, 'grammar_pos_pred': grammar_pos_pred, 'grammar_neg_pred': grammar_neg_pred}
        return out


bad_endings = ['with', 'in', 'on', 'of', 'a', 'at', 'to', 'for', 'an', 'this', 'his', 'her', 'that']


def decode_sequence(ix_to_word, seq):
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j - 1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words) + flag])
        out.append(txt.replace('@@ ', ''))
    return out


def get_self_critical_clipscore_reward(greedy_res, data_gts, gen_result, opt, clipscore_model, clip_vis_feats, vocab):
    batch_size = len(data_gts)
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts)
    assert greedy_res.shape[0] == batch_size
    B = batch_size
    K = seq_per_img
    L = gen_result.shape[1]
    assert gen_result.shape == (B * K, L)
    if clipscore_model.mode == 'refclip_s':
        gts = []
        gts_valid_mask = []
        max_n_refs = max([len(_gts) for _gts in data_gts])
        for i in range(len(data_gts)):
            _gts = decode_sequence(vocab, data_gts[i])
            n_ref = len(_gts)
            _gts.extend([''] * (max_n_refs - n_ref))
            gts.extend(_gts)
            gts_valid_mask.extend([1] * n_ref + [0] * (max_n_refs - n_ref))
        assert len(gts) == B * max_n_refs
        assert len(gts_valid_mask) == B * max_n_refs
    if opt.clipscore_reward_weight > 0:
        with torch.no_grad():
            clipscore_model.eval()
            gen_result = gen_result.data.cpu().numpy()
            res = decode_sequence(vocab, gen_result)
            assert len(res) == B * K, len(res)
            if getattr(opt, 'use_grammar', False) and not getattr(opt, 'joint_out', False):
                text_pre_feat = clipscore_model.text_extract(res, proj_norm=False)
                grammar_logit = clipscore_model.grammar_score_head(text_pre_feat.view(-1, 512))
                grammar_prob = torch.softmax(grammar_logit, dim=-1)[:, 1]
                grammar_prob = grammar_prob.view(B * K).detach()
                text_feat = clipscore_model.clip_model.text_projection(text_pre_feat)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            else:
                text_feat = clipscore_model.text_extract(res)
            assert text_feat.size() == (B * K, 512), text_feat.size()
            assert clip_vis_feats.size() == (B, 512), clip_vis_feats.size()
            vis_feat = clip_vis_feats.view(B, 1, -1).expand(-1, K, -1).contiguous().view(B * K, -1)
            clip_s = clipscore_model(text_feat=text_feat, img_feat=vis_feat, mode='clip_s')
            clip_s = clip_s.view(B * K).detach()
            if clipscore_model.mode == 'refclip_s':
                ref_text_feat = clipscore_model.text_extract(gts)
                ref_text_mask = torch.tensor(gts_valid_mask, dtype=ref_text_feat.dtype, device=ref_text_feat.device)
                assert ref_text_feat.size() == (B * max_n_refs, 512), ref_text_feat.size()
                assert ref_text_mask.size() == (B * max_n_refs,), ref_text_mask.size()
                refclip_s = clipscore_model.calc_refclip_s(text_feat=text_feat, img_feat=vis_feat, ref_text_feat=ref_text_feat.view(B, 1, max_n_refs, -1).expand(-1, K, -1, -1).contiguous().view(B * K * max_n_refs, -1), ref_text_mask=ref_text_mask.view(B, 1, max_n_refs).expand(-1, K, -1).contiguous().view(B * K * max_n_refs), clip_s=clip_s)
                refclip_s = refclip_s.view(B * K).detach()
            greedy_res = greedy_res.data.cpu().numpy()
            res = decode_sequence(vocab, greedy_res)
            assert len(res) == B, len(res)
            if getattr(opt, 'use_grammar', False) and getattr(opt, 'use_grammar_baseline', False) and not getattr(opt, 'joint_out', False):
                text_pre_feat = clipscore_model.text_extract(res, proj_norm=False)
                grammar_logit = clipscore_model.grammar_score_head(text_pre_feat.view(-1, 512))
                grammar_prob_baseline = torch.softmax(grammar_logit, dim=-1)[:, 1]
                grammar_prob_baseline = grammar_prob_baseline.view(B).detach()
                text_feat = clipscore_model.clip_model.text_projection(text_pre_feat)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            else:
                text_feat = clipscore_model.text_extract(res)
            assert text_feat.size() == (B, 512), text_feat.size()
            assert clip_vis_feats.size() == (B, 512), clip_vis_feats.size()
            vis_feat = clip_vis_feats.view(B, 512)
            clip_s_baseline = clipscore_model(text_feat=text_feat, img_feat=vis_feat, mode='clip_s')
            clip_s_baseline = clip_s_baseline.view(B).detach()
            if clipscore_model.mode == 'refclip_s':
                refclip_s_baseline = clipscore_model.calc_refclip_s(text_feat=text_feat, img_feat=vis_feat, ref_text_feat=ref_text_feat, ref_text_mask=ref_text_mask, clip_s=clip_s_baseline)
                refclip_s_baseline = refclip_s_baseline.view(B).detach()
            if clipscore_model.mode == 'clip_s':
                rewards = clip_s - clip_s_baseline.view(B, 1).expand(-1, K).contiguous().flatten()
                unnormalized_mean_reward = clip_s.mean()
            elif clipscore_model.mode == 'refclip_s':
                rewards = refclip_s - refclip_s_baseline.view(B, 1).expand(-1, K).contiguous().flatten()
                unnormalized_mean_reward = refclip_s.mean()
            if getattr(opt, 'use_grammar', False) and not getattr(opt, 'joint_out', False):
                if getattr(opt, 'use_grammar_baseline', False):
                    grammar_rewards = grammar_prob - grammar_prob_baseline.view(B, 1).expand(-1, K).contiguous().flatten()
                else:
                    grammar_rewards = grammar_prob
            else:
                grammar_rewards = None
        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        elif clipscore_model.mode == 'clip_s':
            None
        elif clipscore_model.mode == 'refclip_s':
            None
    else:
        rewards = torch.zeros(B, L)
        unnormalized_mean_reward = None
        grammar_rewards = None
    rewards = opt.clipscore_reward_weight * rewards
    rewards = rewards.view(-1, 1).expand(-1, L).contiguous()
    if getattr(opt, 'use_grammar', False) and not getattr(opt, 'joint_out', False):
        grammar_rewards = grammar_rewards.view(-1, 1).expand(-1, L).contiguous()
    return rewards, unnormalized_mean_reward, grammar_rewards


Bleu_scorer = None


CiderD_scorer = None


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
    batch_size = len(data_gts)
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts)
    assert greedy_res.shape[0] == batch_size
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]
    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({(i + gen_result_size): gts[i] for i in range(batch_size)})
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts_, res_)
        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        else:
            None
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts_, res__)
        bleu_scores = np.array(bleu_scores[3])
        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        else:
            None
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores
    unnormalized_reward_mean = scores[:gen_result_size].flatten().mean()
    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size)
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    return rewards, unnormalized_reward_mean


class LossWrapper(torch.nn.Module):

    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)
        self.clipscore_model = None
        if self.opt.use_clipscore:
            use_grammar = getattr(self.opt, 'use_grammar', False)
            joint_out = getattr(self.opt, 'joint_out', False)
            self.clipscore_model = CLIPScore(mode=opt.clipscore_mode, use_grammar=use_grammar, joint_out=joint_out)
            for p in self.clipscore_model.parameters():
                p.requires_grad = False
            if use_grammar:
                state_dict = torch.load(self.opt.clip_load_path, map_location='cpu')
                self.clipscore_model.load_state_dict(state_dict['state_dict'])

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices, sc_flag, struc_flag, clip_vis_feats=None):
        opt = self.opt
        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method': opt.train_sample_method, 'beam_size': opt.train_beam_size, 'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin' or not 'margin' in opt.structure_loss_type, 'sample_n': opt.train_sample_n}, mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats), 'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1 - opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample', opt={'sample_method': opt.sc_sample_method, 'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method': opt.train_sample_method, 'beam_size': opt.train_beam_size, 'sample_n': opt.train_sample_n}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            if getattr(self.opt, 'use_multi_rewards', False):
                assert self.opt.use_clipscore
                clipscore_reward_normalized, clipscore_unnormalized_mean, grammar_rewards = get_self_critical_clipscore_reward(greedy_res, gts, gen_result, self.opt, self.clipscore_model, clip_vis_feats, self.model.vocab)
                if self.opt.clipscore_mode == 'clip_s':
                    out['CLIP-S'] = clipscore_unnormalized_mean
                elif self.opt.clipscore_mode == 'refclip_s':
                    out['RefCLIP-S'] = clipscore_unnormalized_mean
                if getattr(self.opt, 'use_grammar', False):
                    out['grammar_reward'] = grammar_rewards.mean()
                    reward = clipscore_reward_normalized + grammar_rewards
                else:
                    assert grammar_rewards is None
                    cider_reward_normalized, cider_unnormalized_mean = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
                    out['CIDEr'] = cider_unnormalized_mean
                    if isinstance(cider_reward_normalized, np.ndarray):
                        cider_reward_normalized = torch.from_numpy(cider_reward_normalized)
                    reward = clipscore_reward_normalized + cider_reward_normalized
            elif self.opt.use_clipscore:
                clipscore_reward_normalized, clipscore_unnormalized_mean, _ = get_self_critical_clipscore_reward(greedy_res, gts, gen_result, self.opt, self.clipscore_model, clip_vis_feats, self.model.vocab)
                if self.opt.clipscore_mode == 'clip_s':
                    out['CLIP-S'] = clipscore_unnormalized_mean
                elif self.opt.clipscore_mode == 'refclip_s':
                    out['RefCLIP-S'] = clipscore_unnormalized_mean
                reward = clipscore_reward_normalized
            else:
                cider_reward_normalized, cider_unnormalized_mean = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
                out['CIDEr'] = cider_unnormalized_mean
                reward = cider_reward_normalized
            if isinstance(reward, np.ndarray):
                reward = torch.from_numpy(reward)
            reward = reward
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:, 0].mean()
        out['loss'] = loss
        return out


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = seq > 0
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = -input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def get_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)
    seq_per_img = batch_size // len(data_gts)
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]
    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        else:
            None
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        else:
            None
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores
    return scores


Cider_scorer = None


def get_self_cider_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)
    seq_per_img = batch_size // len(data_gts)
    res = []
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res.append(array_to_str(gen_result[i]))
    scores = []
    for i in range(len(data_gts)):
        tmp = Cider_scorer.my_self_cider([res[i * seq_per_img:(i + 1) * seq_per_img]])

        def get_div(eigvals):
            eigvals = np.clip(eigvals, 0, None)
            return -np.log(np.sqrt(eigvals[-1]) / np.sqrt(eigvals).sum()) / np.log(len(eigvals))
        scores.append(get_div(np.linalg.eigvalsh(tmp[0] / 10)))
    scores = np.array(scores)
    return scores


class StructureLosses(nn.Module):
    """
    This loss is inspired by Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018).
    """

    def __init__(self, opt):
        super(StructureLosses, self).__init__()
        self.opt = opt
        self.loss_type = opt.structure_loss_type

    def forward(self, input, seq, data_gts):
        """
        Input is either logits or log softmax
        """
        out = {}
        batch_size = input.size(0)
        seq_per_img = batch_size // len(data_gts)
        assert seq_per_img == self.opt.train_sample_n, seq_per_img
        mask = seq > 0
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)
        scores = get_scores(data_gts, seq, self.opt)
        scores = torch.from_numpy(scores).type_as(input).view(-1, seq_per_img)
        out['reward'] = scores
        if self.opt.entropy_reward_weight > 0:
            entropy = -(F.softmax(input, dim=2) * F.log_softmax(input, dim=2)).sum(2).data
            entropy = (entropy * mask).sum(1) / mask.sum(1)
            None
            scores = scores + self.opt.entropy_reward_weight * entropy.view(-1, seq_per_img)
        costs = -scores
        if self.loss_type == 'risk' or self.loss_type == 'softmax_margin':
            costs = costs - costs.min(1, keepdim=True)[0]
            costs = costs / costs.max(1, keepdim=True)[0]
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        if self.loss_type == 'seqnll':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)
        elif self.loss_type == 'risk':
            input = input * mask
            input = input.sum(1)
            input = input.view(-1, seq_per_img)
            output = (F.softmax(input.exp()) * costs).sum(1).mean()
        elif self.loss_type == 'max_margin':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input).max(1)[0] / 2
            output = output.mean()
        elif self.loss_type == 'multi_margin':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input)
            output = output.mean()
        elif self.loss_type == 'softmax_margin':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)
        elif self.loss_type == 'real_softmax_margin':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)
        elif self.loss_type == 'new_self_critical':
            """
            A different self critical
            Self critical uses greedy decoding score as baseline;
            This setting uses the average score of the rest samples as baseline
            (suppose c1...cn n samples, reward1 = score1 - 1/(n-1)(score2+..+scoren) )
            """
            baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)
            scores = scores - baseline
            if getattr(self.opt, 'self_cider_reward_weight', 0) > 0:
                _scores = get_self_cider_scores(data_gts, seq, self.opt)
                _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1)
                _scores = _scores.expand_as(scores - 1)
                scores += self.opt.self_cider_reward_weight * _scores
            output = -input * mask * scores.view(-1, 1)
            output = torch.sum(output) / torch.sum(mask)
        out['loss'] = output
        return out


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1)
        self.size = input.size(1)
        true_dist = input.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()


class myResnet(nn.Module):

    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)
        return fc, att


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[0, :, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=torch.ones_like(self.q_proj.weight), out_proj_bias=torch.zeros_like(self.q_proj.bias), use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        attnpool = self.attnpool(x)
        return x, attnpool


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x


class CLIP(nn.Module):

    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int], vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaAtt_attention,
     lambda: ([], {'opt': _mock_config(input_encoding_size=4, rnn_size=4, drop_prob_lm=0.5, att_hid_size=4)}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (Attention,
     lambda: ([], {'opt': _mock_config(rnn_size=4, att_hid_size=4)}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Generator,
     lambda: ([], {'d_model': 4, 'vocab': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LanguageModelCriterion,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4], dtype=torch.int64), torch.rand([4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiHeadedAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiHeadedDotAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (RewardCriterion,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4], dtype=torch.int64), torch.rand([4, 4])], {}),
     False),
    (SublayerConnection,
     lambda: ([], {'size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_j_min_CLIP_Caption_Reward(_paritybench_base):
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

