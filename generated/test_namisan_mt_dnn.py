import sys
_module = sys.modules[__name__]
del sys
calc_metrics = _module
data_utils = _module
log_wrapper = _module
metrics = _module
mrc_eval = _module
my_statics = _module
roberta_utils = _module
squad_eval = _module
task_def = _module
tokenizer_utils = _module
utils = _module
utils_qa = _module
vocab = _module
experiments = _module
common_utils = _module
domain_prepro = _module
extractor = _module
exp_def = _module
glue_prepro = _module
glue_utils = _module
mlm_utils = _module
ner_utils = _module
prepro = _module
squad_prepro = _module
superglue_fairseq = _module
superglue_prepro = _module
superglue_utils = _module
extract_cat = _module
xnli_eval = _module
xnli_prepro = _module
int_test_encoder = _module
int_test_prepro_std = _module
module = _module
common = _module
dropout_wrapper = _module
my_optim = _module
pooler = _module
san = _module
san_model = _module
similarity = _module
sub_layers = _module
mt_dnn = _module
batcher = _module
inference = _module
loss = _module
matcher = _module
model = _module
optim = _module
perturbation = _module
predict = _module
prepare_distillation_data = _module
prepro_gen_std = _module
prepro_std = _module
pretrained_models = _module
tasks = _module
classification_task = _module
ranking_task = _module
regression_task = _module
seqencelabeling_task = _module
sequence_gen_task = _module
span_classification_task = _module
_test_train = _module
test_prepro = _module
train = _module

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


import numpy


import collections


import logging


from typing import Optional


from typing import Tuple


import numpy as np


from torch.utils.data import DataLoader


import numpy.ma as ma


import math


from torch.nn.init import uniform


from torch.nn.init import normal


from torch.nn.init import eye


from torch.nn.init import xavier_uniform


from torch.nn.init import xavier_normal


from torch.nn.init import kaiming_uniform


from torch.nn.init import kaiming_normal


from torch.nn.init import orthogonal


import torch.nn as nn


import torch.nn.functional as F


from copy import deepcopy


from torch.nn import Parameter


from functools import wraps


from torch.nn.utils import weight_norm


from torch.nn.parameter import Parameter


import copy


from collections import Sequence


from torch.utils.data import Dataset


from torch.utils.data import BatchSampler


from torch.utils.data import Sampler


import enum


from numpy.lib.arraysetops import isin


from numpy.lib.function_base import insert


from torch.nn.modules.loss import _Loss


from enum import IntEnum


from torch.nn.modules.normalization import LayerNorm


import torch.optim as optim


from torch.optim.lr_scheduler import *


from torch.optim import Optimizer


class DropoutWrapper(nn.Module):
    """
    This is a dropout wrapper which supports the fix mask dropout
    """

    def __init__(self, dropout_p=0, enable_vbp=True):
        super(DropoutWrapper, self).__init__()
        """variational dropout means fix dropout mask
        ref: https://discuss.pytorch.org/t/dropout-for-rnns/633/11
        """
        self.enable_variational_dropout = enable_vbp
        self.dropout_p = dropout_p

    def forward(self, x):
        """
        :param x: batch * len * input_size
        """
        if self.training == False or self.dropout_p == 0:
            return x
        if len(x.size()) == 3:
            mask = 1.0 / (1 - self.dropout_p) * torch.bernoulli((1 - self.dropout_p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1))
            mask.requires_grad = False
            return mask.unsqueeze(1).expand_as(x) * x
        else:
            return F.dropout(x, p=self.dropout_p, training=self.training)


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
                None
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


def linear(x):
    return x


def activation(func_a):
    """Activation function wrapper"""
    try:
        f = eval('nn.{}'.format(func_a))
    except:
        f = linear
    return f


class Pooler(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.1, actf='tanh'):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = activation(actf)
        self.dropout = DropoutWrapper(dropout_p=dropout_p)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        first_token_tensor = self.dropout(first_token_tensor)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Classifier(nn.Module):

    def __init__(self, x_size, y_size, opt, prefix='decoder', dropout=None):
        super(Classifier, self).__init__()
        self.opt = opt
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(prefix), 0))
        else:
            self.dropout = dropout
        self.merge_opt = opt.get('{}_merge_opt'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        if self.merge_opt == 1:
            self.proj = nn.Linear(x_size * 4, y_size)
        else:
            self.proj = nn.Linear(x_size * 2, y_size)
        if self.weight_norm_on:
            self.proj = weight_norm(self.proj)

    def forward(self, x1, x2, mask=None):
        if self.merge_opt == 1:
            x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        else:
            x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        scores = self.proj(x)
        return scores


class BilinearFlatSim(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """

    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(BilinearFlatSim, self).__init__()
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(y_size, x_size)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy


class FlatSim(nn.Module):

    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(FlatSim, self).__init__()
        assert x_size == y_size
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(x_size * 3, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)
        flat_x = torch.cat([x, y, x * y], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        return scores


class FlatSimV2(nn.Module):

    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(FlatSimV2, self).__init__()
        assert x_size == y_size
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(x_size * 4, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)
        flat_x = torch.cat([x, y, x * y, torch.abs(x - y)], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        return scores


class SimpleFlatSim(nn.Module):

    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(SimpleFlatSim, self).__init__()
        self.opt = opt
        self.weight_norm_on = opt.get('{}_norm_on'.format(prefix), False)
        self.linear = nn.Linear(y_size + x_size, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)
        flat_x = torch.cat([x, y], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        return scores


class FlatSimilarityWrapper(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(FlatSimilarityWrapper, self).__init__()
        self.score_func_str = opt.get('{}_att_type'.format(prefix), 'none').lower()
        self.att_dropout = DropoutWrapper(opt.get('{}_att_dropout'.format(prefix), 0))
        self.score_func = None
        if self.score_func_str == 'bilinear':
            self.score_func = BilinearFlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'simple':
            self.score_func = SimpleFlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'flatsim':
            self.score_func = FlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        else:
            self.score_func = FlatSimV2(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)

    def forward(self, x1, x2, mask):
        scores = self.score_func(x1, x2, mask)
        return scores


class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size, dropout=None):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.dropout = dropout

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, 1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class MLPSelfAttn(nn.Module):

    def __init__(self, input_size, opt={}, prefix='attn_sum', dropout=None):
        super(MLPSelfAttn, self).__init__()
        self.prefix = prefix
        self.FC = nn.Linear(input_size, input_size)
        self.linear = nn.Linear(input_size, 1)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        if self.layer_norm_on:
            self.FC = weight_norm(self.FC)

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(self.f(self.FC(x_flat))).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class SelfAttnWrapper(nn.Module):

    def __init__(self, input_size, prefix='attn_sum', opt={}, dropout=None):
        super(SelfAttnWrapper, self).__init__()
        """
        Self att wrapper, support linear and MLP
        """
        attn_type = opt.get('{}_type'.format(prefix), 'linear')
        if attn_type == 'mlp':
            self.att = MLPSelfAttn(input_size, prefix, opt, dropout)
        else:
            self.att = LinearSelfAttn(input_size, dropout)

    def forward(self, x, x_mask):
        return self.att(x, x_mask)


def generate_mask(new_data, dropout_p=0.0, is_training=False):
    if not is_training:
        dropout_p = 0.0
    new_data = (1 - dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1) - 1)
        new_data[i][one] = 1
    mask = 1.0 / (1 - dropout_p) * torch.bernoulli(new_data)
    mask.requires_grad = False
    return mask


class SANClassifier(nn.Module):
    """Implementation of Stochastic Answer Networks for Natural Language Inference, Xiaodong Liu, Kevin Duh and Jianfeng Gao
    https://arxiv.org/abs/1804.07888
    """

    def __init__(self, x_size, h_size, label_size, opt={}, prefix='decoder', dropout=None):
        super(SANClassifier, self).__init__()
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.prefix = prefix
        self.query_wsum = SelfAttnWrapper(x_size, prefix='mem_cum', opt=opt, dropout=self.dropout)
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.rnn_type = '{}{}'.format(opt.get('{}_rnn_type'.format(prefix), 'gru').upper(), 'Cell')
        self.rnn = getattr(nn, self.rnn_type)(x_size, h_size)
        self.num_turn = opt.get('{}_num_turn'.format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.label_size = label_size
        self.dump_state = opt.get('dump_state_on', False)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)
        if self.weight_norm_on:
            self.rnn = WN(self.rnn)
        self.classifier = Classifier(x_size, self.label_size, opt, prefix=prefix, dropout=self.dropout)

    def forward(self, x, h0, x_mask=None, h_mask=None):
        h0 = self.query_wsum(h0, h_mask)
        if type(self.rnn) is nn.LSTMCell:
            c0 = h0.new(h0.size()).zero_()
        scores_list = []
        for turn in range(self.num_turn):
            att_scores = self.attn(x, h0, x_mask)
            x_sum = torch.bmm(F.softmax(att_scores, 1).unsqueeze(1), x).squeeze(1)
            scores = self.classifier(x_sum, h0)
            scores_list.append(scores)
            if self.rnn is not None:
                h0 = self.dropout(h0)
                if type(self.rnn) is nn.LSTMCell:
                    h0, c0 = self.rnn(x_sum, (h0, c0))
                else:
                    h0 = self.rnn(x_sum, h0)
        if self.mem_type == 1:
            mask = generate_mask(self.alpha.data.new(x.size(0), self.num_turn), self.mem_random_drop, self.training)
            mask = [m.contiguous() for m in torch.unbind(mask, 1)]
            tmp_scores_list = [(mask[idx].view(x.size(0), 1).expand_as(inp) * F.softmax(inp, 1)) for idx, inp in enumerate(scores_list)]
            scores = torch.stack(tmp_scores_list, 2)
            scores = torch.mean(scores, 2)
            scores = torch.log(scores)
        else:
            scores = scores_list[-1]
        if self.dump_state:
            return scores, scores_list
        else:
            return scores


class MaskLmHeader(nn.Module):
    """Mask LM"""

    def __init__(self, embedding_weights=None, bias=False):
        super(MaskLmHeader, self).__init__()
        self.decoder = nn.Linear(embedding_weights.size(1), embedding_weights.size(0), bias=bias)
        self.decoder.weight = embedding_weights
        self.nsp = nn.Linear(embedding_weights.size(1), 2)

    def forward(self, hidden_states):
        mlm_out = self.decoder(hidden_states)
        nsp_out = self.nsp(hidden_states[:, 0, :])
        return mlm_out, nsp_out


class SanLayer(nn.Module):

    def __init__(self, num_hid, bidirect, dropout, rnn_type):
        super().__init__()
        assert isinstance(rnn_type, str)
        rnn_type = rnn_type.upper()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = getattr(nn, rnn_type)
        self._rnn = rnn_cls(num_hid, num_hid, 1, bidirectional=bidirect, dropout=dropout, batch_first=True)
        self._layer_norm = nn.LayerNorm(num_hid, eps=1e-12)
        self.rnn_type = rnn_type
        self.num_hid = num_hid
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = self.ndirections, batch, self.num_hid
        if self.rnn_type == 'LSTM':
            return weight.new(*hid_shape).zero_(), weight.new(*hid_shape).zero_()
        else:
            return weight.new(*hid_shape).zero_()

    def forward(self, x, attention_mask):
        self._rnn.flatten_parameters()
        batch = x.size(0)
        hidden0 = self.init_hidden(batch)
        tmp_output = self._rnn(x, hidden0)[0]
        if self.ndirections > 1:
            size = tmp_output.shape
            tmp_output = tmp_output.view(size[0], size[1], self.num_hid, 2).max(-1)[0]
        output = self._layer_norm(x + tmp_output)
        return output


class SanEncoder(nn.Module):

    def __init__(self, num_hid, nlayers, bidirect, dropout, rnn_type='LSTM'):
        super().__init__()
        layer = SanLayer(num_hid, bidirect, dropout, rnn_type)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(nlayers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class SanPooler(nn.Module):

    def __init__(self, hidden_size, dropout_p):
        super().__init__()
        my_dropout = DropoutWrapper(dropout_p, False)
        self.self_att = SelfAttnWrapper(hidden_size, dropout=my_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        """
        Arguments:
            hidden_states {FloatTensor} -- shape (batch, seq_len, hidden_size)
            attention_mask {ByteTensor} -- 1 indicates padded token
        """
        first_token_tensor = self.self_att(hidden_states, attention_mask)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DotProduct(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(DotProduct, self).__init__()
        assert x1_dim == x2_dim
        self.opt = opt
        self.prefix = prefix
        self.scale_on = opt.get('{}_scale'.format(self.prefix), False)
        self.scalor = 1.0 / numpy.power(x2_dim, 0.5)

    def forward(self, x1, x2):
        assert x1.size(2) == x2.size(2)
        scores = x1.bmm(x2.transpose(1, 2))
        if self.scale_on:
            scores *= self.scalor
        return scores


class DotProductProject(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(DotProductProject, self).__init__()
        self.prefix = prefix
        self.opt = opt
        self.hidden_size = opt.get('{}_hidden_size'.format(self.prefix), 64)
        self.residual_on = opt.get('{}_residual_on'.format(self.prefix), False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.share = opt.get('{}_share'.format(self.prefix), False)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        self.scale_on = opt.get('{}_scale_on'.format(self.prefix), False)
        self.dropout = dropout
        x1_in_dim = x1_dim
        x2_in_dim = x2_dim
        out_dim = self.hidden_size
        self.proj_1 = nn.Linear(x1_in_dim, out_dim, bias=False)
        if self.layer_norm_on:
            self.proj_1 = weight_norm(self.proj_1)
        if self.share and x1_in_dim == x2_in_dim:
            self.proj_2 = self.proj_1
        else:
            self.proj_2 = nn.Linear(x2_in_dim, out_dim)
            if self.layer_norm_on:
                self.proj_2 = weight_norm(self.proj_2)
        if self.scale_on:
            self.scalar = Parameter(torch.ones(1, 1, 1) / self.hidden_size ** 0.5, requires_grad=False)
        else:
            self.sclalar = Parameter(torch.ones(1, 1, self.hidden_size), requires_grad=True)

    def forward(self, x1, x2):
        assert x1.size(2) == x2.size(2)
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        x1_flat = x1.contiguous().view(-1, x1.size(2))
        x2_flat = x2.contiguous().view(-1, x2.size(2))
        x1_o = self.f(self.proj_1(x1_flat)).view(x1.size(0), x1.size(1), -1)
        x2_o = self.f(self.proj_2(x2_flat)).view(x2.size(0), x2.size(1), -1)
        if self.scale_on:
            scalar = self.scalar.expand_as(x2_o)
            x2_o = scalar * x2_o
        scores = x1_o.bmm(x2_o.transpose(1, 2))
        return scores


class Bilinear(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(Bilinear, self).__init__()
        self.opt = opt
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.transform_on = opt.get('{}_proj_on'.format(self.prefix), False)
        self.dropout = dropout
        if self.transform_on:
            self.proj = nn.Linear(x1_dim, x2_dim)
            if self.layer_norm_on:
                self.proj = weight_norm(self.proj)

    def forward(self, x, y):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        if self.dropout:
            x = self.dropout(x)
            y = self.dropout(y)
        proj = self.proj(y) if self.transform_on else y
        if self.dropout:
            proj = self.dropout(proj)
        scores = x.bmm(proj.unsqueeze(2)).squeeze(2)
        return scores


def init_wrapper(init='xavier_uniform'):
    return eval(init)


class BilinearSum(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(BilinearSum, self).__init__()
        self.x_linear = nn.Linear(x1_dim, 1, bias=False)
        self.y_linear = nn.Linear(x2_dim, 1, bias=False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), False))
        if self.layer_norm_on:
            self.x_linear = weight_norm(self.x_linear)
            self.y_linear = weight_norm(self.y_linear)
        self.init(self.x_linear.weight)
        self.init(self.y_linear.weight)
        self.dropout = dropout

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        score: batch * len1 * len2
        """
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        x1_logits = self.x_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1)
        x2_logits = self.y_linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), 1, -1)
        shape = x1.size(0), x1.size(1), x2.size()
        scores = x1_logits.expand_as(shape) + x2_logits.expand_as(shape)
        return scores


class Trilinear(nn.Module):
    """Function used in BiDAF"""

    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(Trilinear, self).__init__()
        self.prefix = prefix
        self.x_linear = nn.Linear(x1_dim, 1, bias=False)
        self.x_dot_linear = nn.Linear(x1_dim, 1, bias=False)
        self.y_linear = nn.Linear(x2_dim, 1, bias=False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), 'xavier_uniform'))
        if self.layer_norm_on:
            self.x_linear = weight_norm(self.x_linear)
            self.x_dot_linear = weight_norm(self.x_dot_linear)
            self.y_linear = weight_norm(self.y_linear)
        self.init(self.x_linear.weight)
        self.init(self.x_dot_linear.weight)
        self.init(self.y_linear.weight)
        self.dropout = dropout

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        score: batch * len1 * len2
        """
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        x1_logits = self.x_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1)
        x2_logits = self.y_linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), 1, -1)
        x1_dot = self.x_dot_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1).expand_as(x1)
        x1_dot = x1 * x1_dot
        scores = x1_dot.bmm(x2.transpose(1, 2))
        scores += x1_logits.expand_as(scores) + x2_logits.expand_as(scores)
        return scores


class SimilarityWrapper(nn.Module):

    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(SimilarityWrapper, self).__init__()
        self.score_func_str = opt.get('{}_sim_func'.format(prefix), 'dotproductproject').lower()
        self.score_func = None
        if self.score_func_str == 'dotproduct':
            self.score_func = DotProduct(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'dotproductproject':
            self.score_func = DotProductProject(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'bilinear':
            self.score_func = Bilinear(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'bilinearsum':
            self.score_func = BilinearSum(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'trilinear':
            self.score_func = Trilinear(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        else:
            raise NotImplementedError

    def forward(self, x1, x2):
        scores = self.score_func(x1, x2)
        return scores


class AttentionWrapper(nn.Module):

    def __init__(self, x1_dim, x2_dim, x3_dim=None, prefix='attention', opt={}, dropout=None):
        super(AttentionWrapper, self).__init__()
        self.prefix = prefix
        self.att_dropout = opt.get('{}_att_dropout'.format(self.prefix), 0)
        self.score_func = SimilarityWrapper(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        self.drop_diagonal = opt.get('{}_drop_diagonal'.format(self.prefix), False)
        self.output_size = x2_dim if x3_dim is None else x3_dim

    def forward(self, query, key, value, key_padding_mask=None, return_scores=False):
        logits = self.score_func(query, key)
        key_mask = key_padding_mask.unsqueeze(1).expand_as(logits)
        logits.data.masked_fill_(key_mask.data, -float('inf'))
        if self.drop_diagonal:
            assert logits.size(1) == logits.size(2)
            diag_mask = torch.diag(logits.data.new(logits.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(logits)
            logits.data.masked_fill_(diag_mask, -float('inf'))
        prob = F.softmax(logits.view(-1, key.size(1)), 1)
        prob = prob.view(-1, query.size(1), key.size(1))
        if self.att_dropout > 0:
            prob = self.dropout(prob)
        if value is None:
            value = key
        attn = prob.bmm(value)
        if return_scores:
            return attn, prob, logits
        else:
            return attn


class MultiheadAttentionWrapper(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, query_dim, key_dim, value_dim, prefix='attention', opt={}, dropout=None):
        super().__init__()
        self.prefix = prefix
        self.num_heads = opt.get('{}_head'.format(self.prefix), 1)
        self.dropout = DropoutWrapper(opt.get('{}_dropout'.format(self.prefix), 0)) if dropout is None else dropout
        self.qkv_dim = [query_dim, key_dim, value_dim]
        assert query_dim == key_dim, 'query dim must equal with key dim'
        self.hidden_size = opt.get('{}_hidden_size'.format(self.prefix), 64)
        self.proj_on = opt.get('{}_proj_on'.format(prefix), False)
        self.share = opt.get('{}_share'.format(self.prefix), False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.scale_on = opt.get('{}_scale_on'.format(self.prefix), False)
        if self.proj_on:
            self.proj_modules = nn.ModuleList([nn.Linear(dim, self.hidden_size) for dim in self.qkv_dim[0:2]])
            if self.layer_norm_on:
                for proj in self.proj_modules:
                    proj = weight_norm(proj)
            if self.share and self.qkv_dim[0] == self.qkv_dim[1]:
                self.proj_modules[1] = self.proj_modules[0]
            self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
            self.qkv_head_dim = [self.hidden_size // self.num_heads] * 3
            self.qkv_head_dim[2] = value_dim // self.num_heads
            assert self.qkv_head_dim[0] * self.num_heads == self.hidden_size, 'hidden size must be divisible by num_heads'
            assert self.qkv_head_dim[2] * self.num_heads == value_dim, 'value size must be divisible by num_heads'
        else:
            self.qkv_head_dim = [(emb // self.num_heads) for emb in self.qkv_dim]
            assert self.qkv_head_dim[0] * self.num_heads == self.qkv_dim[0], 'query size must be divisible by num_heads'
            assert self.qkv_head_dim[1] * self.num_heads == self.qkv_dim[1], 'key size must be divisible by num_heads'
            assert self.qkv_head_dim[2] * self.num_heads == self.qkv_dim[2], 'value size must be divisible by num_heads'
        if self.scale_on:
            self.scaling = self.qkv_head_dim[0] ** -0.5
        self.drop_diagonal = opt.get('{}_drop_diagonal'.format(self.prefix), False)
        self.output_size = self.qkv_dim[2]

    def forward(self, query, key, value, key_padding_mask=None):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.qkv_dim[0]
        q, k, v = query, key, value
        if self.proj_on:
            if self.dropout:
                q, k = self.dropout(q), self.dropout(k)
            q, k = [self.f(proj(input)) for input, proj in zip([query, key], self.proj_modules)]
        src_len = k.size(0)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.scale_on:
            q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.qkv_head_dim[0]).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.qkv_head_dim[1]).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.qkv_head_dim[2]).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')).type_as(attn_weights)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if self.drop_diagonal:
            assert attn_weights.size(1) == attn_weights.size(2)
            diag_mask = torch.diag(attn_weights.data.new(attn_weights.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(attn_weights)
            attn_weights.data.masked_fill_(diag_mask, -float('inf'))
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = self.dropout(attn_weights)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.qkv_head_dim[2]]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn = attn.transpose(0, 1)
        return attn


class DeepAttentionWrapper(nn.Module):

    def __init__(self, x1_dim, x2_dim, x3_dims, att_cnt, prefix='deep_att', opt=None, dropout=None):
        super(DeepAttentionWrapper, self).__init__()
        self.opt = {} if opt is None else opt
        self.prefix = prefix
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x3_dims = x3_dims
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.attn_list = nn.ModuleList()
        for i in range(0, att_cnt):
            if opt['multihead_on']:
                attention = MultiheadAttentionWrapper(self.x1_dim, self.x2_dim, self.x3_dims[i], prefix, opt, dropout=dropout)
            else:
                attention = AttentionWrapper(self.x1_dim, self.x2_dim, self.x3_dims[i], prefix, opt, self.dropout)
            self.attn_list.append(attention)

    def forward(self, x1, x2, x3, x2_mask):
        rvl = []
        for i in range(0, len(x3)):
            hiddens = self.attn_list[i](x1, x2, x3[i], x2_mask)
            rvl.append(hiddens)
        return torch.cat(rvl, 2)


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=0.0001):
        super(LayerNorm, self).__init__()
        self.alpha = Parameter(torch.ones(1, 1, hidden_size))
        self.beta = Parameter(torch.zeros(1, 1, hidden_size))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            :param x: batch * len * input_size

        Returns:
            normalized x
        """
        mu = torch.mean(x, 2, keepdim=True).expand_as(x)
        sigma = torch.std(x, 2, keepdim=True).expand_as(x)
        return (x - mu) / (sigma + self.eps) * self.alpha.expand_as(x) + self.beta.expand_as(x)


class Criterion(_Loss):

    def __init__(self, alpha=1.0, name='criterion'):
        super().__init__()
        """Alpha is used to weight each loss term
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        return


class CeCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        if weight is not None:
            loss = torch.sum(F.cross_entropy(input, target, reduce=False, reduction='none', ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss


class SeqCeCriterion(CeCriterion):

    def __init__(self, alpha=1.0, name='Seq Cross Entropy Criterion'):
        super().__init__(alpha, name)

    def forward(self, input, target, weight=None, ignore_index=-1):
        target = target.view(-1)
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss


class MseCriterion(Criterion):

    def __init__(self, alpha=1.0, name='MSE Regression Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        if weight:
            loss = torch.mean(F.mse_loss(input.squeeze(), target, reduce=False) * weight.reshape((target.shape[0], 1)))
        else:
            loss = F.mse_loss(input.squeeze(), target)
        loss = loss * self.alpha
        return loss


class KlCriterion(Criterion):

    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction='batchmean')
        loss = loss * self.alpha
        return loss


def stable_kl(logit, target, epsilon=1e-06, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
    ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
    if reduce:
        return (p * (rp - ry) * 2).sum() / bs
    else:
        return (p * (rp - ry) * 2).sum()


class NsKlCriterion(Criterion):

    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = stable_kl(input, target.detach())
        loss = loss * self.alpha
        return loss


class SymKlCriterion(Criterion):

    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
        loss = loss * self.alpha
        return loss


class NsSymKlCriterion(Criterion):

    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        loss = stable_kl(input, target.detach()) + stable_kl(target, input.detach())
        loss = loss * self.alpha
        return loss


class JSCriterion(Criterion):

    def __init__(self, alpha=1.0, name='JS Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        m = F.softmax(target.detach(), dim=-1, dtype=torch.float32) + F.softmax(input.detach(), dim=-1, dtype=torch.float32)
        m = 0.5 * m
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), m, reduction=reduction) + F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), m, reduction=reduction)
        loss = loss * self.alpha
        return loss


class HLCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Hellinger Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits"""
        input = input.float()
        target = target.float()
        si = F.softmax(target.detach(), dim=-1, dtype=torch.float32).sqrt_()
        st = F.softmax(input.detach(), dim=-1, dtype=torch.float32).sqrt_()
        loss = F.mse_loss(si, st)
        loss = loss * self.alpha
        return loss


class RankCeCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, pairwise_size=1):
        input = input.view(-1, pairwise_size)
        target = target.contiguous().view(-1, pairwise_size)[:, 0]
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss


class SpanCeCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Span Cross Entropy Criterion'):
        super().__init__()
        """This is for extractive MRC, e.g., SQuAD, ReCoRD ... etc
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        assert len(input) == 2
        start_input, end_input = input
        if len(target) == 3:
            start_target, end_target, _ = target
        else:
            assert len(target) == 2
            start_target, end_target = target
        if weight:
            b = torch.mean(F.cross_entropy(start_input, start_target, reduce=False, ignore_index=ignore_index) * weight)
            e = torch.mean(F.cross_entropy(end_input, end_target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            b = F.cross_entropy(start_input, start_target, ignore_index=ignore_index)
            e = F.cross_entropy(end_input, end_target, ignore_index=ignore_index)
        loss = 0.5 * (b + e) * self.alpha
        return loss


class SpanYNCeCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Span Cross Entropy Criterion'):
        super().__init__()
        """This is for extractive MRC, e.g., SQuAD, ReCoRD ... etc
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight"""
        assert len(input) == 3
        start_input, end_input, labels_input = input
        start_target, end_target, labels_target = target
        if weight:
            b = torch.mean(F.cross_entropy(start_input, start_target, reduce=False, ignore_index=ignore_index) * weight)
            e = torch.mean(F.cross_entropy(end_input, end_target, reduce=False, ignore_index=ignore_index) * weight)
            e = torch.mean(F.cross_entropy(labels_input, labels_target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            b = F.cross_entropy(start_input, start_target, ignore_index=ignore_index)
            e = F.cross_entropy(end_input, end_target, ignore_index=ignore_index)
            c = F.cross_entropy(labels_input, labels_target, ignore_index=ignore_index)
        loss = 0.5 * (b + e) * self.alpha + c
        return loss


class MlmCriterion(Criterion):

    def __init__(self, alpha=1.0, name='BERT pre-train Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """TODO: support sample weight, xiaodl"""
        mlm_y, y = target
        mlm_p, nsp_p = input
        mlm_p = mlm_p.view(-1, mlm_p.size(-1))
        mlm_y = mlm_y.view(-1)
        mlm_loss = F.cross_entropy(mlm_p, mlm_y, ignore_index=ignore_index)
        nsp_loss = F.cross_entropy(nsp_p, y)
        loss = mlm_loss + nsp_loss
        loss = loss * self.alpha
        return loss


class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3
    SAN = 4
    XLM = 5
    DEBERTA = 6
    ELECTRA = 7
    T5 = 8
    T5G = 9


class TaskDef(dict):

    def __init__(self, label_vocab, n_class, data_type, task_type, metric_meta, split_names, enable_san, dropout_p, loss, kd_loss, adv_loss, actf):
        """
        :param label_vocab: map string label to numbers.
            only valid for Classification task or ranking task.
            For ranking task, better label should have large number
        """
        super().__init__(**{k: repr(v) for k, v in locals().items()})
        self.label_vocab = label_vocab
        self.n_class = n_class
        self.data_type = data_type
        self.task_type = task_type
        self.metric_meta = metric_meta
        self.split_names = split_names
        self.enable_san = enable_san
        self.dropout_p = dropout_p
        self.loss = loss
        self.kd_loss = kd_loss
        self.adv_loss = adv_loss
        self.actf = actf

    @classmethod
    def from_dict(cls, dict_rep):
        return cls(**dict_rep)


class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    SpanClassification = 4
    SpanClassificationYN = 5
    SeqenceLabeling = 6
    MaskLM = 7
    SpanSeqenceLabeling = 8
    SeqenceGeneration = 9
    SeqenceGenerationMRC = 10
    EncSeqenceGeneration = 11
    ClozeChoice = 12


class SANBertNetwork(nn.Module):

    def __init__(self, opt, bert_config=None, initial_from_local=False):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = nn.ModuleList()
        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError('encoder_type is out of pre-defined types')
        self.encoder_type = opt['encoder_type']
        self.preloaded_config = None
        literal_encoder_type = EncoderModelType(self.encoder_type).name.lower()
        config_class, model_class, _ = MODEL_CLASSES[literal_encoder_type]
        if not initial_from_local:
            self.bert = model_class.from_pretrained(opt['init_checkpoint'], cache_dir=opt['transformer_cache'])
        else:
            self.preloaded_config = config_class.from_dict(opt)
            self.preloaded_config.output_hidden_states = True
            self.bert = model_class(self.preloaded_config)
        hidden_size = self.bert.config.hidden_size
        if opt.get('dump_feature', False):
            self.config = opt
            return
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False
        task_def_list = opt['task_def_list']
        self.task_def_list = task_def_list
        self.scoring_list = nn.ModuleList()
        for task_id in range(len(task_def_list)):
            task_def: TaskDef = task_def_list[task_id]
            lab = task_def.n_class
            task_type = task_def.task_type
            task_obj = tasks.get_task_obj(task_def)
            if task_obj is not None:
                out_proj = task_obj.train_build_task_layer(hidden_size, task_def, opt)
            elif task_type == TaskType.ClozeChoice:
                self.pooler = Pooler(hidden_size, dropout_p=opt['dropout_p'], actf=opt['pooler_actf'])
                out_proj = nn.Linear(hidden_size, lab)
            else:
                raise NotImplementedError()
            self.scoring_list.append(out_proj)
        self.config = opt

    def embed_encode(self, input_ids, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.bert.embeddings(input_ids, token_type_ids)
        return embedding_output

    def encode(self, input_ids, token_type_ids, attention_mask, inputs_embeds=None, y_input_ids=None):
        if self.encoder_type == EncoderModelType.T5:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
            last_hidden_state = outputs.last_hidden_state
            all_hidden_states = outputs.hidden_states
        elif self.encoder_type == EncoderModelType.T5G:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=y_input_ids)
            last_hidden_state = outputs.logits
            all_hidden_states = outputs.encoder_last_hidden_state
        else:
            outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
            last_hidden_state = outputs.last_hidden_state
            all_hidden_states = outputs.hidden_states
        return last_hidden_state, all_hidden_states

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0, y_input_ids=None, fwd_type=0, embed=None):
        if fwd_type == 3:
            generated = self.bert.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=self.config['max_answer_len'], num_beams=self.config['num_beams'], repetition_penalty=self.config['repetition_penalty'], length_penalty=self.config['length_penalty'], early_stopping=True)
            return generated
        elif fwd_type == 2:
            assert embed is not None
            last_hidden_state, all_hidden_states = self.encode(None, token_type_ids, attention_mask, embed, y_input_ids)
        elif fwd_type == 1:
            return self.embed_encode(input_ids, token_type_ids, attention_mask)
        else:
            last_hidden_state, all_hidden_states = self.encode(input_ids, token_type_ids, attention_mask, y_input_ids=y_input_ids)
        task_obj = tasks.get_task_obj(self.task_def_list[task_id])
        task_type = task_obj._task_def.task_type
        if task_obj is not None:
            logits = task_obj.train_forward(last_hidden_state, premise_mask, hyp_mask, self.scoring_list[task_id], enable_san=task_obj._task_def.enable_san)
            return logits
        elif task_type == TaskType.ClozeChoice:
            pooled_output = self.pooler(last_hidden_state)
            pooled_output = self.dropout_list[task_id](pooled_output)
            logits = self.scoring_list[task_id](pooled_output)
            return logits
        else:
            raise NotImplementedError()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CeCriterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Classifier,
     lambda: ([], {'x_size': 4, 'y_size': 4, 'opt': _mock_config(get=_mock_layer)}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (Criterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DotProduct,
     lambda: ([], {'x1_dim': 4, 'x2_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (DotProductProject,
     lambda: ([], {'x1_dim': 4, 'x2_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropoutWrapper,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HLCriterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (JSCriterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (KlCriterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MseCriterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiheadAttentionWrapper,
     lambda: ([], {'query_dim': 4, 'key_dim': 4, 'value_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (NsKlCriterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NsSymKlCriterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Pooler,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SimilarityWrapper,
     lambda: ([], {'x1_dim': 4, 'x2_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SymKlCriterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Trilinear,
     lambda: ([], {'x1_dim': 4, 'x2_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_namisan_mt_dnn(_paritybench_base):
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

