import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
main = _module
model = _module
optimizations = _module
tcanet = _module
tcn_block = _module
tcn_block_time_test = _module
utils = _module
dataset = _module
utils = _module

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


import numpy as np


import pandas as pd


from collections import Counter


from torch.utils import data


from torch.autograd import Variable


import logging


import time


import torch.nn as nn


import torch.optim as optim


import torch.multiprocessing as mp


import torch.nn.functional as F


from torch import nn


from torch.nn.utils import weight_norm


import math


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


import matplotlib


from matplotlib import pyplot as plt


class VariationalDropout(nn.Module):

    def __init__(self, dropout=0.3):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        mask_matrix = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        with torch.no_grad():
            mask = mask_matrix / (1 - self.dropout)
            mask = mask.expand_as(mask_matrix)
        return mask * x


class WeightDropout(nn.Module):

    def __init__(self, module, weights, dropout=0):
        super(WeightDropout, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        for path in self.weights:
            module = self.module
            name_w = path[-1]
            for i in range(len(path) - 1):
                if isinstance(path[i], int):
                    module = module[path[i]]
                elif isinstance(path[i], str):
                    module = getattr(module, path[i])
            try:
                w = getattr(module, name_w)
            except:
                continue
            del module._parameters[name_w]
            module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for path in self.weights:
            module = self.module
            name_w = path[-1]
            for i in range(len(path) - 1):
                if isinstance(path[i], int):
                    module = module[path[i]]
                elif isinstance(path[i], str):
                    module = getattr(module, path[i])
            raw_w = getattr(module, name_w + '_raw')
            w = F.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(module, name_w, w)

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module.forward(*args, **kwargs)


class VariationalHidDropout(nn.Module):

    def __init__(self, vhdropout):
        super(VariationalHidDropout, self).__init__()
        self.vhdrop = vhdropout

    def reset_mask(self, x):
        m = x.data.new(x.size(0), x.size(1), 1)._bernoulli(self.vhdrop)
        with torch.no_grad():
            mask = m / (1 - self.vhdrop)
            self.mask = mask
        return self.mask

    def forward(self, x):
        if not self.training or self.vhdrop == 0:
            return x
        assert self.mask is not None, 'You need to reset mask before using VariationalHidDropout'
        mask = self.mask.expand_as(x)
        return mask * x


class AttentionBlock(nn.Module):

    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        mask = np.array([[(1 if i > j else 0) for i in range(input.size(2))] for j in range(input.size(2))])
        if input.is_cuda:
            mask = torch.ByteTensor(mask)
        else:
            mask = torch.ByteTensor(mask)
        input = input.permute(0, 2, 1)
        kvq_start = time.time()
        keys = self.linear_keys(input)
        query = self.linear_query(input)
        values = self.linear_values(input)
        kvq_cost = time.time() - kvq_start
        qk_start = time.time()
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))
        qk_cost = time.time() - qk_start
        mask_fill_start = time.time()
        temp.data.masked_fill_(mask, -float('inf'))
        mask_fill_cost = time.time() - mask_fill_start
        softmax_start = time.time()
        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1)
        softmax_cost = time.time() - softmax_start
        weight_value_start = time.time()
        value_attentioned = torch.bmm(weight_temp, values).permute(0, 2, 1)
        weight_value_cost = time.time() - weight_value_start
        None
        None
        None
        None
        None
        None
        return value_attentioned, weight_temp


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, key_size, num_sub_blocks, temp_attn, nheads, en_res, stride, dilation, padding, vhdrop_layer, visual, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.nheads = nheads
        self.visual = visual
        self.en_res = en_res
        self.temp_attn = temp_attn
        if self.temp_attn == True:
            if self.nheads > 1:
                self.attentions = [AttentionBlock(n_inputs, key_size, n_inputs) for _ in range(self.nheads)]
                for i, attention in enumerate(self.attentions):
                    self.add_module('attention_{}'.format(i), attention)
                self.linear_cat = nn.Linear(n_inputs * self.nheads, n_inputs)
            else:
                self.attention = AttentionBlock(n_inputs, key_size, n_inputs)
        self.net = self._make_layers(num_sub_blocks, n_inputs, n_outputs, kernel_size, stride, dilation, padding, vhdrop_layer, dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def _make_layers(self, num_sub_blocks, n_inputs, n_outputs, kernel_size, stride, dilation, padding, vhdrop_layer, dropout=0.2):
        layers_list = []
        if vhdrop_layer is not None:
            layers_list.append(vhdrop_layer)
        layers_list.append(weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
        layers_list.append(Chomp1d(padding))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(dropout))
        for _ in range(num_sub_blocks - 1):
            layers_list.append(weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
            layers_list.append(Chomp1d(padding))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout))
        return nn.Sequential(*layers_list)

    def init_weights(self):
        layer_idx_list = []
        for name, _ in self.net.named_parameters():
            inlayer_param_list = name.split('.')
            layer_idx_list.append(int(inlayer_param_list[0]))
        layer_idxes = list(set(layer_idx_list))
        for idx in layer_idxes:
            getattr(self.net[idx], 'weight').data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        if self.temp_attn == True:
            en_res_x = None
            if self.nheads > 1:
                x_out = torch.cat([att(x) for att in self.attentions], dim=1)
                out = self.net(self.linear_cat(x_out.transpose(1, 2)).transpose(1, 2))
            else:
                attn_start = time.time()
                out_attn, attn_weight = self.attention(x)
                attn_cost = time.time() - attn_start
                tcn_and_res_start = time.time()
                out = self.net(out_attn)
                weight_x = F.softmax(attn_weight.sum(dim=2), dim=1)
                en_res_x = weight_x.unsqueeze(2).repeat(1, 1, x.size(1)).transpose(1, 2) * x
                en_res_x = en_res_x if self.downsample is None else self.downsample(en_res_x)
                tcn_and_res_cost = time.time() - tcn_and_res_start
            res = x if self.downsample is None else self.downsample(x)
            to_cpu_start = time.time()
            if self.visual:
                attn_weight_cpu = attn_weight.detach().cpu().numpy()
            else:
                attn_weight_cpu = [0] * 10
            del attn_weight
            to_cpu_cost = time.time() - to_cpu_start
            if self.en_res:
                return self.relu(out + res + en_res_x), attn_weight_cpu
            else:
                return self.relu(out + res), attn_weight_cpu
        else:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)


class TemporalConvNet(nn.Module):

    def __init__(self, input_output_size, emb_size, num_channels, num_sub_blocks, temp_attn, nheads, en_res, key_size, kernel_size, visual, vhdropout=0.0, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.vhdrop_layer = None
        self.temp_attn = temp_attn
        if vhdropout != 0.0:
            None
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = emb_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, key_size, num_sub_blocks, temp_attn, nheads, en_res, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, vhdrop_layer=self.vhdrop_layer, visual=visual, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        attn_weight_list = []
        if self.temp_attn:
            out = x
            for i in range(len(self.network)):
                out, attn_weight = self.network[i](out)
                attn_weight_list.append([attn_weight[0], attn_weight[-1]])
            return out, attn_weight_list
        else:
            return self.network(x)


class TCANet(nn.Module):

    def __init__(self, emb_size, input_output_size, num_channels, seq_len, num_sub_blocks, temp_attn, nheads, en_res, conv, key_size, kernel_size=2, dropout=0.3, wdrop=0.0, emb_dropout=0.1, tied_weights=False, dataset_name=None, visual=True):
        super(TCANet, self).__init__()
        self.temp_attn = temp_attn
        self.dataset_name = dataset_name
        self.num_levels = len(num_channels)
        self.word_encoder = nn.Embedding(input_output_size, emb_size)
        if dataset_name == 'mnist':
            self.word_encoder = nn.Embedding(256, emb_size)
        self.tcanet = TemporalConvNet(input_output_size, emb_size, num_channels, num_sub_blocks, temp_attn, nheads, en_res, conv, key_size, kernel_size, visual=visual, dropout=dropout)
        self.drop = nn.Dropout(emb_dropout)
        self.decoder = nn.Linear(num_channels[-1], input_output_size)
        if tied_weights:
            if self.dataset_name != 'mnist':
                self.decoder.weight = self.word_encoder.weight
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.word_encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def get_conv_names(self, num_channels):
        conv_names_list = []
        for level_i in range(len(num_channels)):
            conv_names_list.append(['network', level_i, 'net', 0, 'weight_v'])
            conv_names_list.append(['network', level_i, 'net', 4, 'weight_v'])
        return conv_names_list

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        if self.dataset_name == 'mnist':
            if self.temp_attn:
                y, attn_weight_list = self.tcanet(input)
                o = self.decoder(y[:, :, -1])
                return F.log_softmax(o, dim=1).contiguous()
            else:
                y = self.tcanet(input)
                o = self.decoder(y[:, :, -1])
                return F.log_softmax(o, dim=1).contiguous()
        emb = self.drop(self.word_encoder(input))
        if self.temp_attn:
            y, attn_weight_list = self.tcanet(emb.transpose(1, 2))
            y = self.decoder(y.transpose(1, 2))
            return y.contiguous(), [attn_weight_list[0], attn_weight_list[self.num_levels // 2], attn_weight_list[-1]]
        else:
            y = self.tcanet(emb.transpose(1, 2))
            y = self.decoder(y.transpose(1, 2))
            return y.contiguous()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionBlock,
     lambda: ([], {'in_channels': 4, 'key_size': 4, 'value_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Chomp1d,
     lambda: ([], {'chomp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VariationalDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VariationalHidDropout,
     lambda: ([], {'vhdropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_haohy_TCAN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

