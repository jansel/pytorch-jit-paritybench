import sys
_module = sys.modules[__name__]
del sys
main = _module
generate_multi_language_configs = _module
ch_ppocr_v2_det_converter = _module
ch_ppocr_v2_rec_converter = _module
ch_ppocr_v3_det_converter = _module
ch_ppocr_v3_rec_converter = _module
ch_ppocr_v3_rec_converter_nodistill = _module
det_converter = _module
det_fcenet_converter = _module
e2e_converter = _module
layoutxlm_re_converter = _module
layoutxlm_ser_converter = _module
multilingual_ppocr_v3_rec_converter = _module
ppstructure_table_det_converter = _module
ppstructure_table_rec_converter = _module
ppstructure_table_structure_converter = _module
rec_converter = _module
rec_nrtr_mtb_converter = _module
rec_sar_converter = _module
rec_svtr_converter = _module
rec_vitstr_converter = _module
srn_converter = _module
attention_grucell = _module
attention_head = _module
common = _module
conv = _module
diff = _module
fc = _module
gelu = _module
gru_cell = _module
hard_swish = _module
hs = _module
layernorm = _module
lstm = _module
pp_ocr = _module
pp_rec_resnet_fpn = _module
pp_rec_srn_head = _module
pp_self_attention = _module
pp_table_att_head = _module
pp_table_fpn = _module
pp_table_mobilenet_v3 = _module
pt_rec_resnet_fpn = _module
pt_rec_srn_head = _module
pt_self_attention = _module
pt_table_att_head = _module
pt_table_fpn = _module
pt_table_mobilenet_v3 = _module
rec_resnet_fpn = _module
rec_srn = _module
rec_srn_head = _module
table_att_head = _module
table_det = _module
table_mobile = _module
table_mobilenet_v3 = _module
onnx_optimizer = _module
ptstructure = _module
ptppyolov2 = _module
ppyolo_utils = _module
ppyolov2 = _module
ppyolov2_base = _module
ppyolov2_darknet = _module
ppyolov2_layout = _module
ppyolov2_pt = _module
ppyolov2_resnet = _module
ppyolov2_yolo_fpn = _module
ppyolov2_yolo_head = _module
pt_utils = _module
utils = _module
predict_system = _module
table = _module
matcher = _module
predict_structure = _module
predict_table = _module
tablepyxl = _module
style = _module
utility = _module
infer_ser_e2e = _module
infer_ser_re_e2e = _module
data = _module
vocab = _module
transformers = _module
bert = _module
tokenizer = _module
layoutlm = _module
modeling = _module
layoutxlm = _module
modeling = _module
visual_backbone = _module
model_utils = _module
tokenizer_utils = _module
utils = _module
vqa_utils = _module
pytorchocr = _module
base_ocr_v20 = _module
imaug = _module
gen_table_mask = _module
operators = _module
architectures = _module
base_model = _module
backbones = _module
det_mobilenet_v3 = _module
det_resnet = _module
det_resnet_vd = _module
det_resnet_vd_sast = _module
e2e_resnet_vd_pg = _module
rec_mobilenet_v3 = _module
rec_mv1_enhance = _module
rec_nrtr_mtb = _module
rec_resnet_31 = _module
rec_resnet_fpn = _module
rec_resnet_vd = _module
rec_svtrnet = _module
rec_vitstr = _module
table_mobilenet_v3 = _module
table_resnet_vd = _module
common = _module
heads = _module
cls_head = _module
det_db_head = _module
det_east_head = _module
det_fce_head = _module
det_pse_head = _module
det_sast_head = _module
e2e_pg_head = _module
multiheadAttention = _module
rec_att_head = _module
rec_ctc_head = _module
rec_nrtr_head = _module
rec_sar_head = _module
rec_srn_head = _module
self_attention = _module
table_att_head = _module
necks = _module
db_fpn = _module
east_fpn = _module
fce_fpn = _module
fpn = _module
pg_fpn = _module
rnn = _module
sast_fpn = _module
table_fpn = _module
transforms = _module
stn = _module
tps = _module
tps_spatial_transformer = _module
postprocess = _module
cls_postprocess = _module
db_postprocess = _module
east_postprocess = _module
fce_postprocess = _module
locality_aware_nms = _module
pg_postprocess = _module
pse_postprocess = _module
pse = _module
setup = _module
pse_postprocess = _module
rec_postprocess = _module
sast_postprocess = _module
extract_batchsize = _module
extract_textpoint_fast = _module
extract_textpoint_slow = _module
pgnet_pp_utils = _module
visual = _module
logging = _module
poly_nms = _module
predict_cls = _module
predict_det = _module
predict_e2e = _module
predict_rec = _module
pytorchocr_utility = _module

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


from collections import OrderedDict


import numpy as np


import torch


import copy


import torch.nn as nn


import torch.nn.functional as F


import math


import torchvision


from numbers import Integral


import time


from copy import deepcopy


from torch.nn import Module as Layer


from abc import abstractmethod


from collections import namedtuple


from torch.nn import Module


import logging


import inspect


import functools


import random


from torch import nn


from functools import partial


from torch.nn import Linear


from torch.nn.init import xavier_uniform_


from torch.nn import ModuleList as LayerList


from torch.nn import Dropout


from torch.nn import LayerNorm


from torch.nn import Conv2d


from torch.nn.init import xavier_normal_


from torch.nn import functional as F


import itertools


from numpy.fft import ifft


import string


import torch.distributed as dist


class PTAttentionGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(PTAttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size=input_size + num_embeddings, hidden_size=hidden_size, bias=True)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)
        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.matmul(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots.float()], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return (cur_hidden, cur_hidden), alpha


class PTAttentionHead(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(PTAttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels
        self.attention_cell = PTAttentionGRUCell(in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char.type(torch.int64), onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.size()[0]
        num_steps = batch_max_length
        hidden = torch.zeros((batch_size, self.hidden_size))
        output_hiddens = []
        if targets is not None:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs, char_onehots)
                output_hiddens.append(torch.unsqueeze(outputs, dim=1))
            output = torch.cat(output_hiddens, dim=1)
            probs = self.generator(output)
        else:
            targets = torch.zeros([batch_size], dtype=torch.int32)
            probs = None
            char_onehots = None
            outputs = None
            alpha = None
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(outputs)
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.cat([probs, torch.unsqueeze(probs_step, dim=1)], dim=1)
                next_input = probs_step.argmax(dim=1)
                targets = next_input
        return probs


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(1.2 * x + 3.0, inplace=self.inplace) / 6.0


class GELU(nn.Module):

    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):

    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            raise NotImplementedError
        elif act_type == 'hard_sigmoid':
            self.act = Hsigmoid(inplace)
        elif act_type == 'hard_swish':
            self.act = Hswish(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        elif act_type == 'gelu':
            self.act = GELU(inplace=inplace)
        elif act_type == 'swish':
            self.act = Swish(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)


class ConvBNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=None, name=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        bn_name = 'bn_' + name
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act
        if act is not None:
            self._act = Activation(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self._act(x)
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shortcut=True, if_first=False, name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, act='relu', name=name + '_branch2a')
        self.conv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, act=None, name=name + '_branch2b')
        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, is_vd_mode=False if if_first else True, name=name + '_branch1')
        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = torch.add(short, conv1)
        y = F.relu(y)
        return y


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shortcut=True, if_first=False, name=None):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, act='relu', name=name + '_branch2a')
        self.conv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, act='relu', name=name + '_branch2b')
        self.conv2 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, act=None, name=name + '_branch2c')
        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels, out_channels=out_channels * 4, kernel_size=1, stride=1, is_vd_mode=False if if_first else True, name=name + '_branch1')
        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = torch.add(short, conv2)
        y = F.relu(y)
        return y


class ResNetFPN(nn.Module):

    def __init__(self, in_channels=1, layers=50, **kwargs):
        super(ResNetFPN, self).__init__()
        supported_layers = {(18): {'depth': [2, 2, 2, 2], 'block_class': BasicBlock}, (34): {'depth': [3, 4, 6, 3], 'block_class': BasicBlock}, (50): {'depth': [3, 4, 6, 3], 'block_class': BottleneckBlock}, (101): {'depth': [3, 4, 23, 3], 'block_class': BottleneckBlock}, (152): {'depth': [3, 8, 36, 3], 'block_class': BottleneckBlock}}
        stride_list = [(2, 2), (2, 2), (1, 1), (1, 1)]
        num_filters = [64, 128, 256, 512]
        self.depth = supported_layers[layers]['depth']
        self.conv = ConvBNLayer(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, act='relu', name='conv1')
        self.block_list = nn.ModuleList()
        in_ch = 64
        if layers >= 50:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = 'res' + str(block + 2) + 'a'
                        else:
                            conv_name = 'res' + str(block + 2) + 'b' + str(i)
                    else:
                        conv_name = 'res' + str(block + 2) + chr(97 + i)
                    bottlenectBlock = BottleneckBlock(in_channels=in_ch, out_channels=num_filters[block], stride=stride_list[block] if i == 0 else 1, name=conv_name)
                    in_ch = num_filters[block] * 4
                    self.block_list.add_module('bottleneckBlock_{}_{}'.format(block, i), bottlenectBlock)
        else:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    conv_name = 'res' + str(block + 2) + chr(97 + i)
                    if i == 0 and block != 0:
                        stride = 2, 1
                    else:
                        stride = 1, 1
                    basicBlock = BasicBlock(in_channels=in_ch, out_channels=num_filters[block], stride=stride_list[block] if i == 0 else 1, is_first=block == i == 0, name=conv_name)
                    in_ch = basicBlock.out_channels
                    self.block_list.add_module(conv_name, basicBlock)
        out_ch_list = [in_ch // 4, in_ch // 2, in_ch]
        self.base_block = nn.ModuleList()
        self.conv_trans = []
        self.bn_block = []
        for i in [-2, -3]:
            in_channels = out_ch_list[i + 1] + out_ch_list[i]
            bb_0 = nn.Conv2d(in_channels=in_channels, out_channels=out_ch_list[i], kernel_size=1, bias=True)
            self.base_block.add_module('F_{}_base_block_0'.format(i), bb_0)
            bb_1 = nn.Conv2d(in_channels=out_ch_list[i], out_channels=out_ch_list[i], kernel_size=3, padding=1, bias=True)
            self.base_block.add_module('F_{}_base_block_1'.format(i), bb_1)
            bb_2 = nn.Sequential(nn.BatchNorm2d(out_ch_list[i]), Activation('relu'))
            self.base_block.add_module('F_{}_base_block_2'.format(i), bb_2)
        bb_3 = nn.Conv2d(in_channels=out_ch_list[i], out_channels=512, kernel_size=1, bias=True)
        self.base_block.add_module('F_{}_base_block_3'.format(i), bb_3)
        self.out_channels = 512

    def __call__(self, x):
        x = self.conv(x)
        fpn_list = []
        F = []
        for i in range(len(self.depth)):
            fpn_list.append(np.sum(self.depth[:i + 1]))
        for i, block in enumerate(self.block_list):
            x = block(x)
            for number in fpn_list:
                if i + 1 == number:
                    F.append(x)
        base = F[-1]
        j = 0
        for i, block in enumerate(self.base_block):
            if i % 3 == 0 and i < 6:
                j = j + 1
                b, c, w, h = F[-j - 1].shape
                if [w, h] == list(base.shape[2:]):
                    base = base
                else:
                    base = self.conv_trans[j - 1](base)
                    base = self.bn_block[j - 1](base)
                base = torch.cat([base, F[-j - 1]], dim=1)
            base = block(base)
        return base


class ShortCut(nn.Module):

    def __init__(self, in_channels, out_channels, stride, name, is_first=False):
        super(ShortCut, self).__init__()
        self.use_conv = True
        if in_channels != out_channels or stride != 1 or is_first == True:
            if stride == (1, 1):
                self.conv = ConvBNLayer(in_channels, out_channels, 1, 1, name=name)
            else:
                self.conv = ConvBNLayer(in_channels, out_channels, 1, stride, name=name)
        else:
            self.use_conv = False

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        return x


class Lambda(nn.Module):
    """An easy way to create a pytorch layer for a simple `func`."""

    def __init__(self, func):
        """create a layer that simply calls `func` with `x`"""
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class FFN(nn.Module):
    """
    Feed-Forward Network
    """

    def __init__(self, d_inner_hid, d_model, dropout_rate):
        super(FFN, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = torch.nn.Linear(in_features=d_model, out_features=d_inner_hid)
        self.fc2 = torch.nn.Linear(in_features=d_inner_hid, out_features=d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        if self.dropout_rate:
            hidden = F.dropout(hidden, p=self.dropout_rate)
        out = self.fc2(hidden)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """

    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = torch.nn.Linear(in_features=d_model, out_features=d_key * n_head, bias=False)
        self.k_fc = torch.nn.Linear(in_features=d_model, out_features=d_key * n_head, bias=False)
        self.v_fc = torch.nn.Linear(in_features=d_model, out_features=d_value * n_head, bias=False)
        self.proj_fc = torch.nn.Linear(in_features=d_value * n_head, out_features=d_model, bias=False)

    def _prepare_qkv(self, queries, keys, values, cache=None):
        if keys is None:
            keys, values = queries, queries
            static_kv = False
        else:
            static_kv = True
        q = self.q_fc(queries)
        q = torch.reshape(q, shape=[q.size(0), q.size(1), self.n_head, self.d_key])
        q = q.permute(0, 2, 1, 3)
        if cache is not None and static_kv and 'static_k' in cache:
            k = cache['static_k']
            v = cache['static_v']
        else:
            k = self.k_fc(keys)
            v = self.v_fc(values)
            k = torch.reshape(k, shape=[k.size(0), k.size(1), self.n_head, self.d_key])
            k = k.permute(0, 2, 1, 3)
            v = torch.reshape(v, shape=[v.size(0), v.size(1), self.n_head, self.d_value])
            v = v.permute(0, 2, 1, 3)
        if cache is not None:
            if static_kv and not 'static_k' in cache:
                cache['static_k'], cache['static_v'] = k, v
            elif not static_kv:
                cache_k, cache_v = cache['k'], cache['v']
                k = torch.cat([cache_k, k], dim=2)
                v = torch.cat([cache_v, v], dim=2)
                cache['k'], cache['v'] = k, v
        return q, k, v

    def forward(self, queries, keys, values, attn_bias, cache=None):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q, k, v = self._prepare_qkv(queries, keys, values, cache)
        product = torch.matmul(q, k.transpose(2, 3))
        product = product * self.d_model ** -0.5
        if attn_bias is not None:
            product += attn_bias
        weights = F.softmax(product, dim=-1)
        if self.dropout_rate:
            weights = F.dropout(weights, p=self.dropout_rate)
        out = torch.matmul(weights, v)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, shape=[out.size(0), out.size(1), out.shape[2] * out.shape[3]])
        out = self.proj_fc(out)
        return out


class LambdaXY(nn.Module):
    """An easy way to create a pytorch layer for a simple `func`."""

    def __init__(self, func):
        """create a layer that simply calls `func` with `x`"""
        super().__init__()
        self.func = func

    def forward(self, x, y):
        return self.func(x, y)


class PrePostProcessLayer(nn.Module):
    """
    PrePostProcessLayer
    """

    def __init__(self, process_cmd, d_model, dropout_rate):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.functors = nn.ModuleList()
        cur_a_len = 0
        cur_n_len = 0
        cur_d_len = 0
        for cmd in self.process_cmd:
            if cmd == 'a':
                self.functors.add_module('add_res_connect_{}'.format(cur_a_len), LambdaXY(lambda x, y: x + y if y is not None else x))
                cur_a_len += 1
            elif cmd == 'n':
                layerNorm = torch.nn.LayerNorm(normalized_shape=d_model, elementwise_affine=True, eps=1e-05)
                self.functors.add_module('layer_norm_%d' % cur_n_len, layerNorm)
                cur_n_len += 1
            elif cmd == 'd':
                self.functors.add_module('add_drop_{}'.format(cur_d_len), Lambda(lambda x: F.dropout(x, p=dropout_rate) if dropout_rate else x))
                cur_d_len += 1

    def forward(self, x, residual=None):
        for i, (cmd, functor) in enumerate(zip(self.process_cmd, self.functors)):
            if cmd == 'a':
                x = functor(x, residual)
            else:
                x = functor(x)
        return x


class EncoderLayer(nn.Module):
    """
    EncoderLayer
    """

    def __init__(self, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd='n', postprocess_cmd='da'):
        super(EncoderLayer, self).__init__()
        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model, prepostprocess_dropout)
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head, attention_dropout)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model, prepostprocess_dropout)
        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model, prepostprocess_dropout)
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model, prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        attn_output = self.self_attn(self.preprocesser1(enc_input), None, None, attn_bias)
        attn_output = self.postprocesser1(attn_output, enc_input)
        ffn_output = self.ffn(self.preprocesser2(attn_output))
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output


class Encoder(nn.Module):
    """
    encoder
    """

    def __init__(self, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd='n', postprocess_cmd='da'):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        for i in range(n_layer):
            encoderLayer = EncoderLayer(n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd)
            self.encoder_layers.add_module('layer_%d' % i, encoderLayer)
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model, prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_input, attn_bias)
            enc_input = enc_output
        enc_output = self.processer(enc_output)
        return enc_output


class PrepareEncoder(nn.Module):

    def __init__(self, src_vocab_size, src_emb_dim, src_max_len, dropout_rate=0, bos_idx=0, word_emb_param_name=None, pos_enc_param_name=None):
        super(PrepareEncoder, self).__init__()
        self.src_emb_dim = src_emb_dim
        self.src_max_len = src_max_len
        self.emb = torch.nn.Embedding(num_embeddings=self.src_max_len, embedding_dim=self.src_emb_dim, sparse=True)
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos):
        src_word_emb = src_word.type(torch.float32)
        src_word_emb = self.src_emb_dim ** 0.5 * src_word_emb
        src_pos = torch.squeeze(src_pos, dim=-1)
        src_pos_enc = self.emb(src_pos.type(torch.int64))
        src_pos_enc.stop_gradient = True
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = F.dropout(enc_input, p=self.dropout_rate)
        else:
            out = enc_input
        return out


class WrapEncoderForFeature(nn.Module):

    def __init__(self, src_vocab_size, max_length, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, weight_sharing, bos_idx=0):
        super(WrapEncoderForFeature, self).__init__()
        self.prepare_encoder = PrepareEncoder(src_vocab_size, d_model, max_length, prepostprocess_dropout, bos_idx=bos_idx, word_emb_param_name='src_word_emb_table')
        self.encoder = Encoder(n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd)

    def forward(self, enc_inputs):
        conv_features, src_pos, src_slf_attn_bias = enc_inputs
        enc_input = self.prepare_encoder(conv_features, src_pos)
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class PVAM(nn.Module):

    def __init__(self, in_channels, char_num, max_text_length, num_heads, num_encoder_tus, hidden_dims):
        super(PVAM, self).__init__()
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.hidden_dims = hidden_dims
        t = 256
        c = 512
        self.wrap_encoder_for_feature = WrapEncoderForFeature(src_vocab_size=1, max_length=t, n_layer=self.num_encoder_TUs, n_head=self.num_heads, d_key=int(self.hidden_dims / self.num_heads), d_value=int(self.hidden_dims / self.num_heads), d_model=self.hidden_dims, d_inner_hid=self.hidden_dims, prepostprocess_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, preprocess_cmd='n', postprocess_cmd='da', weight_sharing=True)
        self.flatten0 = Lambda(lambda x: torch.flatten(x, start_dim=0, end_dim=1))
        self.fc0 = torch.nn.Linear(in_features=in_channels, out_features=in_channels)
        self.emb = torch.nn.Embedding(num_embeddings=self.max_length, embedding_dim=in_channels)
        self.flatten1 = Lambda(lambda x: torch.flatten(x, start_dim=0, end_dim=2))
        self.fc1 = torch.nn.Linear(in_features=in_channels, out_features=1, bias=False)

    def forward(self, inputs, encoder_word_pos, gsrm_word_pos):
        b, c, h, w = inputs.shape
        conv_features = torch.reshape(inputs, shape=[-1, c, h * w])
        conv_features = conv_features.permute(0, 2, 1)
        b, t, c = conv_features.shape
        enc_inputs = [conv_features, encoder_word_pos, None]
        word_features = self.wrap_encoder_for_feature(enc_inputs)
        b, t, c = word_features.shape
        word_features = self.fc0(word_features)
        word_features_ = torch.reshape(word_features, [-1, 1, t, c])
        word_features_ = word_features_.repeat([1, self.max_length, 1, 1])
        word_pos_feature = self.emb(gsrm_word_pos)
        word_pos_feature_ = torch.reshape(word_pos_feature, [-1, self.max_length, 1, c])
        word_pos_feature_ = word_pos_feature_.repeat([1, 1, t, 1])
        y = word_pos_feature_ + word_features_
        y = torch.tanh(y)
        attention_weight = self.fc1(y)
        attention_weight = torch.reshape(attention_weight, shape=[-1, self.max_length, t])
        attention_weight = F.softmax(attention_weight, dim=-1)
        pvam_features = torch.matmul(attention_weight, word_features)
        return pvam_features


class PrepareDecoder(nn.Module):

    def __init__(self, src_vocab_size, src_emb_dim, src_max_len, dropout_rate=0, bos_idx=0, word_emb_param_name=None, pos_enc_param_name=None):
        super(PrepareDecoder, self).__init__()
        self.src_emb_dim = src_emb_dim
        """
        self.emb0 = Embedding(num_embeddings=src_vocab_size,
                              embedding_dim=src_emb_dim)
        """
        self.emb0 = torch.nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=self.src_emb_dim, padding_idx=bos_idx)
        self.emb1 = torch.nn.Embedding(num_embeddings=src_max_len, embedding_dim=self.src_emb_dim)
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos):
        src_word = torch.squeeze(src_word.type(torch.int64), dim=-1)
        src_word_emb = self.emb0(src_word)
        src_word_emb = self.src_emb_dim ** 0.5 * src_word_emb
        src_pos = torch.squeeze(src_pos, dim=-1)
        src_pos_enc = self.emb1(src_pos)
        src_pos_enc.stop_gradient = True
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = F.dropout(enc_input, p=self.dropout_rate)
        else:
            out = enc_input
        return out


class WrapEncoder(nn.Module):
    """
    embedder + encoder
    """

    def __init__(self, src_vocab_size, max_length, n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd, weight_sharing, bos_idx=0):
        super(WrapEncoder, self).__init__()
        self.prepare_decoder = PrepareDecoder(src_vocab_size, d_model, max_length, prepostprocess_dropout, bos_idx=bos_idx)
        self.encoder = Encoder(n_layer, n_head, d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd)

    def forward(self, enc_inputs):
        src_word, src_pos, src_slf_attn_bias = enc_inputs
        enc_input = self.prepare_decoder(src_word, src_pos)
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class GSRM(nn.Module):

    def __init__(self, in_channels, char_num, max_text_length, num_heads, num_encoder_tus, num_decoder_tus, hidden_dims):
        super(GSRM, self).__init__()
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.num_decoder_TUs = num_decoder_tus
        self.hidden_dims = hidden_dims
        self.fc0 = torch.nn.Linear(in_features=in_channels, out_features=self.char_num)
        self.wrap_encoder0 = WrapEncoder(src_vocab_size=self.char_num + 1, max_length=self.max_length, n_layer=self.num_decoder_TUs, n_head=self.num_heads, d_key=int(self.hidden_dims / self.num_heads), d_value=int(self.hidden_dims / self.num_heads), d_model=self.hidden_dims, d_inner_hid=self.hidden_dims, prepostprocess_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, preprocess_cmd='n', postprocess_cmd='da', weight_sharing=True)
        self.wrap_encoder1 = WrapEncoder(src_vocab_size=self.char_num + 1, max_length=self.max_length, n_layer=self.num_decoder_TUs, n_head=self.num_heads, d_key=int(self.hidden_dims / self.num_heads), d_value=int(self.hidden_dims / self.num_heads), d_model=self.hidden_dims, d_inner_hid=self.hidden_dims, prepostprocess_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, preprocess_cmd='n', postprocess_cmd='da', weight_sharing=True)
        self.mul = lambda x: torch.matmul(x, self.wrap_encoder0.prepare_decoder.emb0.weight.t())

    def forward(self, inputs, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2):
        b, t, c = inputs.shape
        pvam_features = torch.reshape(inputs, [-1, c])
        word_out = self.fc0(pvam_features)
        word_ids = torch.argmax(F.softmax(word_out, dim=-1), dim=1)
        word_ids = torch.reshape(word_ids, shape=[-1, t, 1])
        """
        This module is achieved through bi-transformers,
        ngram_feature1 is the froward one, ngram_fetaure2 is the backward one
        """
        pad_idx = self.char_num
        word1 = F.pad(word_ids.type(torch.float32), [0, 0, 1, 0, 0, 0], value=1.0 * pad_idx)
        word1 = word1.type(torch.int64)
        word1 = word1[:, :-1, :]
        word2 = word_ids
        enc_inputs_1 = [word1, gsrm_word_pos, gsrm_slf_attn_bias1]
        enc_inputs_2 = [word2, gsrm_word_pos, gsrm_slf_attn_bias2]
        gsrm_feature1 = self.wrap_encoder0(enc_inputs_1)
        gsrm_feature2 = self.wrap_encoder1(enc_inputs_2)
        gsrm_feature2 = F.pad(gsrm_feature2, [0, 0, 0, 1, 0, 0], value=0.0)
        gsrm_feature2 = gsrm_feature2[:, 1:]
        gsrm_features = gsrm_feature1 + gsrm_feature2
        gsrm_out = self.mul(gsrm_features)
        b, t, c = gsrm_out.shape
        gsrm_out = torch.reshape(gsrm_out, [-1, c])
        return gsrm_features, word_out, gsrm_out


class VSFD(nn.Module):

    def __init__(self, in_channels=512, pvam_ch=512, char_num=38):
        super(VSFD, self).__init__()
        self.char_num = char_num
        self.fc0 = torch.nn.Linear(in_features=in_channels * 2, out_features=pvam_ch)
        self.fc1 = torch.nn.Linear(in_features=pvam_ch, out_features=self.char_num)

    def forward(self, pvam_feature, gsrm_feature):
        b, t, c1 = pvam_feature.shape
        b, t, c2 = gsrm_feature.shape
        combine_feature_ = torch.cat([pvam_feature, gsrm_feature], dim=2)
        img_comb_feature_ = torch.reshape(combine_feature_, shape=[-1, c1 + c2])
        img_comb_feature_map = self.fc0(img_comb_feature_)
        img_comb_feature_map = torch.sigmoid(img_comb_feature_map)
        img_comb_feature_map = torch.reshape(img_comb_feature_map, shape=[-1, t, c1])
        combine_feature = img_comb_feature_map * pvam_feature + (1.0 - img_comb_feature_map) * gsrm_feature
        img_comb_feature = torch.reshape(combine_feature, shape=[-1, c1])
        out = self.fc1(img_comb_feature)
        return out


class SRNHead(nn.Module):

    def __init__(self, in_channels, out_channels, max_text_length, num_heads, num_encoder_TUs, num_decoder_TUs, hidden_dims, **kwargs):
        super(SRNHead, self).__init__()
        self.char_num = out_channels
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_TUs
        self.num_decoder_TUs = num_decoder_TUs
        self.hidden_dims = hidden_dims
        self.pvam = PVAM(in_channels=in_channels, char_num=self.char_num, max_text_length=self.max_length, num_heads=self.num_heads, num_encoder_tus=self.num_encoder_TUs, hidden_dims=self.hidden_dims)
        self.gsrm = GSRM(in_channels=in_channels, char_num=self.char_num, max_text_length=self.max_length, num_heads=self.num_heads, num_encoder_tus=self.num_encoder_TUs, num_decoder_tus=self.num_decoder_TUs, hidden_dims=self.hidden_dims)
        self.vsfd = VSFD(in_channels=in_channels, char_num=self.char_num)
        self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.gsrm.wrap_encoder0.prepare_decoder.emb0

    def forward(self, inputs, others):
        encoder_word_pos = others[0]
        gsrm_word_pos = others[1].type(torch.long)
        gsrm_slf_attn_bias1 = others[2]
        gsrm_slf_attn_bias2 = others[3]
        pvam_feature = self.pvam(inputs, encoder_word_pos, gsrm_word_pos)
        gsrm_feature, word_out, gsrm_out = self.gsrm(pvam_feature, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2)
        final_out = self.vsfd(pvam_feature, gsrm_feature)
        if not self.training:
            final_out = F.softmax(final_out, dim=1)
        _, decoded_out = torch.topk(final_out, k=1)
        predicts = OrderedDict([('predict', final_out), ('pvam_feature', pvam_feature), ('decoded_out', decoded_out), ('word_out', word_out), ('gsrm_out', gsrm_out)])
        return predicts


class AttentionGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)
        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.matmul(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots.float()], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return (cur_hidden, cur_hidden), alpha


class TableAttentionHead(nn.Module):

    def __init__(self, in_channels, hidden_size, loc_type, in_max_len=488, **kwargs):
        super(TableAttentionHead, self).__init__()
        self.input_size = in_channels[-1]
        self.hidden_size = hidden_size
        self.elem_num = 30
        self.max_text_length = 100
        self.max_elem_length = kwargs.get('max_elem_length', 500)
        self.max_cell_num = 500
        self.structure_attention_cell = AttentionGRUCell(self.input_size, hidden_size, self.elem_num, use_gru=False)
        self.structure_generator = nn.Linear(hidden_size, self.elem_num)
        self.loc_type = loc_type
        self.in_max_len = in_max_len
        if self.loc_type == 1:
            self.loc_generator = nn.Linear(hidden_size, 4)
        else:
            if self.in_max_len == 640:
                self.loc_fea_trans = nn.Linear(400, self.max_elem_length + 1)
            elif self.in_max_len == 800:
                self.loc_fea_trans = nn.Linear(625, self.max_elem_length + 1)
            else:
                self.loc_fea_trans = nn.Linear(256, self.max_elem_length + 1)
            self.loc_generator = nn.Linear(self.input_size + hidden_size, 4)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char.type(torch.int64), onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None):
        fea = inputs[-1]
        if len(fea.shape) == 3:
            pass
        else:
            last_shape = int(np.prod(fea.shape[2:]))
            fea = torch.reshape(fea, [fea.shape[0], fea.shape[1], last_shape])
            fea = fea.permute(0, 2, 1)
        batch_size = fea.shape[0]
        hidden = torch.zeros((batch_size, self.hidden_size))
        output_hiddens = []
        if self.training and targets is not None:
            raise NotImplementedError
        else:
            temp_elem = torch.zeros([batch_size], dtype=torch.int32)
            structure_probs = None
            loc_preds = None
            elem_onehots = None
            outputs = None
            alpha = None
            max_elem_length = torch.as_tensor(self.max_elem_length)
            i = 0
            while i < max_elem_length + 1:
                elem_onehots = self._char_to_onehot(temp_elem, onehot_dim=self.elem_num)
                (outputs, hidden), alpha = self.structure_attention_cell(hidden, fea, elem_onehots)
                output_hiddens.append(torch.unsqueeze(outputs, dim=1))
                structure_probs_step = self.structure_generator(outputs)
                temp_elem = structure_probs_step.argmax(dim=1, keepdim=False)
                i += 1
            output = torch.cat(output_hiddens, dim=1)
            structure_probs = self.structure_generator(output)
            structure_probs = F.softmax(structure_probs, dim=-1)
            if self.loc_type == 1:
                loc_preds = self.loc_generator(output)
                loc_preds = F.sigmoid(loc_preds)
            else:
                loc_fea = fea.permute(0, 2, 1)
                loc_fea = self.loc_fea_trans(loc_fea)
                loc_fea = loc_fea.permute(0, 2, 1)
                loc_concat = torch.cat([output, loc_fea], dim=2)
                loc_preds = self.loc_generator(loc_concat)
                loc_preds = F.sigmoid(loc_preds)
        return {'structure_probs': structure_probs, 'loc_preds': loc_preds}


class AttentionLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionLSTMCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        if not use_gru:
            self.rnn = nn.LSTMCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        else:
            self.rnn = nn.GRUCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden[0]), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)
        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.matmul(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots.float()], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return (cur_hidden, cur_hidden), alpha


class AttentionLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionLSTM, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels
        self.attention_cell = AttentionLSTMCell(in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length
        hidden = torch.zeros((batch_size, self.hidden_size)), torch.zeros((batch_size, self.hidden_size))
        output_hiddens = []
        if targets is not None:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                hidden = hidden[1][0], hidden[1][1]
                output_hiddens.append(torch.unsqueeze(hidden[0], dim=1))
            output = torch.cat(output_hiddens, dim=1)
            probs = self.generator(output)
        else:
            targets = torch.zeros([batch_size], dtype=torch.int32)
            probs = None
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden[0])
                hidden = hidden[1][0], hidden[1][1]
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.cat([probs, torch.unsqueeze(probs_step, dim=1)], dim=1)
                next_input = probs_step.argmax(dim=1)
                targets = next_input
        return probs


class TableFPN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(TableFPN, self).__init__()
        self.out_channels = 512
        self.in2_conv = nn.Conv2d(in_channels=in_channels[0], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels=in_channels[1], out_channels=self.out_channels, kernel_size=1, stride=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels=in_channels[2], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels=in_channels[3], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.p5_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p2_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.fuse_conv = nn.Conv2d(in_channels=self.out_channels * 4, out_channels=512, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)
        out4 = in4 + F.interpolate(in5, size=in4.shape[2:4], mode='nearest')
        out3 = in3 + F.interpolate(out4, size=in3.shape[2:4], mode='nearest')
        out2 = in2 + F.interpolate(out3, size=in2.shape[2:4], mode='nearest')
        p4 = F.interpolate(out4, size=in5.shape[2:4], mode='nearest')
        p3 = F.interpolate(out3, size=in5.shape[2:4], mode='nearest')
        p2 = F.interpolate(out2, size=in5.shape[2:4], mode='nearest')
        fuse = torch.cat([in5, p4, p3, p2], dim=1)
        fuse_conv = self.fuse_conv(fuse) * 0.005
        return [c5 + fuse_conv]


def hard_sigmoid(x, slope=0.1666667, offset=0.5):
    return torch.clamp(slope * x + offset, 0.0, 1.0)


class SEModule(nn.Module):

    def __init__(self, in_channels, reduction=4, name=''):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hard_sigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs


class ResidualUnit(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se, act=None, name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se
        self.expand_conv = ConvBNLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, if_act=True, act=act, name=name + '_expand')
        self.bottleneck_conv = ConvBNLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2), groups=mid_channels, if_act=True, act=act, name=name + '_depthwise')
        if self.if_se:
            self.mid_se = SEModule(mid_channels, name=name + '_se')
        self.linear_conv = ConvBNLayer(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, if_act=False, act=None, name=name + '_linear')

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = torch.add(inputs, x)
        return x


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Module):

    def __init__(self, in_channels=3, model_name='large', scale=0.5, disable_se=False, **kwargs):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super(MobileNetV3, self).__init__()
        self.disable_se = disable_se
        if model_name == 'large':
            cfg = [[3, 16, 16, False, 'relu', 1], [3, 64, 24, False, 'relu', 2], [3, 72, 24, False, 'relu', 1], [5, 72, 40, True, 'relu', 2], [5, 120, 40, True, 'relu', 1], [5, 120, 40, True, 'relu', 1], [3, 240, 80, False, 'hardswish', 2], [3, 200, 80, False, 'hardswish', 1], [3, 184, 80, False, 'hardswish', 1], [3, 184, 80, False, 'hardswish', 1], [3, 480, 112, True, 'hardswish', 1], [3, 672, 112, True, 'hardswish', 1], [5, 672, 160, True, 'hardswish', 2], [5, 960, 160, True, 'hardswish', 1], [5, 960, 160, True, 'hardswish', 1]]
            cls_ch_squeeze = 960
        elif model_name == 'small':
            cfg = [[3, 16, 16, True, 'relu', 2], [3, 72, 24, False, 'relu', 2], [3, 88, 24, False, 'relu', 1], [5, 96, 40, True, 'hardswish', 2], [5, 240, 40, True, 'hardswish', 1], [5, 240, 40, True, 'hardswish', 1], [5, 120, 48, True, 'hardswish', 1], [5, 144, 48, True, 'hardswish', 1], [5, 288, 96, True, 'hardswish', 2], [5, 576, 96, True, 'hardswish', 1], [5, 576, 96, True, 'hardswish', 1]]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError('mode[' + model_name + '_model] is not implemented!')
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, 'supported scale are {} but input scale is {}'.format(supported_scale, scale)
        inplanes = 16
        self.conv = ConvBNLayer(in_channels=in_channels, out_channels=make_divisible(inplanes * scale), kernel_size=3, stride=2, padding=1, groups=1, if_act=True, act='hardswish', name='conv1')
        self.stages = nn.ModuleList()
        self.out_channels = []
        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for k, exp, c, se, nl, s in cfg:
            se = se and not self.disable_se
            start_idx = 2 if model_name == 'large' else 0
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block_list.append(ResidualUnit(in_channels=inplanes, mid_channels=make_divisible(scale * exp), out_channels=make_divisible(scale * c), kernel_size=k, stride=s, use_se=se, act=nl, name='conv' + str(i + 2)))
            inplanes = make_divisible(scale * c)
            i += 1
        block_list.append(ConvBNLayer(in_channels=inplanes, out_channels=make_divisible(scale * cls_ch_squeeze), kernel_size=1, stride=1, padding=0, groups=1, if_act=True, act='hardswish', name='conv_last'))
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))

    def forward(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


DISABLE_SE = True


INPUT_SIZE = 1, 3, 488, 488


IN_CHANNELS = INPUT_SIZE[1]


MODEL_NAME = 'large'


SCALE = 1.0


class PTNet(torch.nn.Module):

    def __init__(self, **kwargs):
        super(PTNet, self).__init__()
        self.backbone = pt_table_mobilenet_v3.MobileNetV3(in_channels=IN_CHANNELS, model_name=MODEL_NAME, scale=SCALE, disable_se=DISABLE_SE)
        head_in_channels = self.backbone.out_channels

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        return x


class DeformableConvV2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, weight_attr=None, bias_attr=None, lr_scale=1, regularizer=None, skip_quant=False, dcn_bias_regularizer=None, dcn_bias_lr_scale=2.0):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size ** 2 * groups
        self.mask_channel = kernel_size ** 2 * groups
        if bias_attr:
            dcn_bias_attr = True
        else:
            dcn_bias_attr = False
        self.conv_dcn = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2 * dilation, dilation=dilation, groups=groups // 2 if groups > 1 else 1, bias=dcn_bias_attr)
        self.conv_offset = nn.Conv2d(in_channels, groups * 3 * kernel_size ** 2, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True)
        if skip_quant:
            self.conv_offset.skip_quant = True

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = torch.split(offset_mask, split_size_or_sections=[self.offset_channel, self.mask_channel], dim=1)
        mask = torch.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class DropBlock(nn.Module):

    def __init__(self, block_size, keep_prob, name, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890
        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1.0 - self.keep_prob) / self.block_size ** 2
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)
            matrix = torch.rand(x.size(), dtype=x.dtype) < gamma
            mask_inv = F.max_pool2d(matrix, self.block_size, stride=1, padding=self.block_size // 2, data_format=self.data_format)
            mask = 1.0 - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class PPYOLODetBlockCSP(nn.Module):

    def __init__(self, cfg, ch_in, ch_out, act, norm_type, name, data_format='NCHW'):
        """
        PPYOLODetBlockCSP layer
        Args:
            cfg (list): layer configs for this block
            ch_in (int): input channel
            ch_out (int): output channel
            act (str): default mish
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlockCSP, self).__init__()
        self.data_format = data_format
        self.conv1 = ConvBNLayer(ch_in, ch_out, 1, padding=0, act=act, norm_type=norm_type, name=name + '_left', data_format=data_format)
        self.conv2 = ConvBNLayer(ch_in, ch_out, 1, padding=0, act=act, norm_type=norm_type, name=name + '_right', data_format=data_format)
        self.conv3 = ConvBNLayer(ch_out * 2, ch_out * 2, 1, padding=0, act=act, norm_type=norm_type, name=name, data_format=data_format)
        self.conv_module = nn.Sequential()
        for idx, (layer_name, layer, args, kwargs) in enumerate(cfg):
            name = name.replace('.', '_')
            layer_name = layer_name.replace('.', '_')
            kwargs.update(name=name + '_' + layer_name, data_format=data_format)
            _layer = layer(*args, **kwargs)
            self.conv_module.add_module(layer_name, _layer)

    def forward(self, inputs):
        conv_left = self.conv1(inputs)
        conv_right = self.conv2(inputs)
        conv_left = self.conv_module(conv_left)
        if self.data_format == 'NCHW':
            conv = torch.cat([conv_left, conv_right], dim=1)
        else:
            conv = torch.cat([conv_left, conv_right], dim=-1)
        conv = self.conv3(conv)
        return conv, conv


class SPP(nn.Module):

    def __init__(self, ch_in, ch_out, k, pool_size, norm_type, freeze_norm=False, name='', act='leaky', data_format='NCHW'):
        """
        SPP layer, which consist of four pooling layer follwed by conv layer
        Args:
            ch_in (int): input channel of conv layer
            ch_out (int): output channel of conv layer
            k (int): kernel size of conv layer
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            act (str): activation function
            data_format (str): data format, NCHW or NHWC
        """
        super(SPP, self).__init__()
        self.pool = nn.ModuleList()
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            self.pool.add_module('{}_pool_{}'.format(name, i), nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False))
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, norm_type=norm_type, freeze_norm=freeze_norm, name=name, act=act, data_format=data_format)

    def forward(self, x):
        outs = [x]
        for i, pool in enumerate(self.pool):
            outs.append(pool(x))
        if self.data_format == 'NCHW':
            y = torch.cat(outs, dim=1)
        else:
            y = torch.cat(outs, dim=-1)
        y = self.conv(y)
        return y


class PPYOLOPAN(nn.Module):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self, in_channels=[512, 1024, 2048], norm_type='bn', data_format='NCHW', act='mish', conv_block_num=3, drop_block=False, block_size=3, keep_prob=1.0, spp=False):
        """
        PPYOLOPAN layer with SPP, DropBlock and CSP connection.
        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            act (str): activation function, default mish
            conv_block_num (int): conv block num of each pan block
            drop_block (bool): whether use DropBlock or not
            block_size (int): block size of DropBlock
            keep_prob (float): keep probability of DropBlock
            spp (bool): whether use spp or not
        """
        super(PPYOLOPAN, self).__init__()
        assert len(in_channels) > 0, 'in_channels length should > 0'
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.spp = spp
        self.conv_block_num = conv_block_num
        self.data_format = data_format
        if self.drop_block:
            dropblock_cfg = [['dropblock', DropBlock, [self.block_size, self.keep_prob], dict()]]
        else:
            dropblock_cfg = []
        self.fpn_blocks = nn.ModuleList()
        self.fpn_routes = nn.ModuleDict()
        self.fpn_routes_names = []
        fpn_channels = []
        for i, ch_in in enumerate(self.in_channels[::-1]):
            if i > 0:
                ch_in += 512 // 2 ** (i - 1)
            channel = 512 // 2 ** i
            base_cfg = []
            for j in range(self.conv_block_num):
                base_cfg += [['{}_0'.format(j), ConvBNLayer, [channel, channel, 1], dict(padding=0, act=act, norm_type=norm_type)], ['{}_1'.format(j), ConvBNLayer, [channel, channel, 3], dict(padding=1, act=act, norm_type=norm_type)]]
            if i == 0 and self.spp:
                base_cfg[3] = ['spp', SPP, [channel * 4, channel, 1], dict(pool_size=[5, 9, 13], act=act, norm_type=norm_type)]
            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = 'fpn_{}'.format(i)
            self.fpn_blocks.add_module(name, PPYOLODetBlockCSP(cfg, ch_in, channel, act, norm_type, name, data_format))
            fpn_channels.append(channel * 2)
            if i < self.num_blocks - 1:
                name = 'fpn_transition_{}'.format(i)
                self.fpn_routes.add_module(name, ConvBNLayer(ch_in=channel * 2, ch_out=channel, filter_size=1, stride=1, padding=0, act=act, norm_type=norm_type, data_format=data_format, name=name))
                self.fpn_routes_names.append(name)
        self.pan_blocks = nn.ModuleDict()
        self.pan_blocks_names = []
        self.pan_routes = nn.ModuleDict()
        self.pan_routes_names = []
        self._out_channels = [512 // 2 ** (self.num_blocks - 2)]
        for i in reversed(range(self.num_blocks - 1)):
            name = 'pan_transition_{}'.format(i)
            self.pan_routes.add_module(name, ConvBNLayer(ch_in=fpn_channels[i + 1], ch_out=fpn_channels[i + 1], filter_size=3, stride=2, padding=1, act=act, norm_type=norm_type, data_format=data_format, name=name))
            route_name = [name] + self.pan_routes_names
            self.pan_routes_names = route_name
            base_cfg = []
            ch_in = fpn_channels[i] + fpn_channels[i + 1]
            channel = 512 // 2 ** i
            for j in range(self.conv_block_num):
                base_cfg += [['{}_0'.format(j), ConvBNLayer, [channel, channel, 1], dict(padding=0, act=act, norm_type=norm_type)], ['{}_1'.format(j), ConvBNLayer, [channel, channel, 3], dict(padding=1, act=act, norm_type=norm_type)]]
            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = 'pan_{}'.format(i)
            self.pan_blocks.add_module(name, PPYOLODetBlockCSP(cfg, ch_in, channel, act, norm_type, name, data_format))
            pan_block_name = [name] + self.pan_blocks_names
            self.pan_blocks_names = pan_block_name
            self._out_channels.append(channel * 2)
        self._out_channels = self._out_channels[::-1]

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        fpn_feats = []
        if for_mot:
            emb_feats = []
        for i, (block, fpn_block) in enumerate(zip(blocks, self.fpn_blocks)):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = torch.cat([route, block], dim=1)
                else:
                    block = torch.cat([route, block], dim=-1)
            route, tip = fpn_block(block)
            fpn_feats.append(tip)
            if for_mot:
                emb_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.fpn_routes[self.fpn_routes_names[i]](route)
                route = F.interpolate(route, scale_factor=2.0)
        pan_feats = [fpn_feats[-1]]
        route = fpn_feats[self.num_blocks - 1]
        for i, pan_route_name, pan_block_name in zip(reversed(range(self.num_blocks - 1)), reversed(self.pan_routes_names), reversed(self.pan_blocks_names)):
            block = fpn_feats[i]
            route = self.pan_routes[pan_route_name](route)
            if self.data_format == 'NCHW':
                block = torch.cat([route, block], dim=1)
            else:
                block = torch.cat([route, block], dim=-1)
            route, tip = self.pan_blocks[pan_block_name](block)
            pan_feats.append(tip)
        if for_mot:
            return {'yolo_feats': pan_feats[::-1], 'emb_feats': emb_feats}
        else:
            return pan_feats[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}


class ResNet(nn.Module):

    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, 'supported layers are {} but input layer is {}'.format(supported_layers, layers)
        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]
        self.conv1_1 = ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, act='relu', name='conv1_1')
        self.conv1_2 = ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu', name='conv1_2')
        self.conv1_3 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu', name='conv1_3')
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        self.out_channels = []
        if layers >= 50:
            for block in range(len(depth)):
                block_list = nn.Sequential()
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = 'res' + str(block + 2) + 'a'
                        else:
                            conv_name = 'res' + str(block + 2) + 'b' + str(i)
                    else:
                        conv_name = 'res' + str(block + 2) + chr(97 + i)
                    bottleneck_block = BottleneckBlock(in_channels=num_channels[block] if i == 0 else num_filters[block] * 4, out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, shortcut=shortcut, if_first=block == i == 0, name=conv_name)
                    shortcut = True
                    block_list.add_module('bb_%d_%d' % (block, i), bottleneck_block)
                self.out_channels.append(num_filters[block] * 4)
                self.stages.append(block_list)
        else:
            for block in range(len(depth)):
                block_list = nn.Sequential()
                shortcut = False
                for i in range(depth[block]):
                    conv_name = 'res' + str(block + 2) + chr(97 + i)
                    basic_block = BasicBlock(in_channels=num_channels[block] if i == 0 else num_filters[block], out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, shortcut=shortcut, if_first=block == i == 0, name=conv_name)
                    shortcut = True
                    block_list.add_module('bb_%d_%d' % (block, i), basic_block)
                self.out_channels.append(num_filters[block])
                self.stages.append(block_list)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        out = []
        for block in self.stages:
            y = block(y)
            out.append(y)
        return out


def _de_sigmoid(x, eps=1e-07):
    x = torch.clip(x, eps, 1.0 / eps)
    x = torch.clip(1.0 / x - 1.0, eps, 1.0 / eps)
    x = -torch.log(x)
    return x


class YOLOv3Head(nn.Module):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['loss']

    def __init__(self, in_channels=[1024, 512, 256], anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]], anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes=80, loss='YOLOv3Loss', iou_aware=False, iou_aware_factor=0.4, data_format='NCHW'):
        """
        Head for YOLOv3 network
        Args:
            num_classes (int): number of foreground classes
            anchors (list): anchors
            anchor_masks (list): anchor masks
            loss (object): YOLOv3Loss instance
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOv3Head, self).__init__()
        assert len(in_channels) > 0, 'in_channels length should > 0'
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss = loss
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)
        self.data_format = data_format
        self.yolo_outputs = nn.ModuleList()
        self.yolo_outputs_names = []
        for i in range(len(self.anchors)):
            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            name = 'yolo_output_{}'.format(i)
            conv = nn.Conv2d(in_channels=self.in_channels[i], out_channels=num_filters, kernel_size=1, stride=1, padding=0)
            conv.skip_quant = True
            self.yolo_outputs.add_module(name, conv)
            self.yolo_outputs_names.append(name)

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, 'anchor mask index overflow'
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, (feat, yolo_output) in enumerate(zip(feats, self.yolo_outputs)):
            yolo_output = yolo_output(feat)
            if self.data_format == 'NHWC':
                yolo_output = yolo_output.permute(0, 3, 1, 2)
            yolo_outputs.append(yolo_output)
        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors)
        elif self.iou_aware:
            y = []
            for i, out in enumerate(yolo_outputs):
                na = len(self.anchors[i])
                ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                b, c, h, w = x.shape
                no = c // na
                x = x.reshape((b, na, no, h * w))
                ioup = ioup.reshape((b, na, 1, h * w))
                obj = x[:, :, 4:5, :]
                ioup = F.sigmoid(ioup)
                obj = F.sigmoid(obj)
                obj_t = obj ** (1 - self.iou_aware_factor) * ioup ** self.iou_aware_factor
                obj_t = _de_sigmoid(obj_t)
                loc_t = x[:, :, :4, :]
                cls_t = x[:, :, 5:, :]
                y_t = torch.cat([loc_t, obj_t, cls_t], dim=2)
                y_t = y_t.reshape((b, c, h, w))
                y.append(y_t)
            return y
        else:
            return yolo_outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}


class PPYOLOv2Base(nn.Module):

    def __init__(self, **kwargs):
        super(PPYOLOv2Base, self).__init__()
        self._init_params(**kwargs)
        self._init_network()
        self._initialize_weights()

    def _init_params(self, **kwargs):
        self.num_classes = kwargs.get('INIT_num_classes', 80)
        self.arch = kwargs.get('INIT_arch', 50)
        self.scale_x_y = kwargs.get('INIT_scale_x_y', 1.05)
        self.downsample_ratio = kwargs.get('INIT_downsample_ratio', 32)
        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_network(self):
        if self.arch == 50:
            self._init_network_resnet50()
        elif self.arch == 101:
            self._init_network_resnet101()
        else:
            raise ValueError('INIT_arch must be [50, 101], but got {}'.format(self.arch))

    def _init_network_resnet50(self):
        self.backbone = ResNet(depth=50, ch_in=64, variant='d', lr_mult_list=[1.0, 1.0, 1.0, 1.0], groups=1, base_width=64, norm_type='bn', norm_decay=0, freeze_norm=False, freeze_at=-1, return_idx=[1, 2, 3], dcn_v2_stages=[3], num_stages=4, std_senet=False)
        self.neck = PPYOLOPAN(in_channels=[512, 1024, 2048], norm_type='bn', data_format='NCHW', act='mish', conv_block_num=3, drop_block=True, block_size=3, keep_prob=1.0, spp=True)
        self.head = YOLOv3Head(in_channels=[1024, 512, 256], anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]], anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes=self.num_classes, loss='YOLOv3Loss', iou_aware=True, iou_aware_factor=0.5, data_format='NCHW')

    def _init_network_resnet101(self):
        self.backbone = ResNet(depth=101, ch_in=64, variant='d', lr_mult_list=[1.0, 1.0, 1.0, 1.0], groups=1, base_width=64, norm_type='bn', norm_decay=0, freeze_norm=False, freeze_at=-1, return_idx=[1, 2, 3], dcn_v2_stages=[3], num_stages=4, std_senet=False)
        self.neck = PPYOLOPAN(in_channels=[512, 1024, 2048], norm_type='bn', data_format='NCHW', act='mish', conv_block_num=3, drop_block=False, block_size=3, keep_prob=1.0, spp=True)
        self.head = YOLOv3Head(in_channels=[1024, 512, 256], anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]], anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes=self.num_classes, loss='YOLOv3Loss', iou_aware=True, iou_aware_factor=0.5, data_format='NCHW')

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def load_paddle_weights(self, weights_path):
        None
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        sd = para_state_dict
        for key, value in sd.items():
            None
        for key, value in self.state_dict().items():
            None
        for key, value in self.state_dict().items():
            if key.endswith('num_batches_tracked'):
                continue
            ppname = key
            ppname = ppname.replace('.running_mean', '._mean')
            ppname = ppname.replace('.running_var', '._variance')
            if key.startswith('backbone.conv'):
                pass
            if key.startswith('backbone.res_layers'):
                ppname = ppname.replace('.res_layers', '')
                ppname = ppname.replace('.blocks', '')
            if key.startswith('neck.fpn_blocks'):
                ppname = ppname.replace('.fpn_blocks', '')
                ppname = ppname.replace('.fpn_', '.fpn.')
                ppname = ppname.replace('.conv_module.0_0', '.conv_module.0.0')
                ppname = ppname.replace('.conv_module.0_1', '.conv_module.0.1')
                ppname = ppname.replace('.conv_module.1_0', '.conv_module.1.0')
                ppname = ppname.replace('.conv_module.1_1', '.conv_module.1.1')
                ppname = ppname.replace('.conv_module.2_0', '.conv_module.2.0')
                ppname = ppname.replace('.conv_module.2_1', '.conv_module.2.1')
            if key.startswith('neck.fpn_routes'):
                ppname = ppname.replace('.fpn_routes', '')
                ppname = ppname.replace('.fpn_transition_', '.fpn_transition.')
            if key.startswith('neck.pan_blocks'):
                ppname = ppname.replace('.pan_blocks', '')
                ppname = ppname.replace('.pan_', '.pan.')
                ppname = ppname.replace('.conv_module.0_0', '.conv_module.0.0')
                ppname = ppname.replace('.conv_module.0_1', '.conv_module.0.1')
                ppname = ppname.replace('.conv_module.1_0', '.conv_module.1.0')
                ppname = ppname.replace('.conv_module.1_1', '.conv_module.1.1')
                ppname = ppname.replace('.conv_module.2_0', '.conv_module.2.0')
                ppname = ppname.replace('.conv_module.2_1', '.conv_module.2.1')
            if key.startswith('neck.pan_routes'):
                ppname = ppname.replace('.pan_routes', '')
                ppname = ppname.replace('.pan_transition_', '.pan_transition.')
            if key.startswith('head.yolo_outputs'):
                ppname = ppname.replace('head.yolo_outputs.yolo_output_', 'yolo_head.yolo_output.')
            try:
                weights = sd[ppname]
                self.state_dict()[key].copy_(torch.Tensor(weights))
            except Exception as e:
                None
                None
                raise e
        None

    def load_pytorch_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))
        None

    def save_pytorch_weights(self, weights_path):
        try:
            torch.save(self.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.state_dict(), weights_path)
        None


class DownSample(nn.Module):

    def __init__(self, ch_in, ch_out, filter_size=3, stride=2, padding=1, norm_type='bn', norm_decay=0.0, freeze_norm=False, data_format='NCHW'):
        """
        downsample layer
        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            stride (int): stride, default 2
            padding (int): padding size, default 1
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """
        super(DownSample, self).__init__()
        self.conv_bn_layer = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, filter_size=filter_size, stride=stride, padding=padding, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, data_format=data_format)
        self.ch_out = ch_out

    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out


class Blocks(nn.Module):

    def __init__(self, block, ch_in, ch_out, count, name_adapter, stage_num, variant='b', groups=1, base_width=64, lr=1.0, norm_type='bn', norm_decay=0.0, freeze_norm=True, dcn_v2=False, std_senet=False):
        super(Blocks, self).__init__()
        self.blocks = nn.Sequential()
        for i in range(count):
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)
            layer = block(ch_in=ch_in, ch_out=ch_out, stride=2 if i == 0 and stage_num != 2 else 1, shortcut=False if i == 0 else True, variant=variant, groups=groups, base_width=base_width, lr=lr, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, dcn_v2=dcn_v2, std_senet=std_senet)
            self.blocks.add_module(conv_name, layer)
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, inputs):
        block_out = self.blocks(inputs)
        return block_out


DarkNet_cfg = {(53): [1, 2, 8, 8, 4]}


class DarkNet(nn.Module):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self, depth=53, freeze_at=-1, return_idx=[2, 3, 4], num_stages=5, norm_type='bn', norm_decay=0.0, freeze_norm=False, data_format='NCHW'):
        """
        Darknet, see https://pjreddie.com/darknet/yolo/
        Args:
            depth (int): depth of network
            freeze_at (int): freeze the backbone at which stage
            filter_size (int): filter size, default 3
            return_idx (list): index of stages whose feature maps are returned
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            data_format (str): data format, NCHW or NHWC
        """
        super(DarkNet, self).__init__()
        self.depth = depth
        self.freeze_at = freeze_at
        self.return_idx = return_idx
        self.num_stages = num_stages
        self.stages = DarkNet_cfg[self.depth][0:num_stages]
        self.conv0 = ConvBNLayer(ch_in=3, ch_out=32, filter_size=3, stride=1, padding=1, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, data_format=data_format)
        self.downsample0 = DownSample(ch_in=32, ch_out=32 * 2, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, data_format=data_format)
        self._out_channels = []
        self.darknet_conv_block_list = nn.ModuleList()
        self.downsample_list = nn.ModuleDict()
        self.downsample_list_names = []
        ch_in = [64, 128, 256, 512, 1024]
        for i, stage in enumerate(self.stages):
            name = 'stage_{}'.format(i)
            conv_block = Blocks(int(ch_in[i]), 32 * 2 ** i, stage, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, data_format=data_format, name=name)
            self.darknet_conv_block_list.add_module(name, conv_block)
            if i in return_idx:
                self._out_channels.append(64 * 2 ** i)
        for i in range(num_stages - 1):
            down_name = 'stage_{}_downsample'.format(i)
            downsample = DownSample(ch_in=32 * 2 ** (i + 1), ch_out=32 * 2 ** (i + 2), norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, data_format=data_format)
            self.downsample_list.add_module(down_name, downsample)
            self.downsample_list_names.append(down_name)

    def forward(self, inputs):
        x = inputs
        out = self.conv0(x)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet_conv_block_list):
            out = conv_block_i(out)
            if i in self.return_idx:
                blocks.append(out)
            if i < self.num_stages - 1:
                out = self.downsample_list[self.downsample_list_names[i]](out)
        return blocks


class ConvNormLayer(nn.Module):

    def __init__(self, ch_in, ch_out, filter_size, stride, groups=1, norm_type='bn', norm_decay=0.0, norm_groups=32, lr_scale=1.0, freeze_norm=False, initializer=None):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn']
        bias_attr = False
        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, groups=groups, bias=bias_attr)
        norm_lr = 0.0 if freeze_norm else 1.0
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(ch_out)
        elif norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(ch_out)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out, affine=bias_attr)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out


class SELayer(nn.Module):

    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        stdv = 1.0 / math.sqrt(ch)
        c_ = ch // reduction_ratio
        self.squeeze = nn.Linear(in_features=ch, out_features=c_, bias=True)
        stdv = 1.0 / math.sqrt(c_)
        self.extract = nn.Linear(in_features=c_, out_features=ch, bias=True)

    def forward(self, inputs):
        out = self.pool(inputs)
        out = out.squeeze(dim=3).squeeze(dim=2)
        out = self.squeeze(out)
        out = F.relu(out)
        out = self.extract(out)
        out = F.sigmoid(out)
        out = out.unsqueeze(dim=2).unsqueeze(dim=3)
        scale = out * inputs
        return scale


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, variant='b', groups=1, base_width=4, lr=1.0, norm_type='bn', norm_decay=0.0, freeze_norm=True, dcn_v2=False, std_senet=False):
        super(BottleNeck, self).__init__()
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride
        width = int(ch_out * (base_width / 64.0)) * groups
        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_module('conv', ConvNormLayer(ch_in=ch_in, ch_out=ch_out * self.expansion, filter_size=1, stride=1, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr))
            else:
                self.short = ConvNormLayer(ch_in=ch_in, ch_out=ch_out * self.expansion, filter_size=1, stride=stride, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr)
        self.branch2a = ConvNormLayer(ch_in=ch_in, ch_out=width, filter_size=1, stride=stride1, groups=1, act='relu', norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr)
        self.branch2b = ConvNormLayer(ch_in=width, ch_out=width, filter_size=3, stride=stride2, groups=groups, act='relu', norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr, dcn_v2=dcn_v2)
        self.branch2c = ConvNormLayer(ch_in=width, ch_out=ch_out * self.expansion, filter_size=1, stride=1, groups=1, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm, lr=lr)
        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out * self.expansion)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)
        if self.std_senet:
            out = self.se(out)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        out = torch.add(out, short)
        out = F.relu(out)
        return out


class NameAdapter(object):
    """Fix the backbones variable names for pretrained weight"""

    def __init__(self, model):
        super(NameAdapter, self).__init__()
        self.model = model

    @property
    def model_type(self):
        return getattr(self.model, '_model_type', '')

    @property
    def variant(self):
        return getattr(self.model, 'variant', '')

    def fix_conv_norm_name(self, name):
        if name == 'conv1':
            bn_name = 'bn_' + name
        else:
            bn_name = 'bn' + name[3:]
        if self.model_type == 'SEResNeXt':
            bn_name = name + '_bn'
        return bn_name

    def fix_shortcut_name(self, name):
        if self.model_type == 'SEResNeXt':
            name = 'conv' + name + '_prj'
        return name

    def fix_bottleneck_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            conv_name3 = 'conv' + name + '_x3'
            shortcut_name = name
        else:
            conv_name1 = name + '_branch2a'
            conv_name2 = name + '_branch2b'
            conv_name3 = name + '_branch2c'
            shortcut_name = name + '_branch1'
        return conv_name1, conv_name2, conv_name3, shortcut_name

    def fix_basicblock_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            shortcut_name = name
        else:
            conv_name1 = name + '_branch2a'
            conv_name2 = name + '_branch2b'
            shortcut_name = name + '_branch1'
        return conv_name1, conv_name2, shortcut_name

    def fix_layer_warp_name(self, stage_num, count, i):
        name = 'res' + str(stage_num)
        if count > 10 and stage_num == 4:
            if i == 0:
                conv_name = name + 'a'
            else:
                conv_name = name + 'b' + str(i)
        else:
            conv_name = name + chr(ord('a') + i)
        if self.model_type == 'SEResNeXt':
            conv_name = str(stage_num + 2) + '_' + str(i + 1)
        return conv_name

    def fix_c1_stage_name(self):
        return 'res_conv1' if self.model_type == 'ResNeXt' else 'conv1'


class Res5Head(nn.Module):

    def __init__(self, depth=50):
        super(Res5Head, self).__init__()
        feat_in, feat_out = [1024, 512]
        if depth < 50:
            feat_in = 256
        na = NameAdapter(self)
        block = BottleNeck if depth >= 50 else BasicBlock
        self.res5 = Blocks(block, feat_in, feat_out, count=3, name_adapter=na, stage_num=5)
        self.feat_out = feat_out if depth < 50 else feat_out * 4

    def forward(self, roi_feat, stage=0):
        y = self.res5(roi_feat)
        return y


class YoloDetBlock(nn.Module):

    def __init__(self, ch_in, channel, norm_type, freeze_norm=False, name='', data_format='NCHW'):
        """
        YOLODetBlock layer for yolov3, see https://arxiv.org/abs/1804.02767
        Args:
            ch_in (int): input channel
            channel (int): base channel
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(YoloDetBlock, self).__init__()
        self.ch_in = ch_in
        self.channel = channel
        assert channel % 2 == 0, 'channel {} cannot be divided by 2'.format(channel)
        conv_def = [['conv0', ch_in, channel, 1, '_0_0'], ['conv1', channel, channel * 2, 3, '_0_1'], ['conv2', channel * 2, channel, 1, '_1_0'], ['conv3', channel, channel * 2, 3, '_1_1'], ['route', channel * 2, channel, 1, '_2']]
        self.conv_module = nn.Sequential()
        for idx, (conv_name, ch_in, ch_out, filter_size, post_name) in enumerate(conv_def):
            self.conv_module.add_module(conv_name, ConvBNLayer(ch_in=ch_in, ch_out=ch_out, filter_size=filter_size, padding=(filter_size - 1) // 2, norm_type=norm_type, freeze_norm=freeze_norm, data_format=data_format, name=name + post_name))
        self.tip = ConvBNLayer(ch_in=channel, ch_out=channel * 2, filter_size=3, padding=1, norm_type=norm_type, freeze_norm=freeze_norm, data_format=data_format, name=name + '.tip')

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


def add_coord(x, data_format):
    b = x.size()[0]
    if data_format == 'NCHW':
        h, w = x.shape[2], x.shape[3]
    else:
        h, w = x.shape[1], x.shape[2]
    gx = torch.arange(w, dtype=x.dtype) / ((w - 1.0) * 2.0) - 1.0
    gy = torch.arange(h, dtype=x.dtype) / ((h - 1.0) * 2.0) - 1.0
    if data_format == 'NCHW':
        gx = gx.reshape([1, 1, 1, w]).expand([b, 1, h, w])
        gy = gy.reshape([1, 1, h, 1]).expand([b, 1, h, w])
    else:
        gx = gx.reshape([1, 1, w, 1]).expand([b, h, w, 1])
        gy = gy.reshape([1, h, 1, 1]).expand([b, h, w, 1])
    gx.stop_gradient = True
    gy.stop_gradient = True
    return gx, gy


class CoordConv(nn.Module):

    def __init__(self, ch_in, ch_out, filter_size, padding, norm_type, freeze_norm=False, name='', data_format='NCHW'):
        """
        CoordConv layer
        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(CoordConv, self).__init__()
        self.conv = ConvBNLayer(ch_in + 2, ch_out, filter_size=filter_size, padding=padding, norm_type=norm_type, freeze_norm=freeze_norm, data_format=data_format, name=name)
        self.data_format = data_format

    def forward(self, x):
        gx, gy = add_coord(x, self.data_format)
        if self.data_format == 'NCHW':
            y = torch.cat([x, gx, gy], dim=1)
        else:
            y = torch.cat([x, gx, gy], dim=-1)
        y = self.conv(y)
        return y


class PPYOLODetBlock(nn.Module):

    def __init__(self, cfg, name, data_format='NCHW'):
        """
        PPYOLODetBlock layer
        Args:
            cfg (list): layer configs for this block
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlock, self).__init__()
        self.conv_module = nn.Sequential()
        for idx, (conv_name, layer, args, kwargs) in enumerate(cfg[:-1]):
            kwargs.update(name='{}_{}'.format(name, conv_name), data_format=data_format)
            self.conv_module.add_module(conv_name, layer(*args, **kwargs))
        conv_name, layer, args, kwargs = cfg[-1]
        kwargs.update(name='{}_{}'.format(name, conv_name), data_format=data_format)
        self.tip = layer(*args, **kwargs)

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class PPYOLOTinyDetBlock(nn.Module):

    def __init__(self, ch_in, ch_out, name, drop_block=False, block_size=3, keep_prob=1.0, data_format='NCHW'):
        """
        PPYOLO Tiny DetBlock layer
        Args:
            ch_in (list): input channel number
            ch_out (list): output channel number
            name (str): block name
            drop_block: whether user DropBlock
            block_size: drop block size
            keep_prob: probability to keep block in DropBlock
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLOTinyDetBlock, self).__init__()
        self.drop_block_ = drop_block
        self.conv_module = nn.Sequential()
        cfgs = [['_0', ch_in, ch_out, 1, 1, 0, 1], ['_1', ch_out, ch_out, 5, 1, 2, ch_out], ['_', ch_out, ch_out, 1, 1, 0, 1], ['_route', ch_out, ch_out, 5, 1, 2, ch_out]]
        for cfg in cfgs:
            conv_name, conv_ch_in, conv_ch_out, filter_size, stride, padding, groups = cfg
            self.conv_module.add_module(name + conv_name, ConvBNLayer(ch_in=conv_ch_in, ch_out=conv_ch_out, filter_size=filter_size, stride=stride, padding=padding, groups=groups, name=name + conv_name))
        self.tip = ConvBNLayer(ch_in=ch_out, ch_out=ch_out, filter_size=1, stride=1, padding=0, groups=1, name=name + conv_name)
        if self.drop_block_:
            self.drop_block = DropBlock(block_size=block_size, keep_prob=keep_prob, data_format=data_format, name=name + '_dropblock')

    def forward(self, inputs):
        if self.drop_block_:
            inputs = self.drop_block(inputs)
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class YOLOv3FPN(nn.Module):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self, in_channels=[256, 512, 1024], norm_type='bn', freeze_norm=False, data_format='NCHW'):
        """
        YOLOv3FPN layer
        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOv3FPN, self).__init__()
        assert len(in_channels) > 0, 'in_channels length should > 0'
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        self._out_channels = []
        self.yolo_blocks = nn.ModuleDict()
        self.yolo_blocks_names = []
        self.routes = nn.ModuleDict()
        self.routes_names = []
        self.data_format = data_format
        for i in range(self.num_blocks):
            name = 'yolo_block_{}'.format(i)
            in_channel = in_channels[-i - 1]
            if i > 0:
                in_channel += 512 // 2 ** i
            yolo_block = YoloDetBlock(in_channel, channel=512 // 2 ** i, norm_type=norm_type, freeze_norm=freeze_norm, data_format=data_format, name=name)
            self.yolo_blocks.add_module(name, yolo_block)
            self.yolo_blocks_names.append(name)
            self._out_channels.append(1024 // 2 ** i)
            if i < self.num_blocks - 1:
                name = 'yolo_transition_{}'.format(i)
                route = ConvBNLayer(ch_in=512 // 2 ** i, ch_out=256 // 2 ** i, filter_size=1, stride=1, padding=0, norm_type=norm_type, freeze_norm=freeze_norm, data_format=data_format, name=name)
                self.routes.add_module(name, route)
                self.routes_names.append(name)

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []
        if for_mot:
            emb_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = torch.cat([route, block], dim=1)
                else:
                    block = torch.cat([route, block], dim=-1)
            route, tip = self.yolo_blocks[self.yolo_blocks_names[i]](block)
            yolo_feats.append(tip)
            if for_mot:
                emb_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.routes[self.routes_names[i]](route)
                route = F.interpolate(route, scale_factor=2.0)
        if for_mot:
            return {'yolo_feats': yolo_feats, 'emb_feats': emb_feats}
        else:
            return yolo_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}


class PPYOLOFPN(nn.Module):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self, in_channels=[512, 1024, 2048], norm_type='bn', freeze_norm=False, data_format='NCHW', coord_conv=False, conv_block_num=2, drop_block=False, block_size=3, keep_prob=1.0, spp=False):
        """
        PPYOLOFPN layer
        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            coord_conv (bool): whether use CoordConv or not
            conv_block_num (int): conv block num of each pan block
            drop_block (bool): whether use DropBlock or not
            block_size (int): block size of DropBlock
            keep_prob (float): keep probability of DropBlock
            spp (bool): whether use spp or not
        """
        super(PPYOLOFPN, self).__init__()
        assert len(in_channels) > 0, 'in_channels length should > 0'
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        self.coord_conv = coord_conv
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.spp = spp
        self.conv_block_num = conv_block_num
        self.data_format = data_format
        if self.coord_conv:
            ConvLayer = CoordConv
        else:
            ConvLayer = ConvBNLayer
        if self.drop_block:
            dropblock_cfg = [['dropblock', DropBlock, [self.block_size, self.keep_prob], dict()]]
        else:
            dropblock_cfg = []
        self._out_channels = []
        self.yolo_blocks = nn.ModuleDict()
        self.yolo_blocks_names = []
        self.routes = nn.ModuleDict()
        self.routes_names = []
        for i, ch_in in enumerate(self.in_channels[::-1]):
            if i > 0:
                ch_in += 512 // 2 ** i
            channel = 64 * 2 ** self.num_blocks // 2 ** i
            base_cfg = []
            c_in, c_out = ch_in, channel
            for j in range(self.conv_block_num):
                base_cfg += [['conv{}'.format(2 * j), ConvLayer, [c_in, c_out, 1], dict(padding=0, norm_type=norm_type, freeze_norm=freeze_norm)], ['conv{}'.format(2 * j + 1), ConvBNLayer, [c_out, c_out * 2, 3], dict(padding=1, norm_type=norm_type, freeze_norm=freeze_norm)]]
                c_in, c_out = c_out * 2, c_out
            base_cfg += [['route', ConvLayer, [c_in, c_out, 1], dict(padding=0, norm_type=norm_type, freeze_norm=freeze_norm)], ['tip', ConvLayer, [c_out, c_out * 2, 3], dict(padding=1, norm_type=norm_type, freeze_norm=freeze_norm)]]
            if self.conv_block_num == 2:
                if i == 0:
                    if self.spp:
                        spp_cfg = [['spp', SPP, [channel * 4, channel, 1], dict(pool_size=[5, 9, 13], norm_type=norm_type, freeze_norm=freeze_norm)]]
                    else:
                        spp_cfg = []
                    cfg = base_cfg[0:3] + spp_cfg + base_cfg[3:4] + dropblock_cfg + base_cfg[4:6]
                else:
                    cfg = base_cfg[0:2] + dropblock_cfg + base_cfg[2:6]
            elif self.conv_block_num == 0:
                if self.spp and i == 0:
                    spp_cfg = [['spp', SPP, [c_in * 4, c_in, 1], dict(pool_size=[5, 9, 13], norm_type=norm_type, freeze_norm=freeze_norm)]]
                else:
                    spp_cfg = []
                cfg = spp_cfg + dropblock_cfg + base_cfg
            name = 'yolo_block_{}'.format(i)
            yolo_block = PPYOLODetBlock(cfg, name)
            self.yolo_blocks.add_module(name, yolo_block)
            self.yolo_blocks_names.append(name)
            self._out_channels.append(channel * 2)
            if i < self.num_blocks - 1:
                name = 'yolo_transition_{}'.format(i)
                route = ConvBNLayer(ch_in=channel, ch_out=256 // 2 ** i, filter_size=1, stride=1, padding=0, norm_type=norm_type, freeze_norm=freeze_norm, data_format=data_format, name=name)
                self.routes.add_module(name, route)
                self.routes_names.append(name)

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []
        if for_mot:
            emb_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = torch.cat([route, block], dim=1)
                else:
                    block = torch.cat([route, block], dim=-1)
            route, tip = self.yolo_blocks[self.yolo_blocks_names[i]](block)
            yolo_feats.append(tip)
            if for_mot:
                emb_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.routes[self.routes_names[i]](route)
                route = F.interpolate(route, scale_factor=2.0, data_format=self.data_format)
        if for_mot:
            return {'yolo_feats': yolo_feats, 'emb_feats': emb_feats}
        else:
            return yolo_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}


class PPYOLOTinyFPN(nn.Module):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self, in_channels=[80, 56, 34], detection_block_channels=[160, 128, 96], norm_type='bn', data_format='NCHW', **kwargs):
        """
        PPYOLO Tiny FPN layer
        Args:
            in_channels (list): input channels for fpn
            detection_block_channels (list): channels in fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            kwargs: extra key-value pairs, such as parameter of DropBlock and spp
        """
        super(PPYOLOTinyFPN, self).__init__()
        assert len(in_channels) > 0, 'in_channels length should > 0'
        self.in_channels = in_channels[::-1]
        assert len(detection_block_channels) > 0, 'detection_block_channelslength should > 0'
        self.detection_block_channels = detection_block_channels
        self.data_format = data_format
        self.num_blocks = len(in_channels)
        self.drop_block = kwargs.get('drop_block', False)
        self.block_size = kwargs.get('block_size', 3)
        self.keep_prob = kwargs.get('keep_prob', 1.0)
        self.spp_ = kwargs.get('spp', False)
        if self.spp_:
            self.spp = SPP(self.in_channels[0] * 4, self.in_channels[0], k=1, pool_size=[5, 9, 13], norm_type=norm_type, name='spp')
        self._out_channels = []
        self.yolo_blocks = nn.ModuleDict()
        self.yolo_blocks_names = []
        self.routes = nn.ModuleDict()
        self.routes_names = []
        for i, (ch_in, ch_out) in enumerate(zip(self.in_channels, self.detection_block_channels)):
            name = 'yolo_block_{}'.format(i)
            if i > 0:
                ch_in += self.detection_block_channels[i - 1]
            yolo_block = PPYOLOTinyDetBlock(ch_in, ch_out, name, drop_block=self.drop_block, block_size=self.block_size, keep_prob=self.keep_prob)
            self.yolo_blocks.add_module(name, yolo_block)
            self.yolo_blocks_names.append(name)
            self._out_channels.append(ch_out)
            if i < self.num_blocks - 1:
                name = 'yolo_transition_{}'.format(i)
                route = ConvBNLayer(ch_in=ch_out, ch_out=ch_out, filter_size=1, stride=1, padding=0, norm_type=norm_type, data_format=data_format, name=name)
                self.routes.add_module(name, route)
                self.routes_names.append(name)

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []
        if for_mot:
            emb_feats = []
        for i, block in enumerate(blocks):
            if i == 0 and self.spp_:
                block = self.spp(block)
            if i > 0:
                if self.data_format == 'NCHW':
                    block = torch.cat([route, block], dim=1)
                else:
                    block = torch.cat([route, block], dim=-1)
            route, tip = self.yolo_blocks[self.yolo_blocks_names[i]](block)
            yolo_feats.append(tip)
            if for_mot:
                emb_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.routes[self.routes_names[i]](route)
                route = F.interpolate(route, scale_factor=2.0)
        if for_mot:
            return {'yolo_feats': yolo_feats, 'emb_feats': emb_feats}
        else:
            return yolo_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}


class LayoutLMPooler(Layer):

    def __init__(self, hidden_size, pool_act='tanh'):
        super(LayoutLMPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == 'tanh':
            pooled_output = self.activation(pooled_output)
        return pooled_output


class LayoutLMEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, vocab_size, hidden_size=768, hidden_dropout_prob=0.1, max_position_embeddings=512, max_2d_position_embeddings=1024, layer_norm_eps=1e-12, pad_token_id=0, type_vocab_size=16):
        super(LayoutLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.h_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.w_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, bbox=None, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = torch.ones_like(input_ids, dtype=torch.long)
            seq_length = torch.cumsum(ones, dim=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError('The :obj:`bbox`coordinate values should be within 0-1000 range.') from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeddings + position_embeddings + left_position_embeddings + upper_position_embeddings + right_position_embeddings + lower_position_embeddings + h_position_embeddings + w_position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutXLMPooler(Layer):

    def __init__(self, hidden_size, with_pool):
        super(LayoutXLMPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.with_pool = with_pool

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.with_pool == 'tanh':
            pooled_output = self.activation(pooled_output)
        return pooled_output


class LayoutXLMEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super(LayoutXLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=0)
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.x_position_embeddings = nn.Embedding(config['max_2d_position_embeddings'], config['coordinate_size'])
        self.y_position_embeddings = nn.Embedding(config['max_2d_position_embeddings'], config['coordinate_size'])
        self.h_position_embeddings = nn.Embedding(config['max_2d_position_embeddings'], config['coordinate_size'])
        self.w_position_embeddings = nn.Embedding(config['max_2d_position_embeddings'], config['coordinate_size'])
        self.token_type_embeddings = nn.Embedding(config['type_vocab_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.register_buffer('position_ids', torch.arange(config['max_position_embeddings']).expand((1, -1)))

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError('The :obj:`bbox`coordinate values should be within 0-1000 range.') from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        spatial_position_embeddings = torch.cat([left_position_embeddings, upper_position_embeddings, right_position_embeddings, lower_position_embeddings, h_position_embeddings, w_position_embeddings], dim=-1)
        return spatial_position_embeddings

    def forward(self, input_ids, bbox=None, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = torch.ones_like(input_ids, dtype=torch.long)
            seq_length = torch.cumsum(ones, dim=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError('The :obj:`bbox`coordinate values should be within 0-1000 range.') from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = input_embedings + position_embeddings + left_position_embeddings + upper_position_embeddings + right_position_embeddings + lower_position_embeddings + h_position_embeddings + w_position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutXLMSelfOutput(Layer):

    def __init__(self, config):
        super(LayoutXLMSelfOutput, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutXLMSelfAttention(Layer):

    def __init__(self, config):
        super(LayoutXLMSelfAttention, self).__init__()
        if config['hidden_size'] % config['num_attention_heads'] != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size {} is not a multiple of the number of attention heads {}'.format(config['hidden_size'], config['num_attention_heads']))
        self.fast_qkv = config['fast_qkv']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / config['num_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.has_relative_attention_bias = config['has_relative_attention_bias']
        self.has_spatial_attention_bias = config['has_spatial_attention_bias']
        if config['fast_qkv']:
            self.qkv_linear = nn.Linear(config['hidden_size'], 3 * self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config['hidden_size'], self.all_head_size)
            self.key = nn.Linear(config['hidden_size'], self.all_head_size)
            self.value = nn.Linear(config['hidden_size'], self.all_head_size)
        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.reshape(_sz)
                v = v + self.v_bias.reshape(_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, rel_pos=None, rel_2d_pos=None):
        q, k, v = self.compute_qkv(hidden_states)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)
        query_layer = query_layer / math.sqrt(self.attention_head_size)
        attention_scores = torch.matmul(query_layer, key_layer.permute(0, 1, 3, 2))
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        attention_scores = attention_scores.float().masked_fill_(attention_mask, float('-inf'))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LayoutXLMAttention(Layer):

    def __init__(self, config):
        super(LayoutXLMAttention, self).__init__()
        self.self = LayoutXLMSelfAttention(config)
        self.output = LayoutXLMSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, rel_pos=None, rel_2d_pos=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class LayoutXLMIntermediate(Layer):

    def __init__(self, config):
        super(LayoutXLMIntermediate, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['intermediate_size'])
        if config['hidden_act'] == 'gelu':
            self.intermediate_act_fn = nn.GELU()
        else:
            assert False, 'hidden_act is set as: {}, please check it..'.format(config['hidden_act'])

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LayoutXLMOutput(Layer):

    def __init__(self, config):
        super(LayoutXLMOutput, self).__init__()
        self.dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutXLMLayer(Layer):

    def __init__(self, config):
        super(LayoutXLMLayer, self).__init__()
        self.seq_len_dim = 1
        self.attention = LayoutXLMAttention(config)
        self.add_cross_attention = False
        self.intermediate = LayoutXLMIntermediate(config)
        self.output = LayoutXLMOutput(config)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, rel_pos=None, rel_2d_pos=None):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs
        return outputs


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret


class LayoutXLMEncoder(Layer):

    def __init__(self, config):
        super(LayoutXLMEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutXLMLayer(config) for _ in range(config['num_hidden_layers'])])
        self.has_relative_attention_bias = config['has_relative_attention_bias']
        self.has_spatial_attention_bias = config['has_spatial_attention_bias']
        if self.has_relative_attention_bias:
            self.rel_pos_bins = config['rel_pos_bins']
            self.max_rel_pos = config['max_rel_pos']
            self.rel_pos_onehot_size = config['rel_pos_bins']
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config['num_attention_heads'], bias=False)
        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config['max_rel_2d_pos']
            self.rel_2d_pos_bins = config['rel_2d_pos_bins']
            self.rel_2d_pos_onehot_size = config['rel_2d_pos_bins']
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config['num_attention_heads'], bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config['num_attention_heads'], bias=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
        rel_pos = torch.nn.functional.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states.dtype)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2).contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(rel_pos_x_2d_mat, num_buckets=self.rel_2d_pos_bins, max_distance=self.max_rel_2d_pos)
        rel_pos_y = relative_position_bucket(rel_pos_y_2d_mat, num_buckets=self.rel_2d_pos_bins, max_distance=self.max_rel_2d_pos)
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states.dtype)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states.dtype)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2).contiguous()
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2).contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, output_attentions=False, output_hidden_states=False, bbox=None, position_ids=None):
        all_hidden_states = () if output_hidden_states else None
        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None
        hidden_save = dict()
        hidden_save['input_hidden_states'] = hidden_states
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_save['input_attention_mask'] = attention_mask
            hidden_save['input_layer_head_mask'] = layer_head_mask
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)
            hidden_states = layer_outputs[0]
            hidden_save['{}_data'.format(i)] = hidden_states
        return hidden_states,


class Conv_BN_ReLU(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.1)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.toplayer_ = Conv_BN_ReLU(in_channels[3], out_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer1_ = Conv_BN_ReLU(in_channels[2], out_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer2_ = Conv_BN_ReLU(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer3_ = Conv_BN_ReLU(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.smooth1_ = Conv_BN_ReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth2_ = Conv_BN_ReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth3_ = Conv_BN_ReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.out_channels = out_channels * 4
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _upsample(self, x, scale=1):
        return F.upsample(x, scale_factor=scale, mode='bilinear')

    def _upsample_add(self, x, y, scale=1):
        return F.upsample(x, scale_factor=scale, mode='bilinear') + y

    def forward(self, x):
        f2, f3, f4, f5 = x
        p5 = self.toplayer_(f5)
        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4, 2)
        p4 = self.smooth1_(p4)
        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3, 2)
        p3 = self.smooth2_(p3)
        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2, 2)
        p2 = self.smooth3_(p2)
        p3 = self._upsample(p3, 2)
        p4 = self._upsample(p4, 4)
        p5 = self._upsample(p5, 8)
        fuse = torch.cat([p2, p3, p4, p5], dim=1)
        return fuse


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super(LastLevelMaxPool, self).__init__()
        self.num_levels = 1
        self.in_feature = 'p5'

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class CNNBlockBase(Module):

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super(CNNBlockBase, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.stop_gradient = True


class FrozenBatchNorm(nn.BatchNorm2d):

    def __init__(self, num_channels):
        super(FrozenBatchNorm, self).__init__(num_channels)


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Layer.
        out_channels (int): out_channels
    Returns:
        nn.Layer or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': nn.BatchNorm2d, 'SyncBN': nn.SyncBatchNorm, 'FrozenBN': FrozenBatchNorm}[norm]
    return norm(out_channels)


class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """

    def __init__(self, in_channels=3, out_channels=64, norm='BN'):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super(BasicStem, self).__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, out_channels))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class DeformBottleneckBlock(CNNBlockBase):
    """
    Similar to :class:`BottleneckBlock`, but with :paper:`deformable conv <deformconv>`
    in the 3x3 convolution.
    """

    def __init__(self, in_channels, out_channels, *, bottleneck_channels, stride=1, num_groups=1, norm='BN', stride_in_1x1=False, dilation=1, deform_modulated=False, deform_num_groups=1):
        raise NotImplementedError


def build_resnet_backbone(cfg, input_shape=None):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    if input_shape is None:
        ch = 3
    else:
        ch = input_shape.channels
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(in_channels=ch, out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS, norm=norm)
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    assert res5_dilation in {1, 2}, 'res5_dilation cannot be {}.'.format(res5_dilation)
    num_blocks_per_stage = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3]}[depth]
    if depth in [18, 34]:
        assert out_channels == 64, 'Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34'
        assert not any(deform_on_per_stage), 'MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34'
        assert res5_dilation == 1, 'Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34'
        assert num_groups == 1, 'Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34'
    stages = []
    for idx, stage_idx in enumerate(range(2, 6)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or stage_idx == 5 and dilation == 2 else 2
        stage_kargs = {'num_blocks': num_blocks_per_stage[idx], 'stride_per_block': [first_stride] + [1] * (num_blocks_per_stage[idx] - 1), 'in_channels': in_channels, 'out_channels': out_channels, 'norm': norm}
        if depth in [18, 34]:
            stage_kargs['block_class'] = BasicBlock
        else:
            stage_kargs['bottleneck_channels'] = bottleneck_channels
            stage_kargs['stride_in_1x1'] = stride_in_1x1
            stage_kargs['dilation'] = dilation
            stage_kargs['num_groups'] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs['block_class'] = DeformBottleneckBlock
                stage_kargs['deform_modulated'] = deform_modulated
                stage_kargs['deform_num_groups'] = deform_num_groups
            else:
                stage_kargs['block_class'] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)


def build_resnet_fpn_backbone(cfg, input_shape=None):
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels, norm=cfg.MODEL.FPN.NORM, top_block=LastLevelMaxPool(), fuse_type=cfg.MODEL.FPN.FUSE_TYPE)
    return backbone


def read_config(fp=None):
    if fp is None:
        dir_name = os.path.dirname(os.path.abspath(__file__))
        fp = os.path.join(dir_name, 'visual_backbone.yaml')
    with open(fp, 'r') as fin:
        yacs_config = _yacs_config
        cfg = yacs_config.CfgNode().load_cfg(fin)
    cfg.freeze()
    return cfg


class VisualBackbone(Module):

    def __init__(self, config):
        super(VisualBackbone, self).__init__()
        self.cfg = read_config()
        self.backbone = build_resnet_fpn_backbone(self.cfg)
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer('pixel_mean', torch.as_tensor(self.cfg.MODEL.PIXEL_MEAN).reshape([num_channels, 1, 1]))
        self.register_buffer('pixel_std', torch.as_tensor(self.cfg.MODEL.PIXEL_STD).reshape([num_channels, 1, 1]))
        self.out_feature_key = 'p2'
        self.pool = nn.AdaptiveAvgPool2d(config['image_feature_pool_shape'][:2])
        if len(config['image_feature_pool_shape']) == 2:
            config['image_feature_pool_shape'].append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config['image_feature_pool_shape'][2]

    def forward(self, images):
        images_input = (torch.as_tensor(images) - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_dim=2).transpose([0, 2, 1])
        return features


class BiaffineAttention(Layer):
    """Implements a biaffine attention operator for binary relation classification."""

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bilinear = nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = nn.Linear(2 * in_features, out_features)

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))


class REDecoder(Layer):

    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1):
        super(REDecoder, self).__init__()
        self.entity_emb = nn.Embedding(3, hidden_size)
        projection = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(hidden_dropout_prob), nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(hidden_dropout_prob))
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(hidden_size // 2, 2)

    def build_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]['start']) <= 2:
                entities[b] = {'end': [1, 1], 'label': [0, 0], 'start': [0, 0]}
            all_possible_relations = set([(i, j) for i in range(len(entities[b]['label'])) for j in range(len(entities[b]['label'])) if entities[b]['label'][i] == 1 and entities[b]['label'][j] == 2])
            if len(all_possible_relations) == 0:
                all_possible_relations = {(0, 1)}
            positive_relations = set(list(zip(relations[b]['head'], relations[b]['tail'])))
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)
            relation_per_doc = {'head': [i[0] for i in reordered_relations], 'tail': [i[1] for i in reordered_relations], 'label': [1] * len(positive_relations) + [0] * (len(reordered_relations) - len(positive_relations))}
            assert len(relation_per_doc['head']) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel['head_id'] = relations['head'][i]
            rel['head'] = entities['start'][rel['head_id']], entities['end'][rel['head_id']]
            rel['head_type'] = entities['label'][rel['head_id']]
            rel['tail_id'] = relations['tail'][i]
            rel['tail'] = entities['start'][rel['tail_id']], entities['end'][rel['tail_id']]
            rel['tail_type'] = entities['label'][rel['tail_id']]
            rel['type'] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.shape
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = torch.as_tensor(relations[b]['head'])
            tail_entities = torch.as_tensor(relations[b]['tail'])
            relation_labels = torch.as_tensor(relations[b]['label'], dtype=torch.long)
            entities_start_index = torch.as_tensor(entities[b]['start'])
            entities_labels = torch.as_tensor(entities[b]['label'])
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)
            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)
            tmp_hidden_states = hidden_states[b][head_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = torch.unsqueeze(tmp_hidden_states, dim=0)
            head_repr = torch.cat((tmp_hidden_states, head_label_repr), dim=-1)
            tmp_hidden_states = hidden_states[b][tail_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = torch.unsqueeze(tmp_hidden_states, dim=0)
            tail_repr = torch.cat((tmp_hidden_states, tail_label_repr), dim=-1)
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss = None
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
        return loss, all_pred_relations


class Conv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop('norm', None)
        activation = kwargs.pop('activation', None)
        super(Conv2d, self).__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super(Conv2d, self).forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ShapeSpec(namedtuple('_ShapeSpec', ['channels', 'height', 'width', 'stride'])):

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()

    @abstractmethod
    def forward(self, *args):
        pass

    @property
    def size_divisibility(self) ->int:
        return 0

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}


COMMUNITY_MODEL_PREFIX = 'https://bj.bcebos.com/paddlenlp/models/transformers/community/'


class InitTrackerMeta(type(Layer)):
    """
    This metaclass wraps the `__init__` method of a class to add `init_config`
    attribute for instances of that class, and `init_config` use a dict to track
    the initial configuration. If the class has `_wrap_init` method, it would be
    hooked after `__init__` and called as `_wrap_init(self, init_fn, init_args)`.
    Since InitTrackerMeta would be used as metaclass for pretrained model classes,
    which always are Layer and `type(Layer)` is not `type`, thus use `type(Layer)`
    rather than `type` as base class for it to avoid inheritance metaclass
    conflicts.
    """

    def __init__(cls, name, bases, attrs):
        init_func = cls.__init__
        help_func = getattr(cls, '_wrap_init', None) if '__init__' in attrs else None
        cls.__init__ = InitTrackerMeta.init_and_track_conf(init_func, help_func)
        super(InitTrackerMeta, cls).__init__(name, bases, attrs)

    @staticmethod
    def init_and_track_conf(init_func, help_func=None):
        """
        wraps `init_func` which is `__init__` method of a class to add `init_config`
        attribute for instances of that class.
        Args:
            init_func (callable): It should be the `__init__` method of a class.
            help_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `_wrap_init(self, init_func, *init_args, **init_args)`.
                Default None.
        
        Returns:
            function: the wrapped function
        """

        @functools.wraps(init_func)
        def __impl__(self, *args, **kwargs):
            init_func(self, *args, **kwargs)
            if help_func:
                help_func(self, init_func, *args, **kwargs)
            self.init_config = kwargs
            if args:
                kwargs['init_args'] = args
            kwargs['init_class'] = self.__class__.__name__
        return __impl__


MODEL_HOME = '/root/.paddlenlp/models'


def fn_args_to_dict(func, *args, **kwargs):
    """
    Inspect function `func` and its arguments for running, and extract a
    dict mapping between argument names and keys. 
    """
    if hasattr(inspect, 'getfullargspec'):
        spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _ = inspect.getfullargspec(func)
    else:
        spec_args, spec_varargs, spec_varkw, spec_defaults = inspect.getargspec(func)
    init_dict = dict(zip(spec_args, args))
    kwargs_dict = dict(zip(spec_args[-len(spec_defaults):], spec_defaults)) if spec_defaults else {}
    kwargs_dict.update(kwargs)
    init_dict.update(kwargs_dict)
    return init_dict


def _load_paddle_layoutxlm_weights(torch_model, weights_path):
    with paddle.fluid.dygraph.guard():
        load_layer_state_dict, opti_state_dict = paddle.fluid.load_dygraph(weights_path)
    load_layer = []
    not_load_layer = []
    torch_state_dict = torch_model.state_dict()
    for k, v in load_layer_state_dict.items():
        ppname = name = k
        if ppname.endswith('._mean'):
            name = ppname.replace('._mean', '.running_mean')
        if ppname.endswith('._variance'):
            name = ppname.replace('._variance', '.running_var')
        load_layer.append(name)
        cur_weights = torch_state_dict[name]
        cur_w_shape = cur_weights.shape
        if ppname.endswith('.weight'):
            if len(v.shape) == len(cur_w_shape) == 2 and v.shape[0] == cur_w_shape[1] and v.shape[1] == cur_w_shape[0]:
                if ppname.startswith('layoutxlm.embeddings.'):
                    torch_state_dict[name].copy_(torch.Tensor(v))
                else:
                    torch_state_dict[name].copy_(torch.Tensor(v.T))
            else:
                torch_state_dict[name].copy_(torch.Tensor(v))
        else:
            torch_state_dict[name].copy_(torch.Tensor(v))
    None


def _load_torch_weights(torch_model, weights_path):
    torch_model.load_state_dict(torch.load(weights_path))
    None


def load_layoutxlm_weights(torch_model, weights_path):
    if weights_path.endswith('.pdparams'):
        _load_paddle_layoutxlm_weights(torch_model, weights_path)
    else:
        _load_torch_weights(torch_model, weights_path)


logger_initialized = {}


@functools.lru_cache()
def get_logger(name='root', log_file=None, log_level=logging.DEBUG):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger_initialized[name] = True
    return logger


logger = get_logger()


def build_backbone(config, model_type):
    if model_type == 'det':
        support_dict = ['MobileNetV3', 'ResNet', 'ResNet_vd', 'ResNet_SAST']
    elif model_type == 'rec' or model_type == 'cls':
        support_dict = ['MobileNetV1Enhance', 'MobileNetV3', 'ResNet', 'ResNetFPN', 'MTB', 'ResNet31', 'SVTRNet', 'ViTSTR']
    elif model_type == 'e2e':
        support_dict = ['ResNet']
    elif model_type == 'table':
        support_dict = ['ResNet', 'MobileNetV3']
    else:
        raise NotImplementedError
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('when model typs is {}, backbone only support {}'.format(model_type, support_dict))
    module_class = eval(module_name)(**config)
    return module_class


def build_head(config, **kwargs):
    support_dict = ['DBHead', 'PSEHead', 'EASTHead', 'SASTHead', 'CTCHead', 'ClsHead', 'AttentionHead', 'SRNHead', 'PGHead', 'Transformer', 'TableAttentionHead', 'SARHead', 'FCEHead']
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(support_dict))
    None
    module_class = eval(module_name)(**config, **kwargs)
    return module_class


def build_neck(config):
    support_dict = ['FPN', 'DBFPN', 'EASTFPN', 'SASTFPN', 'SequenceEncoder', 'PGFPN', 'TableFPN', 'RSEFPN', 'LKPAN', 'FCEFPN']
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('neck only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


def build_transform(config):
    support_dict = ['TPS', 'STN_ON']
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('transform only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


class BaseModel(nn.Module):

    def __init__(self, config, **kwargs):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()
        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels
        config['Backbone']['in_channels'] = in_channels
        self.backbone = build_backbone(config['Backbone'], model_type)
        in_channels = self.backbone.out_channels
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config['Head']['in_channels'] = in_channels
            self.head = build_head(config['Head'], **kwargs)
        self.return_all_feats = config.get('return_all_feats', False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        y = dict()
        if self.use_transform:
            x = self.transform(x)
        x = self.backbone(x)
        y['backbone_out'] = x
        if self.use_neck:
            x = self.neck(x)
        y['neck_out'] = x
        if self.use_head:
            x = self.head(x)
        if isinstance(x, dict) and 'ctc_nect' in x.keys():
            y['neck_out'] = x['ctc_neck']
            y['head_out'] = x
        elif isinstance(x, dict):
            y.update(x)
        else:
            y['head_out'] = x
        if self.return_all_feats:
            if self.training:
                return y
            else:
                return {'head_out': y['head_out']}
        else:
            return x


class ResNet_vd(nn.Module):

    def __init__(self, in_channels=3, layers=50, dcn_stage=None, out_indices=None, **kwargs):
        super(ResNet_vd, self).__init__()
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, 'supported layers are {} but input layer is {}'.format(supported_layers, layers)
        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]
        self.dcn_stage = dcn_stage if dcn_stage is not None else [False, False, False, False]
        self.out_indices = out_indices if out_indices is not None else [0, 1, 2, 3]
        self.conv1_1 = ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, act='relu', name='conv1_1')
        self.conv1_2 = ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu', name='conv1_2')
        self.conv1_3 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu', name='conv1_3')
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        self.out_channels = []
        if layers >= 50:
            for block in range(len(depth)):
                block_list = nn.Sequential()
                shortcut = False
                is_dcn = self.dcn_stage[block]
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = 'res' + str(block + 2) + 'a'
                        else:
                            conv_name = 'res' + str(block + 2) + 'b' + str(i)
                    else:
                        conv_name = 'res' + str(block + 2) + chr(97 + i)
                    bottleneck_block = BottleneckBlock(in_channels=num_channels[block] if i == 0 else num_filters[block] * 4, out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, shortcut=shortcut, if_first=block == i == 0, name=conv_name, is_dcn=is_dcn)
                    shortcut = True
                    block_list.add_module('bb_%d_%d' % (block, i), bottleneck_block)
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block] * 4)
                self.stages.append(block_list)
        else:
            for block in range(len(depth)):
                block_list = nn.Sequential()
                shortcut = False
                for i in range(depth[block]):
                    conv_name = 'res' + str(block + 2) + chr(97 + i)
                    basic_block = BasicBlock(in_channels=num_channels[block] if i == 0 else num_filters[block], out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, shortcut=shortcut, if_first=block == i == 0, name=conv_name)
                    shortcut = True
                    block_list.add_module('bb_%d_%d' % (block, i), basic_block)
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block])
                self.stages.append(block_list)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        out = []
        for i, block in enumerate(self.stages):
            y = block(y)
            if i in self.out_indices:
                out.append(y)
        return out


class ResNet_SAST(nn.Module):

    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(ResNet_SAST, self).__init__()
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, 'supported layers are {} but input layer is {}'.format(supported_layers, layers)
        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024, 2048] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512, 512]
        self.conv1_1 = ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, act='relu', name='conv1_1')
        self.conv1_2 = ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu', name='conv1_2')
        self.conv1_3 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu', name='conv1_3')
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        self.out_channels = [3, 64]
        if layers >= 50:
            for block in range(len(depth)):
                block_list = nn.Sequential()
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = 'res' + str(block + 2) + 'a'
                        else:
                            conv_name = 'res' + str(block + 2) + 'b' + str(i)
                    else:
                        conv_name = 'res' + str(block + 2) + chr(97 + i)
                    bottleneck_block = BottleneckBlock(in_channels=num_channels[block] if i == 0 else num_filters[block] * 4, out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, shortcut=shortcut, if_first=block == i == 0, name=conv_name)
                    shortcut = True
                    block_list.add_module('bb_%d_%d' % (block, i), bottleneck_block)
                self.out_channels.append(num_filters[block] * 4)
                self.stages.append(block_list)
        else:
            for block in range(len(depth)):
                block_list = nn.Sequential()
                shortcut = False
                for i in range(depth[block]):
                    conv_name = 'res' + str(block + 2) + chr(97 + i)
                    basic_block = BasicBlock(in_channels=num_channels[block] if i == 0 else num_filters[block], out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, shortcut=shortcut, if_first=block == i == 0, name=conv_name)
                    shortcut = True
                    block_list.add_module('bb_%d_%d' % (block, i), basic_block)
                self.out_channels.append(num_filters[block])
                self.stages.append(block_list)

    def forward(self, inputs):
        out = [inputs]
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        out.append(y)
        y = self.pool2d_max(y)
        for block in self.stages:
            y = block(y)
            out.append(y)
        return out


class DepthwiseSeparable(nn.Module):

    def __init__(self, num_channels, num_filters1, num_filters2, num_groups, stride, scale, dw_size=3, padding=1, use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self._depthwise_conv = ConvBNLayer(num_channels=num_channels, num_filters=int(num_filters1 * scale), filter_size=dw_size, stride=stride, padding=padding, num_groups=int(num_groups * scale))
        if use_se:
            self._se = SEModule(int(num_filters1 * scale))
        self._pointwise_conv = ConvBNLayer(num_channels=int(num_filters1 * scale), filter_size=1, num_filters=int(num_filters2 * scale), stride=1, padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        if self.use_se:
            y = self._se(y)
        y = self._pointwise_conv(y)
        return y


class MobileNetV1Enhance(nn.Module):

    def __init__(self, in_channels=3, scale=0.5, last_conv_stride=1, last_pool_type='max', **kwargs):
        super().__init__()
        self.scale = scale
        self.block_list = []
        self.conv1 = ConvBNLayer(num_channels=in_channels, filter_size=3, channels=3, num_filters=int(32 * scale), stride=2, padding=1)
        conv2_1 = DepthwiseSeparable(num_channels=int(32 * scale), num_filters1=32, num_filters2=64, num_groups=32, stride=1, scale=scale)
        self.block_list.append(conv2_1)
        conv2_2 = DepthwiseSeparable(num_channels=int(64 * scale), num_filters1=64, num_filters2=128, num_groups=64, stride=1, scale=scale)
        self.block_list.append(conv2_2)
        conv3_1 = DepthwiseSeparable(num_channels=int(128 * scale), num_filters1=128, num_filters2=128, num_groups=128, stride=1, scale=scale)
        self.block_list.append(conv3_1)
        conv3_2 = DepthwiseSeparable(num_channels=int(128 * scale), num_filters1=128, num_filters2=256, num_groups=128, stride=(2, 1), scale=scale)
        self.block_list.append(conv3_2)
        conv4_1 = DepthwiseSeparable(num_channels=int(256 * scale), num_filters1=256, num_filters2=256, num_groups=256, stride=1, scale=scale)
        self.block_list.append(conv4_1)
        conv4_2 = DepthwiseSeparable(num_channels=int(256 * scale), num_filters1=256, num_filters2=512, num_groups=256, stride=(2, 1), scale=scale)
        self.block_list.append(conv4_2)
        for _ in range(5):
            conv5 = DepthwiseSeparable(num_channels=int(512 * scale), num_filters1=512, num_filters2=512, num_groups=512, stride=1, dw_size=5, padding=2, scale=scale, use_se=False)
            self.block_list.append(conv5)
        conv5_6 = DepthwiseSeparable(num_channels=int(512 * scale), num_filters1=512, num_filters2=1024, num_groups=512, stride=(2, 1), dw_size=5, padding=2, scale=scale, use_se=True)
        self.block_list.append(conv5_6)
        conv6 = DepthwiseSeparable(num_channels=int(1024 * scale), num_filters1=1024, num_filters2=1024, num_groups=1024, stride=last_conv_stride, dw_size=5, padding=2, use_se=True, scale=scale)
        self.block_list.append(conv6)
        self.block_list = nn.Sequential(*self.block_list)
        if last_pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y


class MTB(nn.Module):

    def __init__(self, cnn_num, in_channels):
        super(MTB, self).__init__()
        self.block = nn.Sequential()
        self.out_channels = in_channels
        self.cnn_num = cnn_num
        if self.cnn_num == 2:
            for i in range(self.cnn_num):
                self.block.add_module('conv_{}'.format(i), nn.Conv2d(in_channels=in_channels if i == 0 else 32 * 2 ** (i - 1), out_channels=32 * 2 ** i, kernel_size=3, stride=2, padding=1))
                self.block.add_module('relu_{}'.format(i), nn.ReLU())
                self.block.add_module('bn_{}'.format(i), nn.BatchNorm2d(32 * 2 ** i))

    def forward(self, images):
        x = self.block(images)
        if self.cnn_num == 2:
            x = x.permute(0, 3, 2, 1)
            x_shape = x.shape
            x = torch.reshape(x, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3]))
        return x


class ResNet31(nn.Module):
    """
    Args:
        in_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        out_indices (None | Sequence[int]): Indices of output stages.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    """

    def __init__(self, in_channels=3, layers=[1, 2, 5, 3], channels=[64, 128, 256, 256, 512, 512, 512], out_indices=None, last_stage_pool=False):
        super(ResNet31, self).__init__()
        assert isinstance(in_channels, int)
        assert isinstance(last_stage_pool, bool)
        self.out_indices = out_indices
        self.last_stage_pool = last_stage_pool
        self.conv1_1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(channels[0])
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(channels[1])
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block2 = self._make_layer(channels[1], channels[2], layers[0])
        self.conv2 = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block3 = self._make_layer(channels[2], channels[3], layers[1])
        self.conv3 = nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.relu3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=True)
        self.block4 = self._make_layer(channels[3], channels[4], layers[2])
        self.conv4 = nn.Conv2d(channels[4], channels[4], kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(channels[4])
        self.relu4 = nn.ReLU(inplace=True)
        self.pool5 = None
        if self.last_stage_pool:
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block5 = self._make_layer(channels[4], channels[5], layers[3])
        self.conv5 = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(channels[5])
        self.relu5 = nn.ReLU(inplace=True)
        self.out_channels = channels[-1]

    def _make_layer(self, input_channels, output_channels, blocks):
        layers = []
        for _ in range(blocks):
            downsample = None
            if input_channels != output_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(output_channels))
            layers.append(BasicBlock(input_channels, output_channels, downsample=downsample))
            input_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        outs = []
        for i in range(4):
            layer_index = i + 2
            pool_layer = getattr(self, 'pool{}'.format(layer_index))
            block_layer = getattr(self, 'block{}'.format(layer_index))
            conv_layer = getattr(self, 'conv{}'.format(layer_index))
            bn_layer = getattr(self, 'bn{}'.format(layer_index))
            relu_layer = getattr(self, 'relu{}'.format(layer_index))
            if pool_layer is not None:
                x = pool_layer(x)
            x = block_layer(x)
            x = conv_layer(x)
            x = bn_layer(x)
            x = relu_layer(x)
            outs.append(x)
        if self.out_indices is not None:
            return tuple([outs[i] for i in self.out_indices])
        return x


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = torch.as_tensor(1 - drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='gelu', drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = Activation(act_type=act_layer, inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMixer(nn.Module):

    def __init__(self, dim, num_heads=8, HW=[8, 25], local_k=[3, 3]):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(dim, dim, local_k, 1, [local_k[0] // 2, local_k[1] // 2], groups=num_heads)

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, mixer='Global', HW=[8, 25], local_k=[7, 11], qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(H * W, H + hk - 1, W + wk - 1, dtype=torch.float32)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.0
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
            mask_inf = torch.full([H * W, H * W], fill_value=float('-Inf'), dtype=torch.float32)
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze(0).unsqueeze(1)
        self.mixer = mixer

    def forward(self, x):
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q.matmul(k.permute(0, 1, 3, 2))
        if self.mixer == 'Local':
            attn += self.mask
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = attn.matmul(v).permute(0, 2, 1, 3).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mixer='Global', local_mixer=[7, 11], HW=None, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer='gelu', norm_layer='nn.LayerNorm', epsilon=1e-06, prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(dim, num_heads=num_heads, mixer=mixer, HW=HW, local_k=local_mixer, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=[32, 100], in_channels=3, embed_dim=768, sub_num=2, patch_size=[4, 4], mode='pope'):
        super().__init__()
        num_patches = img_size[1] // 2 ** sub_num * (img_size[0] // 2 ** sub_num)
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.Sequential(ConvBNLayer(in_channels=in_channels, out_channels=embed_dim // 2, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True), ConvBNLayer(in_channels=embed_dim // 2, out_channels=embed_dim, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True))
            if sub_num == 3:
                self.proj = nn.Sequential(ConvBNLayer(in_channels=in_channels, out_channels=embed_dim // 4, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True), ConvBNLayer(in_channels=embed_dim // 4, out_channels=embed_dim // 2, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True), ConvBNLayer(in_channels=embed_dim // 2, out_channels=embed_dim, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True))
        elif mode == 'linear':
            self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.num_patches = img_size[0] // patch_size[0] * img_size[1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], "Input image size ({}*{}) doesn't match model ({}*{}).".format(H, W, self.img_size[0], self.img_size[1])
        x = self.proj(x).flatten(2).permute(0, 2, 1)
        return x


class SubSample(nn.Module):

    def __init__(self, in_channels, out_channels, types='Pool', stride=[2, 1], sub_norm='nn.LayerNorm', act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))
        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class SVTRNet(nn.Module):

    def __init__(self, img_size=[32, 100], in_channels=3, embed_dim=[64, 128, 256], depth=[3, 6, 3], num_heads=[2, 4, 8], mixer=['Local'] * 6 + ['Global'] * 6, local_mixer=[[7, 11], [7, 11], [7, 11]], patch_merging='Conv', mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0, last_drop=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer='nn.LayerNorm', sub_norm='nn.LayerNorm', epsilon=1e-06, out_channels=192, out_char_num=25, block_unit='Block', act='gelu', last_stage=True, sub_num=2, prenorm=True, use_lenhead=False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(img_size=img_size, in_channels=in_channels, embed_dim=embed_dim[0], sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // 2 ** sub_num, img_size[1] // 2 ** sub_num]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)
        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.ModuleList([Block_unit(dim=embed_dim[0], num_heads=num_heads[0], mixer=mixer[0:depth[0]][i], HW=self.HW, local_mixer=local_mixer[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer=act, attn_drop=attn_drop_rate, drop_path=dpr[0:depth[0]][i], norm_layer=norm_layer, epsilon=epsilon, prenorm=prenorm) for i in range(depth[0])])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(embed_dim[0], embed_dim[1], sub_norm=sub_norm, stride=[2, 1], types=patch_merging)
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList([Block_unit(dim=embed_dim[1], num_heads=num_heads[1], mixer=mixer[depth[0]:depth[0] + depth[1]][i], HW=HW, local_mixer=local_mixer[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer=act, attn_drop=attn_drop_rate, drop_path=dpr[depth[0]:depth[0] + depth[1]][i], norm_layer=norm_layer, epsilon=epsilon, prenorm=prenorm) for i in range(depth[1])])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(embed_dim[1], embed_dim[2], sub_norm=sub_norm, stride=[2, 1], types=patch_merging)
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks3 = nn.ModuleList([Block_unit(dim=embed_dim[2], num_heads=num_heads[2], mixer=mixer[depth[0] + depth[1]:][i], HW=HW, local_mixer=local_mixer[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer=act, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1]:][i], norm_layer=norm_layer, epsilon=epsilon, prenorm=prenorm) for i in range(depth[2])])
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            self.last_conv = nn.Conv2d(in_channels=embed_dim[2], out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.hardswish = Activation('hard_swish', inplace=True)
            self.dropout = nn.Dropout(p=last_drop)
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], eps=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = Activation('hard_swish', inplace=True)
            self.dropout_len = nn.Dropout(p=last_drop)
        torch.nn.init.xavier_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(x.permute(0, 2, 1).reshape([-1, self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(x.permute(0, 2, 1).reshape([-1, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = self.avg_pool(x.permute(0, 2, 1).reshape([-1, self.embed_dim[2], h, self.HW[1]]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        if self.use_lenhead:
            return x, len_x
        return x


scale_dim_heads = {'tiny': [192, 3], 'small': [384, 6], 'base': [768, 12]}


class ViTSTR(nn.Module):

    def __init__(self, img_size=[224, 224], in_channels=1, scale='tiny', seqlen=27, patch_size=[16, 16], embed_dim=None, depth=12, num_heads=None, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_path_rate=0.0, drop_rate=0.0, attn_drop_rate=0.0, norm_layer='nn.LayerNorm', act_layer='gelu', epsilon=1e-06, out_channels=None, **kwargs):
        super().__init__()
        self.seqlen = seqlen
        embed_dim = embed_dim if embed_dim is not None else scale_dim_heads[scale][0]
        num_heads = num_heads if num_heads is not None else scale_dim_heads[scale][1]
        out_channels = out_channels if out_channels is not None else embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size, mode='linear')
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, epsilon=epsilon, prenorm=False) for i in range(depth)])
        self.norm = eval(norm_layer)(embed_dim, eps=epsilon)
        self.out_channels = out_channels
        torch.nn.init.xavier_normal_(self.pos_embed)
        torch.nn.init.xavier_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, :self.seqlen]
        return x.permute(0, 2, 1).unsqueeze(2)


class ClsHead(nn.Module):
    """
    Class orientation
    Args:
        params(dict): super parameters for build Class network
    """

    def __init__(self, in_channels, class_dim, **kwargs):
        super(ClsHead, self).__init__()
        self.training = False
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, class_dim, bias=True)

    def forward(self, x):
        x = self.pool(x)
        x = torch.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x


class Head(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=3, padding=1, bias=False)
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = Activation(act_type='relu')
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=2, stride=2)
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = Activation(act_type='relu')
        self.conv3 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=1, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        binarize_name_list = ['conv2d_56', 'batch_norm_47', 'conv2d_transpose_0', 'batch_norm_48', 'conv2d_transpose_1', 'binarize']
        thresh_name_list = ['conv2d_57', 'batch_norm_49', 'conv2d_transpose_2', 'batch_norm_50', 'conv2d_transpose_3', 'thresh']
        self.binarize = Head(in_channels)
        self.thresh = Head(in_channels)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'maps': shrink_maps}
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
        return {'maps': y}


class EASTHead(nn.Module):
    """
    """

    def __init__(self, in_channels, model_name, **kwargs):
        super(EASTHead, self).__init__()
        self.model_name = model_name
        if self.model_name == 'large':
            num_outputs = [128, 64, 1, 8]
        else:
            num_outputs = [64, 32, 1, 8]
        self.det_conv1 = ConvBNLayer(in_channels=in_channels, out_channels=num_outputs[0], kernel_size=3, stride=1, padding=1, if_act=True, act='relu', name='det_head1')
        self.det_conv2 = ConvBNLayer(in_channels=num_outputs[0], out_channels=num_outputs[1], kernel_size=3, stride=1, padding=1, if_act=True, act='relu', name='det_head2')
        self.score_conv = ConvBNLayer(in_channels=num_outputs[1], out_channels=num_outputs[2], kernel_size=1, stride=1, padding=0, if_act=False, act=None, name='f_score')
        self.geo_conv = ConvBNLayer(in_channels=num_outputs[1], out_channels=num_outputs[3], kernel_size=1, stride=1, padding=0, if_act=False, act=None, name='f_geo')

    def forward(self, x):
        f_det = self.det_conv1(x)
        f_det = self.det_conv2(f_det)
        f_score = self.score_conv(f_det)
        f_score = torch.sigmoid(f_score)
        f_geo = self.geo_conv(f_det)
        f_geo = (torch.sigmoid(f_geo) - 0.5) * 2 * 800
        pred = {'f_score': f_score, 'f_geo': f_geo}
        return pred


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class FCEHead(nn.Module):
    """The class for implementing FCENet head.
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
    Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        in_channels (int): The number of input channels.
        scales (list[int]) : The scale of each layer.
        fourier_degree (int) : The maximum Fourier transform degree k.
    """

    def __init__(self, in_channels, fourier_degree=5):
        super().__init__()
        assert isinstance(in_channels, int)
        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.fourier_degree = fourier_degree
        self.out_channels_cls = 4
        self.out_channels_reg = (2 * self.fourier_degree + 1) * 2
        self.out_conv_cls = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels_cls, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.out_conv_reg = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels_reg, kernel_size=3, stride=1, padding=1, groups=1, bias=True)

    def forward(self, feats, targets=None):
        cls_res, reg_res = multi_apply(self.forward_single, feats)
        level_num = len(cls_res)
        outs = {}
        if not self.training:
            for i in range(level_num):
                tr_pred = F.softmax(cls_res[i][:, 0:2, :, :], dim=1)
                tcl_pred = F.softmax(cls_res[i][:, 2:, :, :], dim=1)
                outs['level_{}'.format(i)] = torch.cat([tr_pred, tcl_pred, reg_res[i]], dim=1)
        else:
            preds = [[cls_res[i], reg_res[i]] for i in range(level_num)]
            outs['levels'] = preds
        return outs

    def forward_single(self, x):
        cls_predict = self.out_conv_cls(x)
        reg_predict = self.out_conv_reg(x)
        return cls_predict, reg_predict


class PSEHead(nn.Module):

    def __init__(self, in_channels, hidden_dim=256, out_channels=7, **kwargs):
        super(PSEHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)
        return {'maps': out}


class SAST_Header1(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(SAST_Header1, self).__init__()
        out_channels = [64, 64, 128]
        self.score_conv = nn.Sequential(ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_score1'), ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_score2'), ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_score3'), ConvBNLayer(out_channels[2], 1, 3, 1, act=None, name='f_score4'))
        self.border_conv = nn.Sequential(ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_border1'), ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_border2'), ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_border3'), ConvBNLayer(out_channels[2], 4, 3, 1, act=None, name='f_border4'))

    def forward(self, x):
        f_score = self.score_conv(x)
        f_score = torch.sigmoid(f_score)
        f_border = self.border_conv(x)
        return f_score, f_border


class SAST_Header2(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(SAST_Header2, self).__init__()
        out_channels = [64, 64, 128]
        self.tvo_conv = nn.Sequential(ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_tvo1'), ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_tvo2'), ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_tvo3'), ConvBNLayer(out_channels[2], 8, 3, 1, act=None, name='f_tvo4'))
        self.tco_conv = nn.Sequential(ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_tco1'), ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_tco2'), ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_tco3'), ConvBNLayer(out_channels[2], 2, 3, 1, act=None, name='f_tco4'))

    def forward(self, x):
        f_tvo = self.tvo_conv(x)
        f_tco = self.tco_conv(x)
        return f_tvo, f_tco


class SASTHead(nn.Module):
    """
    """

    def __init__(self, in_channels, **kwargs):
        super(SASTHead, self).__init__()
        self.head1 = SAST_Header1(in_channels)
        self.head2 = SAST_Header2(in_channels)

    def forward(self, x):
        f_score, f_border = self.head1(x)
        f_tvo, f_tco = self.head2(x)
        predicts = {}
        predicts['f_score'] = f_score
        predicts['f_border'] = f_border
        predicts['f_tvo'] = f_tvo
        predicts['f_tco'] = f_tco
        return predicts


class PGHead(nn.Module):
    """
    """

    def __init__(self, in_channels, **kwargs):
        super(PGHead, self).__init__()
        self.conv_f_score1 = ConvBNLayer(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0, act='relu', name='conv_f_score{}'.format(1))
        self.conv_f_score2 = ConvBNLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, act='relu', name='conv_f_score{}'.format(2))
        self.conv_f_score3 = ConvBNLayer(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, act='relu', name='conv_f_score{}'.format(3))
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.conv_f_boder1 = ConvBNLayer(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0, act='relu', name='conv_f_boder{}'.format(1))
        self.conv_f_boder2 = ConvBNLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, act='relu', name='conv_f_boder{}'.format(2))
        self.conv_f_boder3 = ConvBNLayer(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, act='relu', name='conv_f_boder{}'.format(3))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.conv_f_char1 = ConvBNLayer(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0, act='relu', name='conv_f_char{}'.format(1))
        self.conv_f_char2 = ConvBNLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, act='relu', name='conv_f_char{}'.format(2))
        self.conv_f_char3 = ConvBNLayer(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, act='relu', name='conv_f_char{}'.format(3))
        self.conv_f_char4 = ConvBNLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, act='relu', name='conv_f_char{}'.format(4))
        self.conv_f_char5 = ConvBNLayer(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, act='relu', name='conv_f_char{}'.format(5))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=37, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.conv_f_direc1 = ConvBNLayer(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0, act='relu', name='conv_f_direc{}'.format(1))
        self.conv_f_direc2 = ConvBNLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, act='relu', name='conv_f_direc{}'.format(2))
        self.conv_f_direc3 = ConvBNLayer(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, act='relu', name='conv_f_direc{}'.format(3))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, x):
        f_score = self.conv_f_score1(x)
        f_score = self.conv_f_score2(f_score)
        f_score = self.conv_f_score3(f_score)
        f_score = self.conv1(f_score)
        f_score = torch.sigmoid(f_score)
        f_border = self.conv_f_boder1(x)
        f_border = self.conv_f_boder2(f_border)
        f_border = self.conv_f_boder3(f_border)
        f_border = self.conv2(f_border)
        f_char = self.conv_f_char1(x)
        f_char = self.conv_f_char2(f_char)
        f_char = self.conv_f_char3(f_char)
        f_char = self.conv_f_char4(f_char)
        f_char = self.conv_f_char5(f_char)
        f_char = self.conv3(f_char)
        f_direction = self.conv_f_direc1(x)
        f_direction = self.conv_f_direc2(f_direction)
        f_direction = self.conv_f_direc3(f_direction)
        f_direction = self.conv4(f_direction)
        predicts = {}
        predicts['f_score'] = f_score
        predicts['f_border'] = f_border
        predicts['f_char'] = f_char
        predicts['f_direction'] = f_direction
        return predicts


class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        	ext{MultiHead}(Q, K, V) = 	ext{Concat}(head_1,\\dots,head_h)W^O
        	ext{where} head_i = 	ext{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()
        self.conv1 = torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv2 = torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv3 = torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))

    def _reset_parameters(self):
        xavier_uniform_(self.out_proj.weight)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, attn_mask=None):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """
        q_shape = query.shape
        src_shape = key.shape
        q = self._in_proj_q(query)
        k = self._in_proj_k(key)
        v = self._in_proj_v(value)
        q *= self.scaling
        q = torch.reshape(q, (q_shape[0], q_shape[1], self.num_heads, self.head_dim))
        q = q.permute(1, 2, 0, 3)
        k = torch.reshape(k, (src_shape[0], q_shape[1], self.num_heads, self.head_dim))
        k = k.permute(1, 2, 0, 3)
        v = torch.reshape(v, (src_shape[0], q_shape[1], self.num_heads, self.head_dim))
        v = v.permute(1, 2, 0, 3)
        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == q_shape[1]
            assert key_padding_mask.shape[1] == src_shape[0]
        attn_output_weights = torch.matmul(q, k.permute(0, 1, 3, 2))
        if attn_mask is not None:
            attn_mask = torch.unsqueeze(torch.unsqueeze(attn_mask, 0), 0)
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = torch.reshape(attn_output_weights, [q_shape[1], self.num_heads, q_shape[0], src_shape[0]])
            key = torch.unsqueeze(torch.unsqueeze(key_padding_mask, 1), 2)
            key = key.type(torch.float32)
            y = torch.full(size=key.shape, fill_value=float('-Inf'), dtype=torch.float32)
            y = torch.where(key == 0.0, key, y)
            attn_output_weights += y
        attn_output_weights = F.softmax(attn_output_weights.type(torch.float32), dim=-1, dtype=torch.float32 if attn_output_weights.dtype == torch.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, v)
        attn_output = torch.reshape(attn_output.permute(2, 0, 1, 3), [q_shape[0], q_shape[1], self.embed_dim])
        attn_output = self.out_proj(attn_output)
        return attn_output

    def _in_proj_q(self, query):
        query = query.permute(1, 2, 0)
        query = torch.unsqueeze(query, dim=2)
        res = self.conv1(query)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res

    def _in_proj_k(self, key):
        key = key.permute(1, 2, 0)
        key = torch.unsqueeze(key, dim=2)
        res = self.conv2(key)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res

    def _in_proj_v(self, value):
        value = value.permute(1, 2, 0)
        value = torch.unsqueeze(value, dim=2)
        res = self.conv3(value)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res


class AttentionHead(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels
        self.attention_cell = AttentionGRUCell(in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char.type(torch.int64), onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.size()[0]
        num_steps = batch_max_length
        hidden = torch.zeros((batch_size, self.hidden_size))
        output_hiddens = []
        if targets is not None:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs, char_onehots)
                output_hiddens.append(torch.unsqueeze(outputs, dim=1))
            output = torch.cat(output_hiddens, dim=1)
            probs = self.generator(output)
        else:
            targets = torch.zeros([batch_size], dtype=torch.int32)
            probs = None
            char_onehots = None
            outputs = None
            alpha = None
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(outputs)
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.cat([probs, torch.unsqueeze(probs_step, dim=1)], dim=1)
                next_input = probs_step.argmax(dim=1)
                targets = next_input
        return probs


class CTCHead(nn.Module):

    def __init__(self, in_channels, out_channels=6625, fc_decay=0.0004, mid_channels=None, return_feats=False, **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels, bias=True)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels, bias=True)
            self.fc2 = nn.Linear(mid_channels, out_channels, bias=True)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, labels=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)
        if self.return_feats:
            result = x, predicts
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts
        return result


class Beam:
    """ Beam search """

    def __init__(self, size, device=False):
        self.size = size
        self._done = False
        self.scores = torch.zeros((size,), dtype=torch.float32)
        self.all_scores = []
        self.prev_ks = []
        self.next_ys = [torch.full((size,), 0, dtype=torch.int64)]
        self.next_ys[0][0] = 2

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        """Update beam status and check if finished or not."""
        num_words = word_prob.shape[1]
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]
        flat_beam_lk = beam_lk.reshape([-1])
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        if self.next_ys[-1][0] == 3:
            self._done = True
            self.all_scores.append(self.scores)
        return self._done

    def sort_scores(self):
        """Sort the scores."""
        return self.scores, torch.tensor([i for i in range(int(self.scores.shape[0]))], dtype=torch.int32)

    def get_the_best_score_and_idx(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """Get the decoded sequence for the current timestep."""
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [([2] + h) for h in hyps]
            dec_seq = torch.tensor(hyps, dtype=torch.int64)
        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return list(map(lambda x: x.item(), hyp[::-1]))


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab, padding_idx, scale_embedding):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        w0 = np.random.normal(0.0, d_model ** -0.5, (vocab, d_model)).astype(np.float32)
        self.embedding.weight.data = torch.from_numpy(w0)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        	ext{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        	ext{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        	ext{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).type(torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.unsqueeze(pe, 0)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


def _get_clones(module, N):
    return LayerList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):
    """TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        output = tgt
        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, attention_dropout_rate=0.0, residual_dropout_rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=(1, 1))
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(residual_dropout_rate)
        self.dropout2 = Dropout(residual_dropout_rate)
        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt = tgt.permute(1, 2, 0)
        tgt = torch.unsqueeze(tgt, 2)
        tgt2 = self.conv2(F.relu(self.conv1(tgt)))
        tgt2 = torch.squeeze(tgt2, 2)
        tgt2 = tgt2.permute(2, 0, 1)
        tgt = torch.squeeze(tgt, 2)
        tgt = tgt.permute(2, 0, 1)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        """Pass the input through the endocder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        output = src
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=None, src_key_padding_mask=None)
        return output


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, attention_dropout_rate=0.0, residual_dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=(1, 1))
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(residual_dropout_rate)
        self.dropout2 = Dropout(residual_dropout_rate)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the endocder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src.permute(1, 2, 0)
        src = torch.unsqueeze(src, 2)
        src2 = self.conv2(F.relu(self.conv1(src)))
        src2 = torch.squeeze(src2, 2)
        src2 = src2.permute(2, 0, 1)
        src = torch.squeeze(src, 2)
        src = src.permute(2, 0, 1)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Transformer(nn.Module):
    """A transformer model. User is able to modify the attributes as needed. The architechture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, beam_size=0, num_decoder_layers=6, dim_feedforward=1024, attention_dropout_rate=0.0, residual_dropout_rate=0.1, custom_encoder=None, custom_decoder=None, in_channels=0, out_channels=0, scale_embedding=True):
        super(Transformer, self).__init__()
        self.out_channels = out_channels
        self.embedding = Embeddings(d_model=d_model, vocab=self.out_channels, padding_idx=0, scale_embedding=scale_embedding)
        self.positional_encoding = PositionalEncoding(dropout=residual_dropout_rate, dim=d_model)
        if custom_encoder is not None:
            self.encoder = custom_encoder
        elif num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, attention_dropout_rate, residual_dropout_rate)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        else:
            self.encoder = None
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, attention_dropout_rate, residual_dropout_rate)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self._reset_parameters()
        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels, bias=False)
        w0 = np.random.normal(0.0, d_model ** -0.5, (self.out_channels, d_model)).astype(np.float32)
        self.tgt_word_prj.weight.data = torch.from_numpy(w0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]
        tgt_key_padding_mask = self.generate_padding_mask(tgt)
        tgt = self.embedding(tgt).permute(1, 0, 2)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[0])
        if self.encoder is not None:
            src = self.positional_encoding(src.permute(1, 0, 2))
            memory = self.encoder(src)
        else:
            memory = src.squeeze(2).permute(2, 0, 1)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=None)
        output = output.permute(1, 0, 2)
        logit = self.tgt_word_prj(output)
        return logit

    def forward(self, src, targets=None):
        """Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
        Examples:
            >>> output = transformer_model(src, tgt)
        """
        if self.training:
            max_len = targets[1].max()
            tgt = targets[0][:, :2 + max_len]
            return self.forward_train(src, tgt)
        elif self.beam_size > 0:
            return self.forward_beam(src)
        else:
            return self.forward_test(src)

    def forward_test(self, src):
        bs = src.shape[0]
        if self.encoder is not None:
            src = self.positional_encoding(src.permute(1, 0, 2))
            memory = self.encoder(src)
        else:
            memory = torch.squeeze(src, 2).permute(2, 0, 1)
        dec_seq = torch.full((bs, 1), 2, dtype=torch.int64)
        dec_prob = torch.full((bs, 1), 1.0, dtype=torch.float32)
        for len_dec_seq in range(1, 25):
            dec_seq_embed = self.embedding(dec_seq).permute(1, 0, 2)
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(dec_seq_embed.shape[0])
            output = self.decoder(dec_seq_embed, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
            dec_output = output.permute(1, 0, 2)
            dec_output = dec_output[:, -1, :]
            tgt_word_prj = self.tgt_word_prj(dec_output)
            word_prob = F.softmax(tgt_word_prj, dim=1)
            preds_idx = word_prob.argmax(dim=1)
            if torch.equal(preds_idx, torch.full(preds_idx.shape, 3, dtype=torch.int64)):
                break
            preds_prob = torch.max(word_prob, dim=1).values
            dec_seq = torch.cat([dec_seq, torch.reshape(preds_idx, (-1, 1))], dim=1)
            dec_prob = torch.cat([dec_prob, torch.reshape(preds_prob, (-1, 1))], dim=1)
        return [dec_seq, dec_prob]

    def forward_beam(self, images):
        """ Translation work in one batch """

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """ Indicate the position of an instance in a tensor. """
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            """ Collect tensor parts associated to active instances. """
            beamed_tensor_shape = beamed_tensor.shape
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = n_curr_active_inst * n_bm, beamed_tensor_shape[1], beamed_tensor_shape[2]
            beamed_tensor = beamed_tensor.reshape([n_prev_active_inst, -1])
            beamed_tensor = beamed_tensor.index_select(curr_active_inst_idx, axis=0)
            beamed_tensor = beamed_tensor.reshape(new_shape)
            return beamed_tensor

        def collate_active_info(src_enc, inst_idx_to_position_map, active_inst_idx_list):
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.tensor(active_inst_idx, dtype=torch.int64)
            active_src_enc = collect_active_part(src_enc.permute(1, 0, 2), active_inst_idx, n_prev_active_inst, n_bm).permute(1, 0, 2)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            return active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm, memory_key_padding_mask):
            """ Decode and update beam status, and then return active beam idx """

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq)
                dec_partial_seq = dec_partial_seq.reshape([-1, len_dec_seq])
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, n_active_inst, n_bm, memory_key_padding_mask):
                dec_seq = self.embedding(dec_seq).permute(1, 0, 2)
                dec_seq = self.positional_encoding(dec_seq)
                tgt_mask = self.generate_square_subsequent_mask(dec_seq.shape[0])
                dec_output = self.decoder(dec_seq, enc_output, tgt_mask=tgt_mask, tgt_key_padding_mask=None, memory_key_padding_mask=memory_key_padding_mask)
                dec_output = dec_output.permute(1, 0, 2)
                dec_output = dec_output[:, -1, :]
                word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=1)
                word_prob = torch.reshape(word_prob, (n_active_inst, n_bm, -1))
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list
            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, enc_output, n_active_inst, n_bm, None)
            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)
            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]
                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores
        with torch.no_grad():
            if self.encoder is not None:
                src = self.positional_encoding(images.permute(1, 0, 2))
                src_enc = self.encoder(src)
            else:
                src_enc = images.squeeze(2).transpose([0, 2, 1])
            n_bm = self.beam_size
            src_shape = src_enc.shape
            inst_dec_beams = [Beam(n_bm) for _ in range(1)]
            active_inst_idx_list = list(range(1))
            src_enc = src_enc.repeat(1, n_bm, 1)
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            for len_dec_seq in range(1, 25):
                src_enc_copy = src_enc.clone()
                active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, src_enc_copy, inst_idx_to_position_map, n_bm, None)
                if not active_inst_idx_list:
                    break
                src_enc, inst_idx_to_position_map = collate_active_info(src_enc_copy, inst_idx_to_position_map, active_inst_idx_list)
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
        result_hyp = []
        hyp_scores = []
        for bs_hyp, score in zip(batch_hyp, batch_scores):
            l = len(bs_hyp[0])
            bs_hyp_pad = bs_hyp[0] + [3] * (25 - l)
            result_hyp.append(bs_hyp_pad)
            score = float(score) / l
            hyp_score = [score for _ in range(25)]
            hyp_scores.append(hyp_score)
        return [torch.tensor(np.array(result_hyp), dtype=torch.int64), torch.tensor(hyp_scores)]

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = torch.zeros([sz, sz], dtype=torch.float32)
        mask_inf = torch.triu(torch.full(size=[sz, sz], fill_value=float('-Inf'), dtype=torch.float32), diagonal=1)
        mask = mask + mask_inf
        return mask

    def generate_padding_mask(self, x):
        padding_mask = x == torch.tensor(0, dtype=x.dtype)
        return padding_mask

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class PositionalEncoding_2d(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        	ext{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        	ext{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        	ext{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding_2d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).type(torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.unsqueeze(pe, 0).permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(dim, dim)
        self.linear1.weight.data.fill_(1.0)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear2 = nn.Linear(dim, dim)
        self.linear2.weight.data.fill_(1.0)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        w_pe = self.pe[:x.shape[-1], :]
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        w_pe = w_pe * w1
        w_pe = w_pe.permute(1, 2, 0)
        w_pe = torch.unsqueeze(w_pe, 2)
        h_pe = self.pe[:x.shape[-2], :]
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        h_pe = h_pe * w2
        h_pe = h_pe.permute(1, 2, 0)
        h_pe = torch.unsqueeze(h_pe, 3)
        x = x + w_pe + h_pe
        x = torch.reshape(x, [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]).permute(2, 0, 1)
        return self.dropout(x)


class SAREncoder(nn.Module):
    """
    Args:
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        enc_drop_rnn (float): Dropout probability of RNN layer in encoder.
        enc_gru (bool): If True, use GRU, else LSTM in encoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """

    def __init__(self, enc_bi_rnn=False, enc_drop_rnn=0.0, enc_gru=False, d_model=512, d_enc=512, mask=True, **kwargs):
        super().__init__()
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(enc_drop_rnn, (int, float))
        assert 0 <= enc_drop_rnn < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)
        self.enc_bi_rnn = enc_bi_rnn
        self.enc_drop_rnn = enc_drop_rnn
        self.mask = mask
        kwargs = dict(input_size=d_model, hidden_size=d_enc, num_layers=2, batch_first=True, dropout=enc_drop_rnn, bidirectional=enc_bi_rnn)
        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(self, feat, img_metas=None):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.size(0)
        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]
        h_feat = feat.shape[2]
        feat_v = F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        feat_v = feat_v.squeeze(2)
        feat_v = feat_v.permute(0, 2, 1).contiguous()
        holistic_feat = self.rnn_encoder(feat_v)[0]
        if valid_ratios is not None:
            valid_hf = []
            T = holistic_feat.size(1)
            for i in range(valid_ratios.size(0)):
                valid_step = torch.min(T, torch.ceil(T * valid_ratios[i])) - 1
                valid_hf.append(holistic_feat[i, valid_step, :])
            valid_hf = torch.stack(valid_hf, dim=0)
        else:
            valid_hf = holistic_feat[:, -1, :]
        holistic_feat = self.linear(valid_hf)
        return holistic_feat


class BaseDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self, feat, out_enc, label=None, img_metas=None, train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, label, img_metas)
        return self.forward_test(feat, out_enc, img_metas)


class ParallelSARDecoder(BaseDecoder):
    """
    Args:
        out_channels (int): Output class number.
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        dec_bi_rnn (bool): If True, use bidirectional RNN in decoder.
        dec_drop_rnn (float): Dropout of RNN layer in decoder.
        dec_gru (bool): If True, use GRU, else LSTM in decoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        d_k (int): Dim of channels of attention module.
        pred_dropout (float): Dropout probability of prediction layer.
        max_seq_len (int): Maximum sequence length for decoding.
        mask (bool): If True, mask padding in feature map.
        start_idx (int): Index of start token.
        padding_idx (int): Index of padding token.
        pred_concat (bool): If True, concat glimpse feature from
            attention with holistic feature and hidden state.
    """

    def __init__(self, out_channels, enc_bi_rnn=False, dec_bi_rnn=False, dec_drop_rnn=0.0, dec_gru=False, d_model=512, d_enc=512, d_k=64, pred_dropout=0.0, max_text_length=30, mask=True, pred_concat=True, **kwargs):
        super().__init__()
        self.num_classes = out_channels
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = out_channels - 2
        self.padding_idx = out_channels - 1
        self.max_seq_len = max_text_length
        self.mask = mask
        self.pred_concat = pred_concat
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(d_k, 1)
        kwargs = dict(input_size=encoder_rnn_out_size, hidden_size=encoder_rnn_out_size, num_layers=2, batch_first=True, dropout=dec_drop_rnn, bidirectional=dec_bi_rnn)
        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)
        self.embedding = nn.Embedding(self.num_classes, encoder_rnn_out_size, padding_idx=self.padding_idx)
        self.pred_dropout = nn.Dropout(pred_dropout)
        pred_num_classes = self.num_classes - 1
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + encoder_rnn_out_size
        else:
            fc_in_channel = d_model
        self.prediction = nn.Linear(fc_in_channel, pred_num_classes)

    def _2d_attention(self, decoder_input, feat, holistic_feat, valid_ratios=None):
        y = self.rnn_decoder(decoder_input)[0]
        attn_query = self.conv1x1_1(y)
        bsz, seq_len, attn_size = attn_query.shape
        attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)
        attn_key = self.conv3x3_1(feat)
        attn_key = attn_key.unsqueeze(1)
        attn_weight = torch.tanh(torch.add(attn_key, attn_query))
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        attn_weight = self.conv1x1_2(attn_weight)
        bsz, T, h, w, c = attn_weight.size()
        assert c == 1
        if valid_ratios is not None:
            for i in range(valid_ratios.size(0)):
                valid_width = torch.min(w, torch.ceil(w * valid_ratios[i]))
                if valid_width < w:
                    attn_weight[i, :, :, valid_width:, :] = float('-inf')
        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.view(bsz, T, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
        attn_feat = torch.sum(torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False)
        if self.pred_concat:
            hf_c = holistic_feat.shape[-1]
            holistic_feat = holistic_feat.expand(bsz, seq_len, hf_c)
            y = self.prediction(torch.cat((y, attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(attn_feat)
        if self.train_mode:
            y = self.pred_dropout(y)
        return y

    def forward_train(self, feat, out_enc, label, img_metas):
        """
        img_metas: [label, valid_ratio]
        """
        if img_metas is not None:
            assert img_metas[0].size(0) == feat.size(0)
        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]
        lab_embedding = self.embedding(label)
        out_enc = out_enc.unsqueeze(1)
        in_dec = torch.cat((out_enc, lab_embedding), dim=1)
        out_dec = self._2d_attention(in_dec, feat, out_enc, valid_ratios=valid_ratios)
        return out_dec[:, 1:, :]

    def forward_test(self, feat, out_enc, img_metas):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]
        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]
        seq_len = self.max_seq_len
        bsz = feat.size(0)
        start_token = torch.full((bsz,), fill_value=self.start_idx, device=feat.device, dtype=torch.long)
        start_token = self.embedding(start_token)
        emb_dim = start_token.shape[1]
        start_token = start_token.unsqueeze(1).expand(bsz, seq_len, emb_dim)
        out_enc = out_enc.unsqueeze(1)
        decoder_input = torch.cat((out_enc, start_token), dim=1)
        outputs = []
        for i in range(1, seq_len + 1):
            decoder_output = self._2d_attention(decoder_input, feat, out_enc, valid_ratios=valid_ratios)
            char_output = decoder_output[:, i, :]
            char_output = F.softmax(char_output, -1)
            outputs.append(char_output)
            _, max_idx = torch.max(char_output, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding
        outputs = torch.stack(outputs, 1)
        return outputs


class SARHead(nn.Module):

    def __init__(self, in_channels, out_channels, enc_dim=512, max_text_length=30, enc_bi_rnn=False, enc_drop_rnn=0.1, enc_gru=False, dec_bi_rnn=False, dec_drop_rnn=0.0, dec_gru=False, d_k=512, pred_dropout=0.1, pred_concat=True, **kwargs):
        super(SARHead, self).__init__()
        self.encoder = SAREncoder(enc_bi_rnn=enc_bi_rnn, enc_drop_rnn=enc_drop_rnn, enc_gru=enc_gru, d_model=in_channels, d_enc=enc_dim)
        self.decoder = ParallelSARDecoder(out_channels=out_channels, enc_bi_rnn=enc_bi_rnn, dec_bi_rnn=dec_bi_rnn, dec_drop_rnn=dec_drop_rnn, dec_gru=dec_gru, d_model=in_channels, d_enc=enc_dim, d_k=d_k, pred_dropout=pred_dropout, max_text_length=max_text_length, pred_concat=pred_concat)

    def forward(self, feat, targets=None):
        """
        img_metas: [label, valid_ratio]
        """
        holistic_feat = self.encoder(feat, targets)
        if self.training:
            label = targets[0]
            final_out = self.decoder(feat, holistic_feat, label, img_metas=targets)
        else:
            final_out = self.decoder(feat, holistic_feat, label=None, img_metas=targets, train_mode=False)
        return final_out


def hard_swish(x, inplace=True):
    return x * F.relu6(x + 3.0, inplace=inplace) / 6.0


class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, groups=None, if_act=True, act='relu', **kwargs):
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.if_act = if_act
        self.act = act
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels * 4), kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(in_channels * 4))
        self.conv3 = nn.Conv2d(in_channels=int(in_channels * 4), out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == 'relu':
                x = F.relu(x)
            elif self.act == 'hardswish':
                x = hard_swish(x)
            else:
                None
                exit()
        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


class ASFBlock(nn.Module):
    """
    This code is refered from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """

    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.spatial_scale = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False, padding=1), nn.ReLU(), nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False), nn.Sigmoid())
        self.channel_scale = nn.Sequential(nn.Conv2d(in_channels=inter_channels, out_channels=out_features_num, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)
        spatial_x = torch.mean(fuse_features, dim=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num
        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i:i + 1] * features_list[i])
        return torch.cat(out_list, dim=1)


class DBFPN(nn.Module):

    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf
        self.in2_conv = nn.Conv2d(in_channels=in_channels[0], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels=in_channels[1], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels=in_channels[2], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels=in_channels[3], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.p5_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p2_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)
        out4 = in4 + F.interpolate(in5, scale_factor=2, mode='nearest')
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode='nearest')
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode='nearest')
        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.interpolate(p5, scale_factor=8, mode='nearest')
        p4 = F.interpolate(p4, scale_factor=4, mode='nearest')
        p3 = F.interpolate(p3, scale_factor=2, mode='nearest')
        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])
        return fuse


class RSELayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=kernel_size, padding=int(kernel_size // 2), bias=False)
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        for i in range(len(in_channels)):
            self.ins_conv.append(RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut))
            self.inp_conv.append(RSELayer(out_channels, out_channels // 4, kernel_size=3, shortcut=shortcut))

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)
        out4 = in4 + F.upsample(in5, scale_factor=2, mode='nearest')
        out3 = in3 + F.upsample(out4, scale_factor=2, mode='nearest')
        out2 = in2 + F.upsample(out3, scale_factor=2, mode='nearest')
        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)
        p5 = F.upsample(p5, scale_factor=8, mode='nearest')
        p4 = F.upsample(p4, scale_factor=4, mode='nearest')
        p3 = F.upsample(p3, scale_factor=2, mode='nearest')
        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class LKPAN(nn.Module):

    def __init__(self, in_channels, out_channels, mode='large', **kwargs):
        super(LKPAN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()
        if mode.lower() == 'lite':
            p_layer = DSConv
        elif mode.lower() == 'large':
            p_layer = nn.Conv2d
        else:
            raise ValueError("mode can only be one of ['lite', 'large'], but received {}".format(mode))
        for i in range(len(in_channels)):
            self.ins_conv.append(nn.Conv2d(in_channels=in_channels[i], out_channels=self.out_channels, kernel_size=1, bias=False))
            self.inp_conv.append(p_layer(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=9, padding=4, bias=False))
            if i > 0:
                self.pan_head_conv.append(nn.Conv2d(in_channels=self.out_channels // 4, out_channels=self.out_channels // 4, kernel_size=3, padding=1, stride=2, bias=False))
            self.pan_lat_conv.append(p_layer(in_channels=self.out_channels // 4, out_channels=self.out_channels // 4, kernel_size=9, padding=4, bias=False))

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)
        out4 = in4 + F.upsample(in5, scale_factor=2, mode='nearest')
        out3 = in3 + F.upsample(out4, scale_factor=2, mode='nearest')
        out2 = in2 + F.upsample(out3, scale_factor=2, mode='nearest')
        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)
        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)
        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)
        p5 = F.upsample(p5, scale_factor=8, mode='nearest')
        p4 = F.upsample(p4, scale_factor=4, mode='nearest')
        p3 = F.upsample(p3, scale_factor=2, mode='nearest')
        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class DeConvBNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, if_act=True, act=None, name=None):
        super(DeConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act
        if act is not None:
            self._act = Activation(act)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self._act(x)
        return x


class EASTFPN(nn.Module):

    def __init__(self, in_channels, model_name, **kwargs):
        super(EASTFPN, self).__init__()
        self.model_name = model_name
        if self.model_name == 'large':
            self.out_channels = 128
        else:
            self.out_channels = 64
        self.in_channels = in_channels[::-1]
        self.h1_conv = ConvBNLayer(in_channels=self.out_channels + self.in_channels[1], out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, if_act=True, act='relu', name='unet_h_1')
        self.h2_conv = ConvBNLayer(in_channels=self.out_channels + self.in_channels[2], out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, if_act=True, act='relu', name='unet_h_2')
        self.h3_conv = ConvBNLayer(in_channels=self.out_channels + self.in_channels[3], out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, if_act=True, act='relu', name='unet_h_3')
        self.g0_deconv = DeConvBNLayer(in_channels=self.in_channels[0], out_channels=self.out_channels, kernel_size=4, stride=2, padding=1, if_act=True, act='relu', name='unet_g_0')
        self.g1_deconv = DeConvBNLayer(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=4, stride=2, padding=1, if_act=True, act='relu', name='unet_g_1')
        self.g2_deconv = DeConvBNLayer(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=4, stride=2, padding=1, if_act=True, act='relu', name='unet_g_2')
        self.g3_conv = ConvBNLayer(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, if_act=True, act='relu', name='unet_g_3')

    def forward(self, x):
        f = x[::-1]
        h = f[0]
        g = self.g0_deconv(h)
        h = torch.cat([g, f[1]], dim=1)
        h = self.h1_conv(h)
        g = self.g1_deconv(h)
        h = torch.cat([g, f[2]], dim=1)
        h = self.h2_conv(h)
        g = self.g2_deconv(h)
        h = torch.cat([g, f[3]], dim=1)
        h = self.h3_conv(h)
        g = self.g3_conv(h)
        return g


class FCEFPN(nn.Module):
    """
    Feature Pyramid Network, see https://arxiv.org/abs/1612.03144
    Args:
        in_channels (list[int]): input channels of each level which can be
            derived from the output shape of backbone by from_config
        out_channels (list[int]): output channel of each level
        spatial_scales (list[float]): the spatial scales between input feature
            maps and original input image which can be derived from the output
            shape of backbone by from_config
        has_extra_convs (bool): whether to add extra conv to the last level.
            default False
        extra_stage (int): the number of extra stages added to the last level.
            default 1
        use_c5 (bool): Whether to use c5 as the input of extra stage,
            otherwise p5 is used. default True
        norm_type (string|None): The normalization type in FPN module. If
            norm_type is None, norm will not be used after conv and if
            norm_type is string, bn, gn, sync_bn are available. default None
        norm_decay (float): weight decay for normalization layer weights.
            default 0.
        freeze_norm (bool): whether to freeze normalization layer.
            default False
        relu_before_extra_convs (bool): whether to add relu before extra convs.
            default False

    """

    def __init__(self, in_channels, out_channels, spatial_scales=[0.25, 0.125, 0.0625, 0.03125], has_extra_convs=False, extra_stage=1, use_c5=True, norm_type=None, norm_decay=0.0, freeze_norm=False, relu_before_extra_convs=True):
        super(FCEFPN, self).__init__()
        self.out_channels = out_channels
        for s in range(extra_stage):
            spatial_scales = spatial_scales + [spatial_scales[-1] / 2.0]
        self.spatial_scales = spatial_scales
        self.has_extra_convs = has_extra_convs
        self.extra_stage = extra_stage
        self.use_c5 = use_c5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.lateral_convs = []
        self.lateral_convs_module = nn.ModuleList()
        self.fpn_convs = []
        self.fpn_convs_module = nn.ModuleList()
        fan = out_channels * 3 * 3
        st_stage = 4 - len(in_channels)
        ed_stage = st_stage + len(in_channels) - 1
        for i in range(st_stage, ed_stage + 1):
            if i == 3:
                lateral_name = 'fpn_inner_res5_sum'
            else:
                lateral_name = 'fpn_inner_res{}_sum_lateral'.format(i + 2)
            in_c = in_channels[i - st_stage]
            if self.norm_type is not None:
                lateral = ConvNormLayer(ch_in=in_c, ch_out=out_channels, filter_size=1, stride=1, norm_type=self.norm_type, norm_decay=self.norm_decay, freeze_norm=self.freeze_norm, initializer=None)
            else:
                lateral = nn.Conv2d(in_channels=in_c, out_channels=out_channels, kernel_size=1)
            self.lateral_convs_module.add_module(lateral_name, lateral)
            self.lateral_convs.append(lateral)
        for i in range(st_stage, ed_stage + 1):
            fpn_name = 'fpn_res{}_sum'.format(i + 2)
            fpn_conv_module = nn.Sequential()
            if self.norm_type is not None:
                fpn_conv = ConvNormLayer(ch_in=out_channels, ch_out=out_channels, filter_size=3, stride=1, norm_type=self.norm_type, norm_decay=self.norm_decay, freeze_norm=self.freeze_norm, initializer=None)
            else:
                fpn_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
            self.fpn_convs_module.add_module(fpn_name, fpn_conv)
            self.fpn_convs.append(fpn_conv)
        if self.has_extra_convs:
            for i in range(self.extra_stage):
                lvl = ed_stage + 1 + i
                if i == 0 and self.use_c5:
                    in_c = in_channels[-1]
                else:
                    in_c = out_channels
                extra_fpn_name = 'fpn_{}'.format(lvl + 2)
                extra_fpn_conv_module = nn.Sequential()
                if self.norm_type is not None:
                    extra_fpn_conv = ConvNormLayer(ch_in=in_c, ch_out=out_channels, filter_size=3, stride=2, norm_type=self.norm_type, norm_decay=self.norm_decay, freeze_norm=self.freeze_norm, initializer=None)
                else:
                    extra_fpn_conv = nn.Conv2d(in_channels=in_c, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
                self.fpn_convs_module.add_module(extra_fpn_name, extra_fpn_conv)
                self.fpn_convs.append(extra_fpn_conv)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], 'spatial_scales': [(1.0 / i.stride) for i in input_shape]}

    def forward(self, body_feats):
        laterals = []
        num_levels = len(body_feats)
        for i in range(num_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))
        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = F.interpolate(laterals[lvl], scale_factor=2.0, mode='nearest')
            laterals[lvl - 1] += upsample
        fpn_output = []
        for lvl in range(num_levels):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))
        if self.extra_stage > 0:
            if not self.has_extra_convs:
                assert self.extra_stage == 1, 'extra_stage should be 1 if FPN has not extra convs'
                fpn_output.append(torch.max_pool2d(fpn_output[-1], 1, stride=2))
            else:
                if self.use_c5:
                    extra_source = body_feats[-1]
                else:
                    extra_source = fpn_output[-1]
                fpn_output.append(self.fpn_convs[num_levels](extra_source))
                for i in range(1, self.extra_stage):
                    if self.relu_before_extra_convs:
                        fpn_output.append(self.fpn_convs[num_levels + i](F.relu(fpn_output[-1])))
                    else:
                        fpn_output.append(self.fpn_convs[num_levels + i](fpn_output[-1]))
        return fpn_output


class PGFPN(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(PGFPN, self).__init__()
        num_inputs = [2048, 2048, 1024, 512, 256]
        num_outputs = [256, 256, 192, 192, 128]
        self.out_channels = 128
        self.conv_bn_layer_1 = ConvBNLayer(in_channels=3, out_channels=32, kernel_size=3, stride=1, act=None, name='FPN_d1')
        self.conv_bn_layer_2 = ConvBNLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, act=None, name='FPN_d2')
        self.conv_bn_layer_3 = ConvBNLayer(in_channels=256, out_channels=128, kernel_size=3, stride=1, act=None, name='FPN_d3')
        self.conv_bn_layer_4 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=2, act=None, name='FPN_d4')
        self.conv_bn_layer_5 = ConvBNLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, act='relu', name='FPN_d5')
        self.conv_bn_layer_6 = ConvBNLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2, act=None, name='FPN_d6')
        self.conv_bn_layer_7 = ConvBNLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, act='relu', name='FPN_d7')
        self.conv_bn_layer_8 = ConvBNLayer(in_channels=128, out_channels=128, kernel_size=1, stride=1, act=None, name='FPN_d8')
        self.conv_h0 = ConvBNLayer(in_channels=num_inputs[0], out_channels=num_outputs[0], kernel_size=1, stride=1, act=None, name='conv_h{}'.format(0))
        self.conv_h1 = ConvBNLayer(in_channels=num_inputs[1], out_channels=num_outputs[1], kernel_size=1, stride=1, act=None, name='conv_h{}'.format(1))
        self.conv_h2 = ConvBNLayer(in_channels=num_inputs[2], out_channels=num_outputs[2], kernel_size=1, stride=1, act=None, name='conv_h{}'.format(2))
        self.conv_h3 = ConvBNLayer(in_channels=num_inputs[3], out_channels=num_outputs[3], kernel_size=1, stride=1, act=None, name='conv_h{}'.format(3))
        self.conv_h4 = ConvBNLayer(in_channels=num_inputs[4], out_channels=num_outputs[4], kernel_size=1, stride=1, act=None, name='conv_h{}'.format(4))
        self.dconv0 = DeConvBNLayer(in_channels=num_outputs[0], out_channels=num_outputs[0 + 1], name='dconv_{}'.format(0))
        self.dconv1 = DeConvBNLayer(in_channels=num_outputs[1], out_channels=num_outputs[1 + 1], act=None, name='dconv_{}'.format(1))
        self.dconv2 = DeConvBNLayer(in_channels=num_outputs[2], out_channels=num_outputs[2 + 1], act=None, name='dconv_{}'.format(2))
        self.dconv3 = DeConvBNLayer(in_channels=num_outputs[3], out_channels=num_outputs[3 + 1], act=None, name='dconv_{}'.format(3))
        self.conv_g1 = ConvBNLayer(in_channels=num_outputs[1], out_channels=num_outputs[1], kernel_size=3, stride=1, act='relu', name='conv_g{}'.format(1))
        self.conv_g2 = ConvBNLayer(in_channels=num_outputs[2], out_channels=num_outputs[2], kernel_size=3, stride=1, act='relu', name='conv_g{}'.format(2))
        self.conv_g3 = ConvBNLayer(in_channels=num_outputs[3], out_channels=num_outputs[3], kernel_size=3, stride=1, act='relu', name='conv_g{}'.format(3))
        self.conv_g4 = ConvBNLayer(in_channels=num_outputs[4], out_channels=num_outputs[4], kernel_size=3, stride=1, act='relu', name='conv_g{}'.format(4))
        self.convf = ConvBNLayer(in_channels=num_outputs[4], out_channels=num_outputs[4], kernel_size=1, stride=1, act=None, name='conv_f{}'.format(4))

    def forward(self, x):
        c0, c1, c2, c3, c4, c5, c6 = x
        f = [c0, c1, c2]
        g = [None, None, None]
        h = [None, None, None]
        h[0] = self.conv_bn_layer_1(f[0])
        h[1] = self.conv_bn_layer_2(f[1])
        h[2] = self.conv_bn_layer_3(f[2])
        g[0] = self.conv_bn_layer_4(h[0])
        g[1] = torch.add(g[0], h[1])
        g[1] = F.relu(g[1])
        g[1] = self.conv_bn_layer_5(g[1])
        g[1] = self.conv_bn_layer_6(g[1])
        g[2] = torch.add(g[1], h[2])
        g[2] = F.relu(g[2])
        g[2] = self.conv_bn_layer_7(g[2])
        f_down = self.conv_bn_layer_8(g[2])
        f1 = [c6, c5, c4, c3, c2]
        g = [None, None, None, None, None]
        h = [None, None, None, None, None]
        h[0] = self.conv_h0(f1[0])
        h[1] = self.conv_h1(f1[1])
        h[2] = self.conv_h2(f1[2])
        h[3] = self.conv_h3(f1[3])
        h[4] = self.conv_h4(f1[4])
        g[0] = self.dconv0(h[0])
        g[1] = torch.add(g[0], h[1])
        g[1] = F.relu(g[1])
        g[1] = self.conv_g1(g[1])
        g[1] = self.dconv1(g[1])
        g[2] = torch.add(g[1], h[2])
        g[2] = F.relu(g[2])
        g[2] = self.conv_g2(g[2])
        g[2] = self.dconv2(g[2])
        g[3] = torch.add(g[2], h[3])
        g[3] = F.relu(g[3])
        g[3] = self.conv_g3(g[3])
        g[3] = self.dconv3(g[3])
        g[4] = torch.add(g[3], h[4])
        g[4] = F.relu(g[4])
        g[4] = self.conv_g4(g[4])
        f_up = self.convf(g[4])
        f_common = torch.add(f_down, f_up)
        f_common = F.relu(f_common)
        return f_common


class Im2Seq(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.squeeze(dim=2)
        x = x.permute(0, 2, 1)
        return x


class EncoderWithRNN_(nn.Module):

    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN_, self).__init__()
        self.out_channels = hidden_size * 2
        self.rnn1 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)
        self.rnn2 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)

    def forward(self, x):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(torch.flip(x, [1]))
        return torch.cat([out1, torch.flip(out2, [1])], 2)


class EncoderWithRNN(nn.Module):

    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Module):

    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(in_channels, hidden_size, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


class EncoderWithSVTR(nn.Module):

    def __init__(self, in_channels, dims=64, depth=2, hidden_dims=120, use_guide=False, num_heads=8, qkv_bias=True, mlp_ratio=2.0, drop_rate=0.1, attn_drop_rate=0.1, drop_path=0.0, qk_scale=None):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(in_channels, in_channels // 8, padding=1, act='swish')
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1, act='swish')
        self.svtr_block = nn.ModuleList([Block(dim=hidden_dims, num_heads=num_heads, mixer='Global', HW=None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer='swish', attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer='nn.LayerNorm', epsilon=1e-05, prenorm=False) for i in range(depth)])
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-06)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act='swish')
        self.conv4 = ConvBNLayer(2 * in_channels, in_channels // 8, padding=1, act='swish')
        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act='swish')
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        h = z
        z = self.conv1(z)
        z = self.conv2(z)
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        z = z.reshape([-1, H, W, C]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z


class SequenceEncoder(nn.Module):

    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {'reshape': Im2Seq, 'fc': EncoderWithFC, 'rnn': EncoderWithRNN, 'svtr': EncoderWithSVTR}
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(encoder_type, support_encoder_dict.keys())
            if encoder_type == 'svtr':
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, **kwargs)
            else:
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x


class FPN_Up_Fusion(nn.Module):

    def __init__(self, in_channels):
        super(FPN_Up_Fusion, self).__init__()
        in_channels = in_channels[::-1]
        out_channels = [256, 256, 192, 192, 128]
        self.h0_conv = ConvBNLayer(in_channels[0], out_channels[0], 1, 1, act=None, name='fpn_up_h0')
        self.h1_conv = ConvBNLayer(in_channels[1], out_channels[1], 1, 1, act=None, name='fpn_up_h1')
        self.h2_conv = ConvBNLayer(in_channels[2], out_channels[2], 1, 1, act=None, name='fpn_up_h2')
        self.h3_conv = ConvBNLayer(in_channels[3], out_channels[3], 1, 1, act=None, name='fpn_up_h3')
        self.h4_conv = ConvBNLayer(in_channels[4], out_channels[4], 1, 1, act=None, name='fpn_up_h4')
        self.g0_conv = DeConvBNLayer(out_channels[0], out_channels[1], 4, 2, act=None, name='fpn_up_g0')
        self.g1_conv = nn.Sequential(ConvBNLayer(out_channels[1], out_channels[1], 3, 1, act='relu', name='fpn_up_g1_1'), DeConvBNLayer(out_channels[1], out_channels[2], 4, 2, act=None, name='fpn_up_g1_2'))
        self.g2_conv = nn.Sequential(ConvBNLayer(out_channels[2], out_channels[2], 3, 1, act='relu', name='fpn_up_g2_1'), DeConvBNLayer(out_channels[2], out_channels[3], 4, 2, act=None, name='fpn_up_g2_2'))
        self.g3_conv = nn.Sequential(ConvBNLayer(out_channels[3], out_channels[3], 3, 1, act='relu', name='fpn_up_g3_1'), DeConvBNLayer(out_channels[3], out_channels[4], 4, 2, act=None, name='fpn_up_g3_2'))
        self.g4_conv = nn.Sequential(ConvBNLayer(out_channels[4], out_channels[4], 3, 1, act='relu', name='fpn_up_fusion_1'), ConvBNLayer(out_channels[4], out_channels[4], 1, 1, act=None, name='fpn_up_fusion_2'))

    def _add_relu(self, x1, x2):
        x = torch.add(x1, x2)
        x = F.relu(x)
        return x

    def forward(self, x):
        f = x[2:][::-1]
        h0 = self.h0_conv(f[0])
        h1 = self.h1_conv(f[1])
        h2 = self.h2_conv(f[2])
        h3 = self.h3_conv(f[3])
        h4 = self.h4_conv(f[4])
        g0 = self.g0_conv(h0)
        g1 = self._add_relu(g0, h1)
        g1 = self.g1_conv(g1)
        g2 = self.g2_conv(self._add_relu(g1, h2))
        g3 = self.g3_conv(self._add_relu(g2, h3))
        g4 = self.g4_conv(self._add_relu(g3, h4))
        return g4


class FPN_Down_Fusion(nn.Module):

    def __init__(self, in_channels):
        super(FPN_Down_Fusion, self).__init__()
        out_channels = [32, 64, 128]
        self.h0_conv = ConvBNLayer(in_channels[0], out_channels[0], 3, 1, act=None, name='fpn_down_h0')
        self.h1_conv = ConvBNLayer(in_channels[1], out_channels[1], 3, 1, act=None, name='fpn_down_h1')
        self.h2_conv = ConvBNLayer(in_channels[2], out_channels[2], 3, 1, act=None, name='fpn_down_h2')
        self.g0_conv = ConvBNLayer(out_channels[0], out_channels[1], 3, 2, act=None, name='fpn_down_g0')
        self.g1_conv = nn.Sequential(ConvBNLayer(out_channels[1], out_channels[1], 3, 1, act='relu', name='fpn_down_g1_1'), ConvBNLayer(out_channels[1], out_channels[2], 3, 2, act=None, name='fpn_down_g1_2'))
        self.g2_conv = nn.Sequential(ConvBNLayer(out_channels[2], out_channels[2], 3, 1, act='relu', name='fpn_down_fusion_1'), ConvBNLayer(out_channels[2], out_channels[2], 1, 1, act=None, name='fpn_down_fusion_2'))

    def forward(self, x):
        f = x[:3]
        h0 = self.h0_conv(f[0])
        h1 = self.h1_conv(f[1])
        h2 = self.h2_conv(f[2])
        g0 = self.g0_conv(h0)
        g1 = torch.add(g0, h1)
        g1 = F.relu(g1)
        g1 = self.g1_conv(g1)
        g2 = torch.add(g1, h2)
        g2 = F.relu(g2)
        g2 = self.g2_conv(g2)
        return g2


class Cross_Attention(nn.Module):

    def __init__(self, in_channels):
        super(Cross_Attention, self).__init__()
        self.theta_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_theta')
        self.phi_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_phi')
        self.g_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_g')
        self.fh_weight_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fh_weight')
        self.fh_sc_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fh_sc')
        self.fv_weight_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fv_weight')
        self.fv_sc_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fv_sc')
        self.f_attn_conv = ConvBNLayer(in_channels * 2, in_channels, 1, 1, act='relu', name='f_attn')

    def _cal_fweight(self, f, shape):
        f_theta, f_phi, f_g = f
        f_theta = f_theta.permute(0, 2, 3, 1)
        f_theta = torch.reshape(f_theta, [shape[0] * shape[1], shape[2], 128])
        f_phi = f_phi.permute(0, 2, 3, 1)
        f_phi = torch.reshape(f_phi, [shape[0] * shape[1], shape[2], 128])
        f_g = f_g.permute(0, 2, 3, 1)
        f_g = torch.reshape(f_g, [shape[0] * shape[1], shape[2], 128])
        f_attn = torch.matmul(f_theta, f_phi.permute(0, 2, 1))
        f_attn = f_attn / 128 ** 0.5
        f_attn = F.softmax(f_attn, dim=-1)
        f_weight = torch.matmul(f_attn, f_g)
        f_weight = torch.reshape(f_weight, [shape[0], shape[1], shape[2], 128])
        return f_weight

    def forward(self, f_common):
        f_shape = f_common.size()
        f_theta = self.theta_conv(f_common)
        f_phi = self.phi_conv(f_common)
        f_g = self.g_conv(f_common)
        fh_weight = self._cal_fweight([f_theta, f_phi, f_g], [f_shape[0], f_shape[2], f_shape[3]])
        fh_weight = fh_weight.permute(0, 3, 1, 2)
        fh_weight = self.fh_weight_conv(fh_weight)
        fh_sc = self.fh_sc_conv(f_common)
        f_h = F.relu(fh_weight + fh_sc)
        fv_theta = f_theta.permute(0, 1, 3, 2)
        fv_phi = f_phi.permute(0, 1, 3, 2)
        fv_g = f_g.permute(0, 1, 3, 2)
        fv_weight = self._cal_fweight([fv_theta, fv_phi, fv_g], [f_shape[0], f_shape[3], f_shape[2]])
        fv_weight = fv_weight.permute(0, 3, 2, 1)
        fv_weight = self.fv_weight_conv(fv_weight)
        fv_sc = self.fv_sc_conv(f_common)
        f_v = F.relu(fv_weight + fv_sc)
        f_attn = torch.cat([f_h, f_v], dim=1)
        f_attn = self.f_attn_conv(f_attn)
        return f_attn


class SASTFPN(nn.Module):

    def __init__(self, in_channels, with_cab=False, **kwargs):
        super(SASTFPN, self).__init__()
        self.in_channels = in_channels
        self.with_cab = with_cab
        self.FPN_Down_Fusion = FPN_Down_Fusion(self.in_channels)
        self.FPN_Up_Fusion = FPN_Up_Fusion(self.in_channels)
        self.out_channels = 128
        self.cross_attention = Cross_Attention(self.out_channels)

    def forward(self, x):
        f_down = self.FPN_Down_Fusion(x)
        f_up = self.FPN_Up_Fusion(x)
        f_common = torch.add(f_down, f_up)
        f_common = F.relu(f_common)
        if self.with_cab:
            f_common = self.cross_attention(f_common)
        return f_common


def conv3x3_block(in_channels, out_channels, stride=1):
    n = 3 * 3 * out_channels
    w = math.sqrt(2.0 / n)
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
    block = nn.Sequential(conv_layer, nn.BatchNorm2d(out_channels), nn.ReLU())
    return block


class STN(nn.Module):

    def __init__(self, in_channels, num_ctrlpoints, activation='none'):
        super(STN, self).__init__()
        self.in_channels = in_channels
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.Sequential(conv3x3_block(in_channels, 32), nn.MaxPool2d(kernel_size=2, stride=2), conv3x3_block(32, 64), nn.MaxPool2d(kernel_size=2, stride=2), conv3x3_block(64, 128), nn.MaxPool2d(kernel_size=2, stride=2), conv3x3_block(128, 256), nn.MaxPool2d(kernel_size=2, stride=2), conv3x3_block(256, 256), nn.MaxPool2d(kernel_size=2, stride=2), conv3x3_block(256, 256))
        self.stn_fc1 = nn.Sequential(nn.Linear(2 * 256, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        fc2_bias = self.init_stn()
        self.stn_fc2 = nn.Linear(512, num_ctrlpoints * 2, bias=True)

    def init_stn(self):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1.0 - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        if self.activation == 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1.0 / ctrl_points - 1.0)
        ctrl_points = torch.Tensor(ctrl_points)
        fc2_bias = torch.reshape(ctrl_points, shape=[ctrl_points.shape[0] * ctrl_points.shape[1]])
        return fc2_bias

    def forward(self, x):
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.shape
        x = torch.reshape(x, shape=(batch_size, -1))
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        x = torch.reshape(x, shape=[-1, self.num_ctrlpoints, 2])
        return img_feat, x


def build_output_control_points(num_control_points, margins):
    margin_x, margin_y = margins
    num_ctrl_pts_per_side = num_control_points // 2
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
    output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
    return output_ctrl_pts


def compute_partial_repr(input_points, control_points):
    N = input_points.shape[0]
    M = control_points.shape[0]
    pairwise_diff = torch.reshape(input_points, shape=[N, 1, 2]) - torch.reshape(control_points, shape=[1, M, 2])
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


def grid_sample(input, grid, canvas=None):
    input.stop_gradient = False
    output = F.grid_sample(input, grid, align_corners=True) if torch.__version__ >= '1.3.0' else F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = input.data.new(input.size()).fill_(1)
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


class TPSSpatialTransformer(nn.Module):

    def __init__(self, output_image_size=None, num_control_points=None, margins=None):
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins
        self.target_height, self.target_width = output_image_size
        target_control_points = build_output_control_points(num_control_points, margins)
        N = num_control_points
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        inverse_kernel = torch.inverse(forward_kernel)
        HW = self.target_height * self.target_width
        target_coordinate = list(itertools.product(range(self.target_height), range(self.target_width)))
        target_coordinate = torch.Tensor(target_coordinate)
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y / (self.target_height - 1)
        X = X / (self.target_width - 1)
        target_coordinate = torch.cat([X, Y], dim=1)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate], dim=1)
        self.inverse_kernel = inverse_kernel
        self.padding_matrix = torch.zeros(3, 2)
        self.target_coordinate_repr = target_coordinate_repr
        self.target_control_points = target_control_points

    def forward(self, input, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.shape[1] == self.num_control_points
        assert source_control_points.shape[2] == 2
        batch_size = source_control_points.size(0)
        Y = torch.cat([source_control_points, self.padding_matrix.expand(batch_size, 3, 2)], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
        grid = torch.reshape(source_coordinate, shape=[-1, self.target_height, self.target_width, 2])
        grid = torch.clamp(grid, 0, 1)
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps, source_coordinate


class STN_ON(nn.Module):

    def __init__(self, in_channels, tps_inputsize, tps_outputsize, num_control_points, tps_margins, stn_activation):
        super(STN_ON, self).__init__()
        self.tps = TPSSpatialTransformer(output_image_size=tuple(tps_outputsize), num_control_points=num_control_points, margins=tuple(tps_margins))
        self.stn_head = STN(in_channels=in_channels, num_ctrlpoints=num_control_points, activation=stn_activation)
        self.tps_inputsize = tps_inputsize
        self.out_channels = in_channels

    def forward(self, image):
        stn_input = torch.nn.functional.interpolate(image, self.tps_inputsize, mode='bilinear', align_corners=True)
        stn_img_feat, ctrl_points = self.stn_head(stn_input)
        x, _ = self.tps(image, ctrl_points)
        return x


class LocalizationNetwork(nn.Module):

    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(LocalizationNetwork, self).__init__()
        self.F = num_fiducial
        F = num_fiducial
        if model_name == 'large':
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64
        self.block_list = nn.Sequential()
        for fno in range(0, len(num_filters_list)):
            num_filters = num_filters_list[fno]
            name = 'loc_conv%d' % fno
            conv = ConvBNLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=3, act='relu', name=name)
            self.block_list.add_module(name, conv)
            if fno == len(num_filters_list) - 1:
                pool = nn.AdaptiveAvgPool2d(1)
            else:
                pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            in_channels = num_filters
            self.block_list.add_module('{}_pool'.format(name), pool)
        name = 'loc_fc1'
        stdv = 1.0 / math.sqrt(num_filters_list[-1] * 1.0)
        self.fc1 = nn.Linear(in_channels, fc_dim, bias=True)
        initial_bias = self.get_initial_fiducials()
        initial_bias = initial_bias.reshape(-1)
        name = 'loc_fc2'
        self.fc2 = nn.Linear(fc_dim, F * 2, bias=True)
        self.out_channels = F * 2

    def forward(self, x):
        """
           Estimating parameters of geometric transformation
           Args:
               image: input
           Return:
               batch_C_prime: the matrix of the geometric transformation
        """
        B = x.shape[0]
        i = 0
        for block in self.block_list:
            x = block(x)
        x = x.squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(shape=[-1, self.F, 2])
        return x

    def get_initial_fiducials(self):
        """ see RARE paper Fig. 6 (a) """
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return initial_bias


class GridGenerator(nn.Module):

    def __init__(self, in_channels, num_fiducial):
        super(GridGenerator, self).__init__()
        self.eps = 1e-06
        self.F = num_fiducial
        name = 'ex_fc'
        self.fc = nn.Linear(in_channels, 6, bias=True)

    def forward(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return:
            batch_P_prime: the grid for the grid_sampler
        """
        C = self.build_C_paddle()
        P = self.build_P_paddle(I_r_size)
        inv_delta_C_tensor = self.build_inv_delta_C_paddle(C).type(torch.float32)
        P_hat_tensor = self.build_P_hat_paddle(C, torch.as_tensor(P)).type(torch.float32)
        inv_delta_C_tensor.stop_gradient = True
        P_hat_tensor.stop_gradient = True
        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)
        batch_C_ex_part_tensor.stop_gradient = True
        batch_C_prime_with_zeros = torch.cat([batch_C_prime, batch_C_ex_part_tensor], dim=1)
        inv_delta_C_tensor = inv_delta_C_tensor
        batch_T = torch.matmul(inv_delta_C_tensor, batch_C_prime_with_zeros)
        P_hat_tensor = P_hat_tensor
        batch_P_prime = torch.matmul(P_hat_tensor, batch_T)
        return batch_P_prime

    def build_C_paddle(self):
        """ Return coordinates of fiducial points in I_r; C """
        F = self.F
        ctrl_pts_x = torch.linspace(-1.0, 1.0, int(F / 2), dtype=torch.float64)
        ctrl_pts_y_top = -1 * torch.ones([int(F / 2)], dtype=torch.float64)
        ctrl_pts_y_bottom = torch.ones([int(F / 2)], dtype=torch.float64)
        ctrl_pts_top = torch.stack([ctrl_pts_x, ctrl_pts_y_top], dim=1)
        ctrl_pts_bottom = torch.stack([ctrl_pts_x, ctrl_pts_y_bottom], dim=1)
        C = torch.cat([ctrl_pts_top, ctrl_pts_bottom], dim=0)
        return C

    def build_P_paddle(self, I_r_size):
        I_r_height, I_r_width = I_r_size
        I_r_grid_x = (torch.arange(-I_r_width, I_r_width, 2, dtype=torch.float64) + 1.0) / torch.as_tensor(np.array([I_r_width]).astype(np.float64))
        I_r_grid_y = (torch.arange(-I_r_height, I_r_height, 2, dtype=torch.float64) + 1.0) / torch.as_tensor(np.array([I_r_height]).astype(np.float64))
        P = torch.stack(torch.meshgrid([I_r_grid_x, I_r_grid_y]), dim=2)
        P = P.permute(1, 0, 2)
        return P.reshape([-1, 2])

    def build_inv_delta_C_paddle(self, C):
        """ Return inv_delta_C which is needed to calculate T """
        F = self.F
        hat_C = torch.zeros((F, F), dtype=torch.float64)
        for i in range(0, F):
            for j in range(i, F):
                if i == j:
                    hat_C[i, j] = 1
                else:
                    r = torch.norm(C[i] - C[j])
                    hat_C[i, j] = r
                    hat_C[j, i] = r
        hat_C = hat_C ** 2 * torch.log(hat_C)
        delta_C = torch.cat([torch.cat([torch.ones((F, 1), dtype=torch.float64), C, hat_C], dim=1), torch.cat([torch.zeros((2, 3), dtype=torch.float64), C.permute(1, 0)], dim=1), torch.cat([torch.zeros((1, 3), dtype=torch.float64), torch.ones((1, F), dtype=torch.float64)], dim=1)], dim=0)
        inv_delta_C = torch.inverse(delta_C)
        return inv_delta_C

    def build_P_hat_paddle(self, C, P):
        F = self.F
        eps = self.eps
        n = P.shape[0]
        P_tile = torch.unsqueeze(P, dim=1).repeat(1, F, 1)
        C_tile = torch.unsqueeze(C, dim=0)
        P_diff = P_tile - C_tile
        rbf_norm = torch.norm(P_diff, p=2, dim=2, keepdim=False)
        rbf = torch.mul(rbf_norm ** 2, torch.log(rbf_norm + eps))
        P_hat = torch.cat([torch.ones((n, 1), dtype=torch.float64), P, rbf], dim=1)
        return P_hat

    def get_expand_tensor(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = batch_C_prime.reshape([B, H * C])
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        batch_C_ex_part_tensor = batch_C_ex_part_tensor.reshape([-1, 3, 2])
        return batch_C_ex_part_tensor


class TPS(nn.Module):

    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(TPS, self).__init__()
        self.loc_net = LocalizationNetwork(in_channels, num_fiducial, loc_lr, model_name)
        self.grid_generator = GridGenerator(self.loc_net.out_channels, num_fiducial)
        self.out_channels = in_channels

    def forward(self, image):
        image.stop_gradient = False
        batch_C_prime = self.loc_net(image)
        batch_P_prime = self.grid_generator(batch_C_prime, image.shape[2:])
        batch_P_prime = batch_P_prime.reshape([-1, image.shape[2], image.shape[3], 2])
        if torch.__version__ < '1.3.0':
            batch_I_r = F.grid_sample(image, grid=batch_P_prime)
        else:
            batch_I_r = F.grid_sample(image, grid=batch_P_prime, align_corners=True)
        return batch_I_r


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASFBlock,
     lambda: ([], {'in_channels': 4, 'inter_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (AttentionGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_embeddings': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (AttentionHead,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Backbone,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (BasicStem,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (BiaffineAttention,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CTCHead,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ClsHead,
     lambda: ([], {'in_channels': 4, 'class_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvNormLayer,
     lambda: ([], {'ch_in': 4, 'ch_out': 4, 'filter_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv_BN_ReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DBHead,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DSConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeConvBNLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropBlock,
     lambda: ([], {'block_size': 4, 'keep_prob': 4, 'name': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {'n_layer': 1, 'n_head': 4, 'd_key': 4, 'd_value': 4, 'd_model': 4, 'd_inner_hid': 4, 'prepostprocess_dropout': 0.5, 'attention_dropout': 0.5, 'relu_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (EncoderLayer,
     lambda: ([], {'n_head': 4, 'd_key': 4, 'd_value': 4, 'd_model': 4, 'd_inner_hid': 4, 'prepostprocess_dropout': 0.5, 'attention_dropout': 0.5, 'relu_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (EncoderWithFC,
     lambda: ([], {'in_channels': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderWithRNN,
     lambda: ([], {'in_channels': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (EncoderWithRNN_,
     lambda: ([], {'in_channels': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (FFN,
     lambda: ([], {'d_inner_hid': 4, 'd_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FrozenBatchNorm,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Head,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Lambda,
     lambda: ([], {'func': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LastLevelMaxPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayoutLMPooler,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayoutXLMOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayoutXLMPooler,
     lambda: ([], {'hidden_size': 4, 'with_pool': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayoutXLMSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LocalizationNetwork,
     lambda: ([], {'in_channels': 4, 'num_fiducial': 4, 'loc_lr': 4, 'model_name': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (MTB,
     lambda: ([], {'cnn_num': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'d_key': 4, 'd_value': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 1, 4]), torch.rand([4, 4, 1, 4]), torch.rand([4, 4, 1, 4]), torch.rand([4, 4])], {}),
     False),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PSEHead,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PTAttentionGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_embeddings': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (PTAttentionHead,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'dropout': 0.5, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding_2d,
     lambda: ([], {'dropout': 0.5, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PrePostProcessLayer,
     lambda: ([], {'process_cmd': [4, 4], 'd_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PrepareEncoder,
     lambda: ([], {'src_vocab_size': 4, 'src_emb_dim': 4, 'src_max_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RSELayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SARHead,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SASTHead,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SAST_Header1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SAST_Header2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEModule,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ShortCut,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'name': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SubSample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TPS,
     lambda: ([], {'in_channels': 4, 'num_fiducial': 4, 'loc_lr': 4, 'model_name': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_frotms_PaddleOCR2Pytorch(_paritybench_base):
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

    def test_053(self):
        self._check(*TESTCASES[53])

    def test_054(self):
        self._check(*TESTCASES[54])

    def test_055(self):
        self._check(*TESTCASES[55])

    def test_056(self):
        self._check(*TESTCASES[56])

    def test_057(self):
        self._check(*TESTCASES[57])

    def test_058(self):
        self._check(*TESTCASES[58])

