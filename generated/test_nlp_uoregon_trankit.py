import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
trankit = _module
adapter_transformers = _module
activations = _module
adapter_bert = _module
adapter_config = _module
adapter_model_mixin = _module
adapter_modeling = _module
adapter_training = _module
adapter_utils = _module
benchmark = _module
benchmark = _module
benchmark_args = _module
benchmark_args_utils = _module
benchmark_utils = _module
commands = _module
convert = _module
download = _module
env = _module
run = _module
serving = _module
train = _module
transformers_cli = _module
user = _module
configuration_albert = _module
configuration_auto = _module
configuration_bart = _module
configuration_bert = _module
configuration_camembert = _module
configuration_ctrl = _module
configuration_distilbert = _module
configuration_electra = _module
configuration_encoder_decoder = _module
configuration_flaubert = _module
configuration_gpt2 = _module
configuration_longformer = _module
configuration_marian = _module
configuration_mmbt = _module
configuration_openai = _module
configuration_reformer = _module
configuration_roberta = _module
configuration_t5 = _module
configuration_transfo_xl = _module
configuration_utils = _module
configuration_xlm = _module
configuration_xlm_roberta = _module
configuration_xlnet = _module
convert_albert_original_tf_checkpoint_to_pytorch = _module
convert_bart_original_pytorch_checkpoint_to_pytorch = _module
convert_bert_original_tf_checkpoint_to_pytorch = _module
convert_bert_pytorch_checkpoint_to_original_tf = _module
convert_dialogpt_original_pytorch_checkpoint_to_pytorch = _module
convert_electra_original_tf_checkpoint_to_pytorch = _module
convert_gpt2_original_tf_checkpoint_to_pytorch = _module
convert_graph_to_onnx = _module
convert_longformer_original_pytorch_lightning_to_pytorch = _module
convert_marian_to_pytorch = _module
convert_openai_original_tf_checkpoint_to_pytorch = _module
convert_pytorch_checkpoint_to_tf2 = _module
convert_reformer_trax_checkpoint_to_pytorch = _module
convert_roberta_original_pytorch_checkpoint_to_pytorch = _module
convert_t5_original_tf_checkpoint_to_pytorch = _module
convert_transfo_xl_original_tf_checkpoint_to_pytorch = _module
convert_xlm_original_pytorch_checkpoint_to_pytorch = _module
convert_xlnet_original_tf_checkpoint_to_pytorch = _module
data = _module
data_collator = _module
datasets = _module
glue = _module
language_modeling = _module
metrics = _module
squad_metrics = _module
processors = _module
squad = _module
utils = _module
xnli = _module
file_utils = _module
hf_api = _module
hf_argparser = _module
modelcard = _module
modeling_albert = _module
modeling_auto = _module
modeling_bart = _module
modeling_bert = _module
modeling_camembert = _module
modeling_ctrl = _module
modeling_distilbert = _module
modeling_electra = _module
modeling_encoder_decoder = _module
modeling_flaubert = _module
modeling_gpt2 = _module
modeling_longformer = _module
modeling_marian = _module
modeling_mmbt = _module
modeling_openai = _module
modeling_reformer = _module
modeling_roberta = _module
modeling_t5 = _module
modeling_tf_albert = _module
modeling_tf_auto = _module
modeling_tf_bert = _module
modeling_tf_camembert = _module
modeling_tf_ctrl = _module
modeling_tf_distilbert = _module
modeling_tf_electra = _module
modeling_tf_flaubert = _module
modeling_tf_gpt2 = _module
modeling_tf_openai = _module
modeling_tf_pytorch_utils = _module
modeling_tf_roberta = _module
modeling_tf_t5 = _module
modeling_tf_transfo_xl = _module
modeling_tf_transfo_xl_utilities = _module
modeling_tf_utils = _module
modeling_tf_xlm = _module
modeling_tf_xlm_roberta = _module
modeling_tf_xlnet = _module
modeling_transfo_xl = _module
modeling_transfo_xl_utilities = _module
modeling_utils = _module
modeling_xlm = _module
modeling_xlm_roberta = _module
modeling_xlnet = _module
optimization = _module
optimization_tf = _module
pipelines = _module
tokenization_albert = _module
tokenization_auto = _module
tokenization_bart = _module
tokenization_bert = _module
tokenization_bert_japanese = _module
tokenization_camembert = _module
tokenization_ctrl = _module
tokenization_distilbert = _module
tokenization_electra = _module
tokenization_flaubert = _module
tokenization_gpt2 = _module
tokenization_longformer = _module
tokenization_marian = _module
tokenization_openai = _module
tokenization_reformer = _module
tokenization_roberta = _module
tokenization_t5 = _module
tokenization_transfo_xl = _module
tokenization_utils = _module
tokenization_xlm = _module
tokenization_xlm_roberta = _module
tokenization_xlnet = _module
trainer = _module
trainer_tf = _module
trainer_utils = _module
training_args = _module
training_args_tf = _module
config = _module
iterators = _module
lemmatizer_iterators = _module
mwt_iterators = _module
ner_iterators = _module
tagger_iterators = _module
tokenizer_iterators = _module
layers = _module
crf_layer = _module
seq2seq = _module
models = _module
base_models = _module
classifiers = _module
lemma_model = _module
mwt_model = _module
pipeline = _module
test_all = _module
test_training = _module
tpipeline = _module
base_utils = _module
chuliu_edmonds = _module
conll = _module
mwt_lemma_utils = _module
mwt_utils = _module
seq2seq_utils = _module
seq2seq_vocabs = _module
ner_utils = _module
posdep_utils = _module
scorers = _module
conll18_ud_eval = _module
ner_scorer = _module
tbinfo = _module
tokenizer_utils = _module

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


import math


import torch


import torch.nn.functional as F


from torch import nn


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from abc import ABC


from abc import abstractmethod


from typing import Callable


from typing import List


from typing import Mapping


from typing import Optional


from typing import Tuple


from typing import Union


import inspect


import copy


from collections import defaultdict


from collections import namedtuple


from typing import Iterable


from typing import NamedTuple


from logging import getLogger


import numpy as np


import tensorflow as tf


from typing import Dict


import warnings


import numpy


from typing import Any


from typing import NewType


from torch.nn.utils.rnn import pad_sequence


import time


from enum import Enum


from torch.utils.data.dataset import Dataset


from functools import partial


from functools import wraps


import torch.nn as nn


import random


from torch import Tensor


from torch.nn import functional as F


from functools import reduce


from torch.autograd.function import Function


import re


import functools


from tensorflow.python.keras.saving import hdf5_format


import itertools


from torch import device


from torch import dtype


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


from itertools import chain


from typing import TYPE_CHECKING


from typing import Sequence


from collections import Counter


from collections import OrderedDict


from collections import UserDict


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import SequentialSampler


from copy import deepcopy


from numbers import Number


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        if hidden_act.lower() == 'relu':
            self.f = nn.functional.relu
        elif hidden_act.lower() == 'tanh':
            self.f = torch.tanh
        elif hidden_act.lower() == 'swish':

            def swish(x):
                return x * torch.nn.functional.sigmoid(x)
            self.f = swish
        elif hidden_act.lower() == 'gelu':

            def gelu_new(x):
                """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                    Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            self.f = gelu_new
        elif hidden_act.lower() == 'leakyrelu':
            self.f = nn.functional.leaky_relu
        super().__init__()

    def forward(self, x):
        return self.f(x)


class Adapter(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(self, input_size, down_sample=None, non_linearity='relu', init_bert_weights=True, add_layer_norm_before=True, add_layer_norm_after=False, residual_before_ln=True):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln
        seq_list = []
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2
        seq_list.append(nn.Linear(self.input_size, self.down_sample))
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list.append(self.non_linearity)
        self.adapter_down = nn.Sequential(*seq_list)
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)
        if init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x, residual_input):
        down = self.adapter_down(x)
        up = self.adapter_up(down)
        output = up
        if self.residual_before_ln:
            output = output + residual_input
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)
        if not self.residual_before_ln:
            output = output + residual_input
        return output, down, up

    @staticmethod
    def init_bert_weights(module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertFusion(nn.Module):

    def __init__(self, config):
        super(BertFusion, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.dense_size = int(config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        if not self.config.adapter_fusion['query'] and not self.config.adapter_fusion['key'] and not self.config.adapter_fusion['value']:
            self.dense = nn.Linear(self.dense_size, 1)
        if self.config.adapter_fusion['query']:
            self.query = nn.Linear(int(config.hidden_size), self.dense_size)
            self.query.apply(Adapter.init_bert_weights)
        if self.config.adapter_fusion['key']:
            self.key = nn.Linear(self.dense_size, self.dense_size)
            self.key.apply(Adapter.init_bert_weights)
        if self.config.adapter_fusion['value']:
            self.value = nn.Linear(int(config.hidden_size), int(config.hidden_size), bias=False)
            self.value.apply(Adapter.init_bert_weights)
            if self.config.adapter_fusion['value_initialized']:
                self.value.weight.data = (torch.zeros(int(config.hidden_size), int(config.hidden_size)) + 1e-06).fill_diagonal_(1.0)
        if self.config.adapter_fusion['temperature']:
            self.T = 50.0
        else:
            self.T = 1.0
        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, residual, attention_mask=None):
        if self.config.adapter_fusion['residual_before']:
            value += residual[:, :, None, :].repeat(1, 1, value.size(2), 1)
        if self.config.adapter_fusion['query']:
            query_layer = self.query(query)
        else:
            query_layer = query
        if self.config.adapter_fusion['key']:
            key_layer = self.key(key)
        else:
            key_layer = key
        if self.config.adapter_fusion['value'] and self.config.adapter_fusion['value_before_softmax']:
            value_layer = self.value(value)
        else:
            value_layer = value
        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2, -1)), dim=2)
        attention_scores = self.dropout(attention_scores)
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
        self.T = max(self.T - self.reduction, 1.0)
        if not self.training:
            self.recent_attention = attention_probs.detach().cpu().numpy()
        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2)
        if self.config.adapter_fusion['value'] and not self.config.adapter_fusion['value_before_softmax']:
            context_layer = self.value(context_layer)
        else:
            context_layer = context_layer
        if not self.config.adapter_fusion['residual_before']:
            context_layer += residual
        return context_layer


class AdapterFusionSentLvlDynamic(nn.Module):

    def __init__(self, config, n_tasks):
        super(AdapterFusionSentLvlDynamic, self).__init__()
        self.config = config
        self.dense_size = int(config.hidden_size) // config.text_task_adapter_config['reduction_factor']
        if not self.config.adapter_fusion['query'] and not self.config.adapter_fusion['key'] and not self.config.adapter_fusion['value']:
            self.dense = nn.Linear(self.dense_size, 1)
        if self.config.adapter_fusion['query']:
            self.query = nn.Linear(int(config.hidden_size), self.dense_size)
        if self.config.adapter_fusion['key']:
            self.key = nn.Linear(self.dense_size, self.dense_size)
        if self.config.adapter_fusion['value']:
            self.value = nn.Linear(int(config.hidden_size), int(config.hidden_size))
        if self.config.adapter_fusion['temperature']:
            self.T = 50.0
        else:
            self.T = 1.0
        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, attention_mask):
        attention_mask = (attention_mask == 0).float().squeeze()
        length = torch.sum(attention_mask, dim=1)
        key = key * attention_mask[:, :, None, None].repeat((1, 1, key.size()[-2], key.size()[-1]))
        key_sent = torch.sum(key, dim=1) / length[:, None, None].repeat(1, key.size()[-2], key.size()[-1])
        if self.config.adapter_fusion['query'] and not self.config.adapter_fusion['key'] and not self.config.adapter_fusion['value']:
            query = query * attention_mask[:, :, None].repeat((1, 1, query.size()[-1]))
            query_sent = torch.sum(query, dim=1) / length[:, None].repeat(1, query.size()[-1])
            query_enc = self.query(query_sent)
            scores_t = torch.matmul(key_sent, query_enc[:, :, None]).squeeze(-1)
            probs = nn.Softmax(dim=-1)(scores_t / self.T)
            result = torch.squeeze(torch.matmul(probs[:, None, None, :], value))
        if self.config.adapter_fusion['query'] and self.config.adapter_fusion['key'] and not self.config.adapter_fusion['value']:
            query = query * attention_mask[:, :, None].repeat((1, 1, query.size()[-1]))
            query_sent = torch.sum(query, dim=1) / length[:, None].repeat(1, query.size()[-1])
            query_enc = self.query(query_sent)
            key_enc = self.key(key_sent)
            scores_t = torch.matmul(key_enc, query_enc[:, :, None]).squeeze(-1)
            probs = nn.Softmax(dim=-1)(scores_t / self.T)
            result = torch.squeeze(torch.matmul(probs[:, None, None, :], value))
        if self.config.adapter_fusion['query'] and self.config.adapter_fusion['key'] and self.config.adapter_fusion['value']:
            query = query * attention_mask[:, :, None].repeat((1, 1, query.size()[-1]))
            query_sent = torch.sum(query, dim=1) / length[:, None].repeat(1, query.size()[-1])
            query_enc = self.query(query_sent)
            key_enc = self.key(key_sent)
            value_enc = self.value(value)
            scores_t = torch.matmul(key_enc, query_enc[:, :, None]).squeeze(-1)
            probs = nn.Softmax(dim=-1)(scores_t / self.T)
            result = torch.squeeze(torch.matmul(probs[:, None, None, :], value_enc))
        if not self.config.adapter_fusion['query'] and not self.config.adapter_fusion['key'] and not self.config.adapter_fusion['value']:
            scores = self.dense(key_sent)
            scores_t = scores.transpose(-2, -1)
            probs = nn.Softmax(dim=-1)(scores_t / self.T)
            result = torch.squeeze(torch.matmul(probs.unsqueeze(2), value), dim=2)
        self.T = max(self.T - self.reduction, 1.0)
        return result


def get_subnet_constructor(non_linearity, reduction_factor):

    def subnet(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, dims_in // reduction_factor), Activation_Function_Class(non_linearity), nn.Linear(dims_in // reduction_factor, dims_out))
    return subnet


class NICECouplingBlock(nn.Module):
    """Coupling Block following the NICE design."""

    def __init__(self, dims_in, dims_c=[], non_linearity='relu', reduction_factor=2):
        super().__init__()
        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        assert all([(dims_c[i][1:] == dims_in[0][1:]) for i in range(len(dims_c))]), "Dimensions of input and one or more conditions don't agree."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])
        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.F = subnet_constructor(self.split_len2 + condition_length, self.split_len1)
        self.G = subnet_constructor(self.split_len1 + condition_length, self.split_len2)

    def forward(self, x, c=[], rev=False):
        x1, x2 = x[:, :, :self.split_len1], x[:, :, self.split_len1:]
        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1 = x1 + self.F(x2_c)
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2 = x2 + self.G(y1_c)
        else:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2 = x2 - self.G(x1_c)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1 = x1 - self.F(y2_c)
        return torch.cat((y1, y2), -1)

    def jacobian(self, x, rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, 'Can only use 1 input'
        return input_dims


class GLOWCouplingBlock(nn.Module):
    """Coupling Block following the GLOW design. The only difference to the RealNVP coupling blocks,
    is the fact that it uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate
    subnetworks. This reduces computational cost and speeds up learning.
    clamp:              Soft clamping for the multiplicative component. The amplification or attenuation
                        of each input dimension can be at most ±exp(clamp)."""

    def __init__(self, dims_in, dims_c=[], non_linearity='relu', reduction_factor=2, clamp=5.0):
        super().__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.clamp = clamp
        self.max_s = math.exp(clamp)
        self.min_s = math.exp(-clamp)
        assert all([(tuple(dims_c[i][1:]) == tuple(dims_in[0][1:])) for i in range(len(dims_c))]), f"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])
        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2 * 2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1 * 2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        x1, x2 = x[:, :, :self.split_len1], x[:, :, self.split_len1:]
        if not rev:
            s2, t2 = x1.clone(), x2.clone()
            y1 = self.e(s2) * x1 + t2
            r1 = self.s1(torch.cat([y1, *c], 1) if self.conditional else y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) + torch.sum(self.log_e(s2), dim=tuple(range(1, self.ndims + 1)))
        else:
            r1 = self.s1(torch.cat([x1, *c], 1) if self.conditional else x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)
            r2 = self.s2(torch.cat([y2, *c], 1) if self.conditional else y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = -torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) - torch.sum(self.log_e(s2), dim=tuple(range(1, self.ndims + 1)))
        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims


def gelu_new(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf


def swish(x):
    return x * tf.sigmoid(x)


ACT2FN = {'gelu': gelu_new, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class AlbertAttention(BertSelfAttention):

    def __init__(self, config):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        w = self.dense.weight.t().view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
        b = self.dense.bias
        projected_context_layer = torch.einsum('bfnd,ndh->bfh', context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)


class AlbertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        return (hidden_states,) + attention_output[1:]


class AlbertLayerGroup(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()
        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index])
            hidden_states = layer_output[0]
            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)
            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs


class AlbertTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_attentions = ()
        if self.output_hidden_states:
            all_hidden_states = hidden_states,
        for i in range(self.config.num_hidden_layers):
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
            layer_group_output = self.albert_layer_groups[group_idx](hidden_states, attention_mask, head_mask[group_idx * layers_per_group:(group_idx + 1) * layers_per_group])
            hidden_states = layer_group_output[0]
            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class AlbertMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states
        return prediction_scores


class AlbertSOPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.output_attentions = config.output_attentions
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input, mask, kv=None, cache=None, head_mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)
                    v = torch.cat([v_, v], dim=2)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = k, v
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshape).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        outputs = self.out_lin(context),
        if self.output_attentions:
            outputs = outputs + (weights,)
        return outputs


def point_wise_feed_forward_network(d_model_size, dff):
    return torch.nn.Sequential(torch.nn.Linear(d_model_size, dff), torch.nn.ReLU(), torch.nn.Linear(dff, d_model_size))


class EncoderLayer(torch.nn.Module):

    def __init__(self, d_model_size, num_heads, dff, rate=0.1, output_attentions=False):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads, output_attentions)
        self.ffn = point_wise_feed_forward_network(d_model_size, dff)
        self.layernorm1 = torch.nn.LayerNorm(d_model_size, eps=1e-06)
        self.layernorm2 = torch.nn.LayerNorm(d_model_size, eps=1e-06)
        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask, layer_past=None, attention_mask=None, head_mask=None, use_cache=False):
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(normed, normed, normed, mask, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache)
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output
        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output
        outputs = (out2,) + attn_outputs[1:]
        return outputs


CONFIG_NAME = 'config.json'


def is_tf_available():
    return _tf_available


def is_torch_available():
    return _torch_available


logger = logging.getLogger(__name__)


def http_get(url, temp_file, proxies=None, resume_size=0, user_agent=None):
    ua = 'transformers/{}; python/{}'.format(__version__, sys.version.split()[0])
    if is_torch_available():
        ua += '; torch/{}'.format(torch.__version__)
    if is_tf_available():
        ua += '; tensorflow/{}'.format(tf.__version__)
    if isinstance(user_agent, dict):
        ua += '; ' + '; '.join('{}/{}'.format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += '; ' + user_agent
    headers = {'user-agent': ua}
    if resume_size > 0:
        headers['Range'] = 'bytes=%d-' % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:
        return
    content_length = response.headers.get('Content-Length')
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', unit_scale=True, total=total, initial=resume_size, desc='Downloading', disable=bool(logger.getEffectiveLevel() == logging.NOTSET))
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()
    if url.endswith('.h5'):
        filename += '.h5'
    return filename


def get_from_cache(url, cache_dir=None, force_download=False, proxies=None, etag_timeout=10, resume_download=False, user_agent=None, local_files_only=False) ->Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.

    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    etag = None
    if not local_files_only:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
            if response.status_code == 200:
                etag = response.headers.get('ETag')
        except (EnvironmentError, requests.exceptions.Timeout):
            pass
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [file for file in fnmatch.filter(os.listdir(cache_dir), filename + '.*') if not file.endswith('.json') and not file.endswith('.lock')]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                if local_files_only:
                    raise ValueError("Cannot find the requested files in the cached path and outgoing traffic has been disabled. To enable model look-ups and downloads online, set 'local_files_only' to False.")
                return None
    if os.path.exists(cache_path) and not force_download:
        return cache_path
    lock_path = cache_path + '.lock'
    with FileLock(lock_path):
        if os.path.exists(cache_path) and not force_download:
            return cache_path
        if resume_download:
            incomplete_path = cache_path + '.incomplete'

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, 'a+b') as f:
                    yield f
            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir, delete=False)
            resume_size = 0
        with temp_file_manager() as temp_file:
            logger.info('%s not found in cache or force_download set to True, downloading to %s', url, temp_file.name)
            http_get(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)
        logger.info('storing %s in cache at %s', url, cache_path)
        os.replace(temp_file.name, cache_path)
        logger.info('creating metadata file for %s', cache_path)
        meta = {'url': url, 'etag': etag}
        meta_path = cache_path + '.json'
        with open(meta_path, 'w') as meta_file:
            json.dump(meta, meta_file)
    return cache_path


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ('http', 'https')


def cached_path(url_or_filename, cache_dir=None, force_download=False, proxies=None, resume_download=False, user_agent=None, extract_compressed_file=False, force_extract=False, local_files_only=False) ->Optional[str]:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletly recieved file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and overide the folder where it was extracted.

    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if is_remote_url(url_or_filename):
        output_path = get_from_cache(url_or_filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, user_agent=user_agent, local_files_only=local_files_only)
    elif os.path.exists(url_or_filename):
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == '':
        raise EnvironmentError('file {} not found'.format(url_or_filename))
    else:
        raise ValueError('unable to parse {} as a URL or as a local path'.format(url_or_filename))
    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace('.', '-') + '-extracted'
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)
        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted
        lock_path = output_path + '.lock'
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, 'r') as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError('Archive format of {} could not be identified'.format(output_path))
        return output_path_extracted
    return output_path


CLOUDFRONT_DISTRIB_PREFIX = 'https://cdn.huggingface.co'


S3_BUCKET_PREFIX = 'https://s3.amazonaws.com/models.huggingface.co/bert'


def hf_bucket_url(model_id: str, filename: str, use_cdn=True) ->str:
    """
    Resolve a model identifier, and a file name, to a HF-hosted url
    on either S3 or Cloudfront (a Content Delivery Network, or CDN).

    Cloudfront is replicated over the globe so downloads are way faster
    for the end user (and it also lowers our bandwidth costs). However, it
    is more aggressively cached by default, so may not always reflect the
    latest changes to the underlying file (default TTL is 24 hours).

    In terms of client-side caching from this library, even though
    Cloudfront relays the ETags from S3, using one or the other
    (or switching from one to the other) will affect caching: cached files
    are not shared between the two because the cached file's name contains
    a hash of the url.
    """
    endpoint = CLOUDFRONT_DISTRIB_PREFIX if use_cdn else S3_BUCKET_PREFIX
    legacy_format = '/' not in model_id
    if legacy_format:
        return f'{endpoint}/{model_id}-{filename}'
    else:
        return f'{endpoint}/{model_id}/{filename}'


class PretrainedConfig(object):
    """ Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods for loading/downloading/saving configurations.

        Note:
            A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does **not** load the model weights.
            It only affects the model's configuration.

        Class attributes (overridden by derived classes):
            - ``model_type``: a string that identifies the model type, that we serialize into the JSON file, and that we use to recreate the correct object in :class:`~transformers.AutoConfig`.

        Args:
            finetuning_task (:obj:`string` or :obj:`None`, `optional`, defaults to :obj:`None`):
                Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.
            num_labels (:obj:`int`, `optional`, defaults to `2`):
                Number of classes to use when the model is a classification model (sequences/tokens)
            output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Should the model returns attentions weights.
            output_hidden_states (:obj:`string`, `optional`, defaults to :obj:`False`):
                Should the model returns all hidden-states.
            torchscript (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Is the model used with Torchscript (for PyTorch models).
    """
    model_type: str = ''

    def __init__(self, **kwargs):
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.use_cache = kwargs.pop('use_cache', True)
        self.torchscript = kwargs.pop('torchscript', False)
        self.use_bfloat16 = kwargs.pop('use_bfloat16', False)
        self.pruned_heads = kwargs.pop('pruned_heads', {})
        self.is_encoder_decoder = kwargs.pop('is_encoder_decoder', False)
        self.is_decoder = kwargs.pop('is_decoder', False)
        self.max_length = kwargs.pop('max_length', 20)
        self.min_length = kwargs.pop('min_length', 0)
        self.do_sample = kwargs.pop('do_sample', False)
        self.early_stopping = kwargs.pop('early_stopping', False)
        self.num_beams = kwargs.pop('num_beams', 1)
        self.temperature = kwargs.pop('temperature', 1.0)
        self.top_k = kwargs.pop('top_k', 50)
        self.top_p = kwargs.pop('top_p', 1.0)
        self.repetition_penalty = kwargs.pop('repetition_penalty', 1.0)
        self.length_penalty = kwargs.pop('length_penalty', 1.0)
        self.no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', 0)
        self.bad_words_ids = kwargs.pop('bad_words_ids', None)
        self.num_return_sequences = kwargs.pop('num_return_sequences', 1)
        self.architectures = kwargs.pop('architectures', None)
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.id2label = kwargs.pop('id2label', None)
        self.label2id = kwargs.pop('label2id', None)
        if self.id2label is not None:
            kwargs.pop('num_labels', None)
            self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        else:
            self.num_labels = kwargs.pop('num_labels', 2)
        self.prefix = kwargs.pop('prefix', None)
        self.bos_token_id = kwargs.pop('bos_token_id', None)
        self.pad_token_id = kwargs.pop('pad_token_id', None)
        self.eos_token_id = kwargs.pop('eos_token_id', None)
        self.decoder_start_token_id = kwargs.pop('decoder_start_token_id', None)
        self.task_specific_params = kwargs.pop('task_specific_params', None)
        self.xla_device = kwargs.pop('xla_device', None)
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @property
    def num_labels(self):
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels):
        self.id2label = {i: 'LABEL_{}'.format(i) for i in range(num_labels)}
        self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    def save_pretrained(self, save_directory):
        """
        Save a configuration object to the directory `save_directory`, so that it
        can be re-loaded using the :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`string`):
                Directory where the configuration JSON file will be saved.
        """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file, use_diff=True)
        logger.info('Configuration saved in {}'.format(output_config_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) ->'PretrainedConfig':
        """

        Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pre-trained model configuration.

        Args:
            pretrained_model_name_or_path (:obj:`string`):
                either:
                  - a string with the `shortcut name` of a pre-trained model configuration to load from cache or
                    download, e.g.: ``bert-base-uncased``.
                  - a string with the `identifier name` of a pre-trained model configuration that was user-uploaded to
                    our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                  - a path to a `directory` containing a configuration file saved using the
                    :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                  - a path or url to a saved configuration JSON `file`, e.g.:
                    ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`string`, `optional`):
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            kwargs (:obj:`Dict[str, any]`, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is
                controlled by the `return_unused_kwargs` keyword parameter.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Force to (re-)download the model weights and configuration files and override the cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.
            proxies (:obj:`Dict`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.:
                :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`
                The proxies are used on each request.
            return_unused_kwargs: (`optional`) bool:
                If False, then this function returns just the final configuration object.
                If True, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs` is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part
                of kwargs which has not been used to update `config` and is otherwise ignored.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) ->Tuple[Dict, Dict]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used
        for instantiating a Config using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (:obj:`string`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        local_files_only = kwargs.pop('local_files_only', False)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            config_file = hf_bucket_url(pretrained_model_name_or_path, filename=CONFIG_NAME, use_cdn=False)
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only)
            if resolved_config_file is None:
                raise EnvironmentError
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except EnvironmentError:
            msg = f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file\n\n"
            raise EnvironmentError(msg)
        except json.JSONDecodeError:
            msg = "Couldn't reach server at '{}' to download configuration file or configuration file is not a valid JSON file. Please check network or file content here: {}.".format(config_file, resolved_config_file)
            raise EnvironmentError(msg)
        if resolved_config_file == config_file:
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading configuration file {} from cache at {}'.format(config_file, resolved_config_file))
        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) ->'PretrainedConfig':
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
            kwargs (:obj:`Dict[str, any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object
        """
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
        config = cls(**config_dict)
        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        logger.info('Model config %s', str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: str) ->'PretrainedConfig':
        """
        Constructs a `Config` from the path to a json file of parameters.

        Args:
            json_file (:obj:`string`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return '{} {}'.format(self.__class__.__name__, self.to_json_string())

    def to_diff_dict(self):
        """
        Removes all attributes from config which correspond to the default
        config attributes for better readability and serializes to a Python
        dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()
        default_config_dict = PretrainedConfig().to_dict()
        serializable_config_dict = {}
        for key, value in config_dict.items():
            if key not in default_config_dict or value != default_config_dict[key]:
                serializable_config_dict[key] = value
        return serializable_config_dict

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, 'model_type'):
            output['model_type'] = self.__class__.model_type
        if hasattr(self, 'adapters'):
            output['adapters'] = self.adapters.to_dict()
        return output

    def to_json_string(self, use_diff=True):
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`):
                If set to True, only the difference between the config instance and the default PretrainedConfig() is serialized to JSON string.

        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True, cls=DataclassJSONEncoder) + '\n'

    def to_json_file(self, json_file_path, use_diff=True):
        """
        Save this instance to a json file.

        Args:
            json_file_path (:obj:`string`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`):
                If set to True, only the difference between the config instance and the default PretrainedConfig() is serialized to JSON file.
        """
        with open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: Dict):
        """
        Updates attributes of this class
        with attributes from `config_dict`.

        Args:
            :obj:`Dict[str, any]`: Dictionary of attributes that shall be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)


class BartConfig(PretrainedConfig):
    """
        Configuration class for Bart. Parameters are renamed from the fairseq implementation
    """
    model_type = 'bart'

    def __init__(self, activation_dropout=0.0, activation_function='gelu', vocab_size=50265, d_model=1024, encoder_ffn_dim=4096, encoder_layers=12, encoder_attention_heads=16, decoder_ffn_dim=4096, decoder_layers=12, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, attention_dropout=0.0, dropout=0.1, max_position_embeddings=1024, init_std=0.02, classifier_dropout=0.0, num_labels=3, is_encoder_decoder=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, normalize_before=False, add_final_layer_norm=False, scale_embedding=False, normalize_embedding=True, static_position_embeddings=False, add_bias_logits=False, **common_kwargs):
        """
            :class:`~transformers.BartConfig` is the configuration class for `BartModel`.
            Examples:
                config = BartConfig.from_pretrained('bart-large')
                model = BartModel(config)
        """
        if 'hidden_size' in common_kwargs:
            raise ValueError('hidden size is called d_model')
        super().__init__(num_labels=num_labels, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, **common_kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std
        self.activation_function = activation_function
        self.scale_embedding = scale_embedding
        self.normalize_embedding = normalize_embedding
        self.normalize_before = normalize_before
        self.add_final_layer_norm = add_final_layer_norm
        self.add_bias_logits = add_bias_logits
        self.static_position_embeddings = static_position_embeddings
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout
        self.classif_dropout = classifier_dropout

    @property
    def num_attention_heads(self) ->int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) ->int:
        return self.d_model

    def is_valid_mbart(self) ->bool:
        """Is the configuration aligned with the MBART paper."""
        if self.normalize_before and self.add_final_layer_norm and self.scale_embedding:
            return True
        if self.normalize_before or self.add_final_layer_norm or self.scale_embedding:
            logger.info('This configuration is a mixture of MBART and BART settings')
        return False


def LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None
        num_embeddings += padding_idx + 1
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if use_cache:
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)
        else:
            positions = create_position_ids_from_input_ids(input, self.padding_idx)
        return super().forward(positions)


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f'odd embedding_dim {embedding_dim} not supported')
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
            The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)])
        out[:, 0:dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, dim // 2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)
        else:
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


def invert_mask(attention_mask):
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)
            dropout_probability = random.uniform(0, 1)
            if self.training and dropout_probability < self.layerdrop:
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)
            if self.output_attentions:
                all_attentions.append(attn)
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.output_hidden_states:
            encoder_states.append(x)
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)
        return x, encoder_states, all_attentions


class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = 'encoder_decoder' if self.encoder_decoder_attention else 'self'

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(self, query, key: Optional[Tensor], key_padding_mask: Optional[Tensor]=None, layer_state: Optional[Dict[str, Optional[Tensor]]]=None, attn_mask: Optional[Tensor]=None, need_weights=False) ->Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if layer_state is not None:
            saved_state = layer_state.get(self.cache_key, {})
            if 'prev_key' in saved_state:
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}
        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)
        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)
        layer_state[self.cache_key] = {'prev_key': k.view(bsz, self.num_heads, -1, self.head_dim), 'prev_value': v.view(bsz, self.num_heads, -1, self.head_dim), 'prev_key_padding_mask': key_padding_mask if not static_kv else None}
        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)
        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        if 'prev_key' in saved_state:
            _prev_key = saved_state['prev_key']
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if 'prev_value' in saved_state:
            _prev_value = saved_state['prev_value']
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get('prev_key_padding_mask', None)
        key_padding_mask = self._cat_prev_key_padding_mask(key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv)
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(key_padding_mask: Optional[Tensor], prev_key_padding_mask: Optional[Tensor], batch_size: int, src_len: int, static_kv: bool) ->Optional[Tensor]:
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


class DecoderLayer(nn.Module):

    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, encoder_decoder_attention=True)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_hidden_states, encoder_attn_mask=None, layer_state=None, causal_mask=None, decoder_padding_mask=None):
        residual = x
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, self_attn_weights = self.self_attn(query=x, key=x, layer_state=layer_state, key_padding_mask=decoder_padding_mask, attn_mask=causal_mask, need_weights=self.output_attentions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(query=x, key=encoder_hidden_states, key_padding_mask=encoder_attn_mask, layer_state=layer_state)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, self_attn_weights, layer_state


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(config.max_position_embeddings, config.d_model, config.pad_token_id)
        else:
            self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, config.d_model, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(self, input_ids, encoder_hidden_states, encoder_padding_mask, decoder_padding_mask, decoder_causal_mask, decoder_cached_states=None, use_cache=False, **unused):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)
        positions = self.embed_positions(input_ids, use_cache=use_cache)
        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]
        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states += x,
            dropout_probability = random.uniform(0, 1)
            if self.training and dropout_probability < self.layerdrop:
                continue
            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None
            x, layer_self_attn, layer_past = decoder_layer(x, encoder_hidden_states, encoder_attn_mask=encoder_padding_mask, decoder_padding_mask=decoder_padding_mask, layer_state=layer_state, causal_mask=decoder_causal_mask)
            if use_cache:
                next_decoder_cache.append(layer_past.copy())
            if self.layer_norm and idx == len(self.layers) - 1:
                x = self.layer_norm(x)
            if self.output_attentions:
                all_self_attns += layer_self_attn,
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        if use_cache:
            next_cache = (encoder_hidden_states, encoder_padding_mask), next_decoder_cache
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AdapterType(str, Enum):
    """Models all currently available model adapter types."""
    text_task = 'text_task'
    text_lang = 'text_lang'

    @classmethod
    def has(cls, value):
        return value in cls.__members__.values()

    def __repr__(self):
        return self.value


def parse_adapter_names(adapter_names: list) ->List[List[str]]:
    if adapter_names is not None:
        if isinstance(adapter_names, str):
            adapter_names = [[adapter_names]]
        elif isinstance(adapter_names, list):
            if isinstance(adapter_names[0], str):
                adapter_names = [adapter_names]
        if not isinstance(adapter_names[0][0], str):
            raise ValueError('Adapter names %s not set correctly', str(adapter_names))
    return adapter_names


class BertSelfOutputAdaptersMixin:
    """Adds adapters to the BertSelfOutput module.
    """

    def _init_adapter_modules(self):
        self.attention_text_task_adapters = nn.ModuleDict(dict())
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        self.attention_text_lang_adapters = nn.ModuleDict(dict())

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config['mh_adapter']:
            adapter = Adapter(input_size=self.config.hidden_size, down_sample=self.config.hidden_size // adapter_config['reduction_factor'], add_layer_norm_before=adapter_config['ln_before'], add_layer_norm_after=adapter_config['ln_after'], non_linearity=adapter_config['non_linearity'], residual_before_ln=adapter_config['adapter_residual_before_ln'])
            if adapter_type == AdapterType.text_task:
                self.attention_text_task_adapters[adapter_name] = adapter
            elif adapter_type == AdapterType.text_lang:
                self.attention_text_lang_adapters[adapter_name] = adapter
            else:
                raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_attention_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(',')
        if self.config.adapters.common_config_value(adapter_names, 'mh_adapter'):
            self.adapter_fusion_layer[','.join(adapter_names)] = BertFusion(self.config)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_fusion: bool):
        """Unfreezes a given list of adapters, the adapter fusion layer, or both

        :param adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
        :param unfreeze_adapters: whether the adapters themselves should be unfreezed
        :param unfreeze_fusion: whether the adapter attention layer for the given adapters should be unfreezed
        """
        if unfreeze_adapters:
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]
            for adapter_name in adapter_names:
                layer = self.get_adapter_layer(adapter_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_names[0], str):
                adapter_names = [adapter_names]
            for adapter_fusion_group in adapter_names:
                fusion_name = ','.join(adapter_fusion_group)
                if fusion_name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[fusion_name].parameters():
                        param.requires_grad = True

    def get_adapter_preparams(self, adapter_config, hidden_states, input_tensor):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration
        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None
        if adapter_config['residual_before_ln']:
            residual = hidden_states
        if hasattr(self.config, 'adapter_fusion') and self.config.adapter_fusion['query_before_ln']:
            query = hidden_states
        if adapter_config['original_ln_before']:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        if not adapter_config['residual_before_ln']:
            residual = hidden_states
        if hasattr(self.config, 'adapter_fusion') and not self.config.adapter_fusion['query_before_ln']:
            query = hidden_states
        return hidden_states, query, residual

    def get_adapter_layer(self, adapter_name):
        """
        Depending on the adapter type we retrieve the correct layer. If no adapter for that name was set at that layer
        we return None
        Args:
            adapter_name: string name of the adapter

        Returns: layer | None

        """
        if adapter_name in self.attention_text_lang_adapters:
            return self.attention_text_lang_adapters[adapter_name]
        if adapter_name in self.attention_text_task_adapters:
            return self.attention_text_task_adapters[adapter_name]
        return None

    def adapter_stack_layer(self, hidden_states, input_tensor, attention_mask, adapter_stack):
        """
        One layer of stacked adapters. This either passes through a single adapter and prepares the data to be passed
        into a subsequent adapter, or the next transformer layer
        OR
        IFF more than one adapter names is set for one stack layer, we assume that fusion is activated. Thus, the
        adapters are fused together.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            input_tensor: residual connection of transformer
            attention_mask: attention mask on token level
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters

        Returns: hidden_states

        """
        adapter_config = self.config.adapters.get(adapter_stack[0])
        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)
        if len(adapter_stack) == 1:
            adapter_layer = self.get_adapter_layer(adapter_stack[0])
            if adapter_layer is not None:
                hidden_states, _, _ = adapter_layer(hidden_states, residual_input=residual)
            return hidden_states
        else:
            return self.adapter_fusion(hidden_states, attention_mask, adapter_stack, residual, query)

    def adapter_fusion(self, hidden_states, attention_mask, adapter_stack, residual, query):
        """
        If more than one adapter name is set for a stack layer, we fuse the adapters.
        For this, we pass through every adapter and learn an attention-like weighting of each adapter.
        The information stored in each of the adapters is thus fused together wrt the current example.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            attention_mask: attention mask on token level
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters
            residual: residual of the previous layer
            query: query by which we attend over the adapters

        Returns: hidden_states

        """
        up_list = []
        for adapter_name in adapter_stack:
            adapter_layer = self.get_adapter_layer(adapter_name)
            if adapter_layer is not None:
                intermediate_output, _, up = adapter_layer(hidden_states, residual_input=residual)
                up_list.append(up)
        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)
            fusion_name = ','.join(adapter_stack)
            hidden_states = self.adapter_fusion_layer[fusion_name](query, up_list, up_list, residual=residual, attention_mask=attention_mask)
        return hidden_states

    def adapters_forward(self, hidden_states, input_tensor, attention_mask, adapter_names=None):
        if adapter_names is not None:
            adapter_names = parse_adapter_names(adapter_names)
            flat_adapter_names = [item for sublist in adapter_names for item in sublist]
        if adapter_names is not None and len((set(self.attention_text_task_adapters.keys()) | set(self.attention_text_lang_adapters.keys())) & set(flat_adapter_names)) > 0:
            for adapter_stack in adapter_names:
                hidden_states = self.adapter_stack_layer(hidden_states=hidden_states, input_tensor=input_tensor, attention_mask=attention_mask, adapter_stack=adapter_stack)
            last_config = self.config.adapters.get(adapter_names[-1][-1])
            if last_config['original_ln_after']:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutput(nn.Module, BertSelfOutputAdaptersMixin):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_adapter_modules()

    def forward(self, hidden_states, input_tensor, attention_mask=None, adapter_names=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapters_forward(hidden_states, input_tensor, attention_mask=attention_mask, adapter_names=adapter_names)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, adapter_names=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states, attention_mask=attention_mask, adapter_names=adapter_names)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutputAdaptersMixin:
    """Adds adapters to the BertOutput module.
    """

    def _init_adapter_modules(self):
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        self.layer_text_task_adapters = nn.ModuleDict(dict())
        self.layer_text_lang_adapters = nn.ModuleDict(dict())

    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_fusion_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(',')
        if self.config.adapters.common_config_value(adapter_names, 'output_adapter'):
            self.adapter_fusion_layer[','.join(adapter_names)] = BertFusion(self.config)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config['output_adapter']:
            adapter = Adapter(input_size=self.config.hidden_size, down_sample=self.config.hidden_size // adapter_config['reduction_factor'], add_layer_norm_before=adapter_config['ln_before'], add_layer_norm_after=adapter_config['ln_after'], non_linearity=adapter_config['non_linearity'], residual_before_ln=adapter_config['adapter_residual_before_ln'])
            if adapter_type == AdapterType.text_task:
                self.layer_text_task_adapters[adapter_name] = adapter
            elif adapter_type == AdapterType.text_lang:
                self.layer_text_lang_adapters[adapter_name] = adapter
            else:
                raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]
            for adapter_name in adapter_names:
                layer = self.get_adapter_layer(adapter_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_names[0], str):
                adapter_names = [adapter_names]
            for adapter_fusion_group in adapter_names:
                fusion_name = ','.join(adapter_fusion_group)
                if fusion_name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[fusion_name].parameters():
                        param.requires_grad = True

    def get_adapter_preparams(self, adapter_config, hidden_states, input_tensor):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration
        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None
        if adapter_config['residual_before_ln']:
            residual = hidden_states
        if hasattr(self.config, 'adapter_fusion') and self.config.adapter_fusion['query_before_ln']:
            query = hidden_states
        if adapter_config['original_ln_before']:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        if not adapter_config['residual_before_ln']:
            residual = hidden_states
        if hasattr(self.config, 'adapter_fusion') and not self.config.adapter_fusion['query_before_ln']:
            query = hidden_states
        return hidden_states, query, residual

    def get_adapter_layer(self, adapter_name):
        """
        Depending on the adapter type we retrieve the correct layer. If no adapter for that name was set at that layer
        we return None
        Args:
            adapter_name: string name of the adapter

        Returns: layer | None

        """
        if adapter_name in self.layer_text_lang_adapters:
            return self.layer_text_lang_adapters[adapter_name]
        if adapter_name in self.layer_text_task_adapters:
            return self.layer_text_task_adapters[adapter_name]
        return None

    def adapter_stack_layer(self, hidden_states, input_tensor, attention_mask, adapter_stack):
        """
        One layer of stacked adapters. This either passes through a single adapter and prepares the data to be passed
        into a subsequent adapter, or the next transformer layer
        OR
        IFF more than one adapter names is set for one stack layer, we assume that fusion is activated. Thus, the
        adapters are fused together.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            input_tensor: residual connection of transformer
            attention_mask: attention mask on token level
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters

        Returns: hidden_states

        """
        adapter_config = self.config.adapters.get(adapter_stack[0])
        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)
        if len(adapter_stack) == 1:
            adapter_layer = self.get_adapter_layer(adapter_stack[0])
            if adapter_layer is not None:
                hidden_states, _, _ = adapter_layer(hidden_states, residual_input=residual)
            return hidden_states
        else:
            return self.adapter_fusion(hidden_states, attention_mask, adapter_stack, residual, query)

    def adapter_fusion(self, hidden_states, attention_mask, adapter_stack, residual, query):
        """
        If more than one adapter name is set for a stack layer, we fuse the adapters.
        For this, we pass through every adapter and learn an attention-like weighting of each adapter.
        The information stored in each of the adapters is thus fused together wrt the current example.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            attention_mask: attention mask on token level
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters
            residual: residual of the previous layer
            query: query by which we attend over the adapters

        Returns: hidden_states

        """
        up_list = []
        for adapter_name in adapter_stack:
            adapter_layer = self.get_adapter_layer(adapter_name)
            if adapter_layer is not None:
                intermediate_output, _, up = adapter_layer(hidden_states, residual_input=residual)
                up_list.append(up)
        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)
            fusion_name = ','.join(adapter_stack)
            hidden_states = self.adapter_fusion_layer[fusion_name](query, up_list, up_list, residual=residual, attention_mask=attention_mask)
        return hidden_states

    def adapters_forward(self, hidden_states, input_tensor, attention_mask, adapter_names=None):
        if adapter_names is not None:
            adapter_names = parse_adapter_names(adapter_names)
            flat_adapter_names = [item for sublist in adapter_names for item in sublist]
        if adapter_names is not None and len((set(self.layer_text_lang_adapters.keys()) | set(self.layer_text_task_adapters.keys())) & set(flat_adapter_names)) > 0:
            for adapter_stack in adapter_names:
                hidden_states = self.adapter_stack_layer(hidden_states=hidden_states, input_tensor=input_tensor, attention_mask=attention_mask, adapter_stack=adapter_stack)
            last_config = self.config.adapters.get(adapter_names[-1][-1])
            if last_config['original_ln_after']:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertOutput(nn.Module, BertOutputAdaptersMixin):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_adapter_modules()

    def forward(self, hidden_states, input_tensor, attention_mask, adapter_names=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapters_forward(hidden_states, input_tensor, attention_mask, adapter_names)
        return hidden_states


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module.
    """

    def add_fusion_layer(self, adapter_names):
        self.attention.output.add_fusion_layer(adapter_names)
        self.output.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        self.attention.output.add_adapter(adapter_name, adapter_type)
        self.output.add_adapter(adapter_name, adapter_type)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        self.attention.output.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)
        self.output.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)


class BertLayer(BertLayerAdaptersMixin, nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, adapter_names=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, adapter_names=adapter_names)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, attention_mask, adapter_names=adapter_names)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoderAdaptersMixin:
    """Adds adapters to the BertEncoder module.
    """

    def add_fusion_layer(self, adapter_names):
        for layer in self.layer:
            layer.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        leave_out = adapter_config.get('leave_out', [])
        for i, layer in enumerate(self.layer):
            if i not in leave_out:
                layer.add_adapter(adapter_name, adapter_type)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        for layer in self.layer:
            layer.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)


class BertEncoder(BertEncoderAdaptersMixin, nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, adapter_names=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, adapter_names=adapter_names)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states, inv_lang_adapter=None):
        hidden_states = self.transform(hidden_states)
        if inv_lang_adapter:
            hidden_states = inv_lang_adapter(hidden_states, rev=True)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output, inv_lang_adapter=None):
        prediction_scores = self.predictions(sequence_output, inv_lang_adapter)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, inv_lang_adapter=None):
        prediction_scores = self.predictions(sequence_output, inv_lang_adapter)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class Embeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.output_attentions = config.output_attentions
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, query, key, value, mask, head_mask=None):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads
        mask_reshp = bs, 1, 1, k_length

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshp).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = nn.Softmax(dim=-1)(scores)
        weights = self.dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        context = self.out_lin(context)
        if self.output_attentions:
            return context, weights
        else:
            return context,


def gelu(x):
    """ Implementation of the gelu activation function.
        XLNet is using OpenAI GPT's gelu
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf


class FFN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in ['relu', 'gelu'], "activation ({}) must be in ['relu', 'gelu']".format(config.activation)
        self.activation = gelu if config.activation == 'gelu' else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        assert config.dim % config.n_heads == 0
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask, head_mask=head_mask)
        if self.output_attentions:
            sa_output, sa_weights = sa_output
        else:
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)
        ffn_output = self.ffn(sa_output)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)
        output = ffn_output,
        if self.output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = ()
        all_attentions = ()
        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            layer_outputs = layer_module(x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i])
            hidden_state = layer_outputs[-1]
            if self.output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        outputs = hidden_state,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class ElectraEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = BertLayerNorm(config.embedding_size, eps=config.layer_norm_eps)


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError('function {} not found in ACT2FN mapping {}'.format(activation_string, list(ACT2FN.keys())))


class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states, attention_mask):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()
        return logits


class ElectraGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = BertLayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation('gelu')(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation('gelu')(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Conv1D(nn.Module):

    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


def prune_conv1d_layer(layer, index, dim=1):
    """ Prune a Conv1D layer (a model parameters) to keep only entries in index.
        A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0])
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


class Attention(nn.Module):

    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = config.output_attentions
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + 2 * self.split_size])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        self.split_size = self.split_size // self.n_head * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.bias[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -10000.0 * (1 - b)
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        if head_mask is not None:
            w = w * head_mask
        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = [a] + attn_outputs[1:]
        return outputs


ACT_FNS = {'relu': nn.ReLU, 'swish': swish, 'gelu': gelu_new}


class MLP(nn.Module):

    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):

    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None):
        attn_outputs = self.attn(x, attention_mask=attention_mask, head_mask=head_mask)
        a = attn_outputs[0]
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        outputs = [h] + attn_outputs[1:]
        return outputs


class LongformerSelfAttention(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert attention_window % 2 == 0, f'`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}'
        assert attention_window > 0, f'`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}'
        self.one_sided_attention_window_size = attention_window // 2

    @staticmethod
    def _skew(x, direction):
        """Convert diagonals into columns (or columns into diagonals depending on `direction`"""
        x_padded = F.pad(x, direction)
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded

    @staticmethod
    def _skew2(x):
        """shift every row 1 step to right converting columns into diagonals"""
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1))
        x = x.view(B, C, -1)
        x = x[:, :, :-M]
        x = x.view(B, C, M, M + L)
        x = x[:, :, :, :-1]
        return x

    @staticmethod
    def _chunk(x, w):
        """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    def _mask_invalid_locations(self, input_tensor, w) ->torch.Tensor:
        affected_seqlen = w
        beginning_mask_2d = input_tensor.new_ones(w, w + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        seqlen = input_tensor.size(1)
        beginning_input = input_tensor[:, :affected_seqlen, :, :w + 1]
        beginning_mask = beginning_mask[:, :seqlen].expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float('inf'))
        ending_input = input_tensor[:, -affected_seqlen:, :, -(w + 1):]
        ending_mask = ending_mask[:, -seqlen:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float('inf'))

    def _sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int):
        """Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
        with an overlap of size w"""
        batch_size, seqlen, num_heads, head_dim = q.size()
        assert seqlen % (w * 2) == 0, f'Sequence length should be multiple of {w * 2}. Given {seqlen}'
        assert q.size() == k.size()
        chunks_count = seqlen // w - 1
        q = q.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)
        k = k.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)
        chunk_q = self._chunk(q, w)
        chunk_k = self._chunk(k, w)
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1))
        diagonal_attn = diagonal_chunk_attn.new_empty((batch_size * num_heads, chunks_count + 1, w, w * 2 + 1))
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1):-1, w + 1:]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]
        diagonal_attn = diagonal_attn.view(batch_size, num_heads, seqlen, 2 * w + 1).transpose(2, 1)
        self._mask_invalid_locations(diagonal_attn, w)
        return diagonal_attn

    def _sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        """Same as _sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
        format from _sliding_chunks_matmul_qk"""
        batch_size, seqlen, num_heads, head_dim = v.size()
        assert seqlen % (w * 2) == 0
        assert prob.size()[:3] == v.size()[:3]
        assert prob.size(3) == 2 * w + 1
        chunks_count = seqlen // w - 1
        chunk_prob = prob.transpose(1, 2).reshape(batch_size * num_heads, seqlen // w, w, 2 * w + 1)
        v = v.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)
        padded_v = F.pad(v, (0, 0, w, w), value=-1)
        chunk_v_size = batch_size * num_heads, chunks_count + 1, 3 * w, head_dim
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)
        skewed_prob = self._skew2(chunk_prob)
        context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
        return context.view(batch_size, num_heads, seqlen, head_dim).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        """
        LongformerSelfAttention expects `len(hidden_states)` to be multiple of `attention_window`.
        Padding to `attention_window` happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention

        `encoder_hidden_states` and `encoder_attention_mask` are not supported and should be None
        """
        assert encoder_hidden_states is None, '`encoder_hidden_states` is not supported and should be None'
        assert encoder_attention_mask is None, '`encoder_attention_mask` is not supported and shiould be None'
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0
            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch, device=num_extra_indices_per_batch.device)
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None
        hidden_states = hidden_states.transpose(0, 1)
        seqlen, batch_size, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)
        q = q.view(seqlen, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seqlen, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        attn_weights = self._sliding_chunks_matmul_qk(q, k, self.one_sided_attention_window_size)
        self._mask_invalid_locations(attn_weights, self.one_sided_attention_window_size)
        if remove_from_windowed_attention_mask is not None:
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            ones = float_mask.new_ones(size=float_mask.size())
            d_mask = self._sliding_chunks_matmul_qk(ones, float_mask, self.one_sided_attention_window_size)
            attn_weights += d_mask
        assert list(attn_weights.size()) == [batch_size, seqlen, self.num_heads, self.one_sided_attention_window_size * 2 + 1]
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)
        attn_weights_fp32 = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_fp32.type_as(attn_weights)
        if key_padding_mask is not None:
            attn_weights = torch.masked_fill(attn_weights, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        v = v.view(seqlen, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        attn = None
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()
        if attn is None:
            attn = self._sliding_chunks_matmul_pv(attn_probs, v, self.one_sided_attention_window_size)
        else:
            attn += self._sliding_chunks_matmul_pv(attn_probs, v, self.one_sided_attention_window_size)
        assert attn.size() == (batch_size, seqlen, self.num_heads, self.head_dim), 'Unexpected size'
        attn = attn.transpose(0, 1).reshape(seqlen, batch_size, embed_dim).contiguous()
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, batch_size, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[extra_attention_mask_nonzeros[::-1]]
            q = self.query_global(selected_hidden_states)
            k = self.key_global(hidden_states)
            v = self.value_global(hidden_states)
            q /= math.sqrt(self.head_dim)
            q = q.contiguous().view(max_num_extra_indices_per_batch, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [batch_size * self.num_heads, max_num_extra_indices_per_batch, seqlen]
            attn_weights = attn_weights.view(batch_size, self.num_heads, max_num_extra_indices_per_batch, seqlen)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -10000.0)
            attn_weights = attn_weights.view(batch_size * self.num_heads, max_num_extra_indices_per_batch, seqlen)
            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [batch_size * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]
            selected_attn_4d = selected_attn.view(batch_size, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1)
        context_layer = attn.transpose(0, 1)
        if self.output_attentions:
            if extra_attention_mask is not None:
                attn_weights = attn_weights.view(batch_size, self.num_heads, max_num_extra_indices_per_batch, seqlen)
            else:
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (context_layer, attn_weights) if self.output_attentions else (context_layer,)
        return outputs


class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding.
    """

    def __init__(self, config, encoder, embeddings):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.proj_embeddings = nn.Linear(config.modal_hidden_size, config.hidden_size)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_modal, start_token=None, end_token=None, position_ids=None, token_type_ids=None):
        token_embeddings = self.proj_embeddings(self.encoder(input_modal))
        seq_length = token_embeddings.size(1)
        if start_token is not None:
            start_token_embeds = self.word_embeddings(start_token)
            seq_length += 1
            token_embeddings = torch.cat([start_token_embeds.unsqueeze(1), token_embeddings], dim=1)
        if end_token is not None:
            end_token_embeds = self.word_embeddings(end_token)
            seq_length += 1
            token_embeddings = torch.cat([token_embeddings, end_token_embeds.unsqueeze(1)], dim=1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_modal.device)
            position_ids = position_ids.unsqueeze(0).expand(input_modal.size(0), seq_length)
        if token_type_ids is None:
            token_type_ids = torch.zeros((input_modal.size(0), seq_length), dtype=torch.long, device=input_modal.device)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


MMBT_INPUTS_DOCSTRING = """    Inputs:
        **input_modal**: ``torch.FloatTensor`` of shape ``(batch_size, ***)``:
            The other modality data. It will be the shape that the encoder for that type expects.
            e.g. With an Image Encoder, the shape would be (batch_size, channels, height, width)
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            It does not expect [CLS] token to be added as it's appended to the end of other modality embeddings.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **modal_start_tokens**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Optional start token to be added to Other Modality Embedding. [CLS] Most commonly used for Classification tasks.
        **modal_end_tokens**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Optional end token to be added to Other Modality Embedding. [SEP] Most commonly used.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate different portions of the inputs.
        **modal_token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, modal_sequence_length)``:
            Segment token indices to indicate different portions of the non-text modality.
            The embeddings from these tokens will be summed with the respective token embeddings for the non-text modality.
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
        **modal_position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, modal_sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings for the non-text modality.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        **encoder_hidden_states**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``:
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model
            is configured as a decoder.
        **encoder_attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


MMBT_START_DOCSTRING = """    MMBT model was proposed in
    `Supervised Multimodal Bitransformers for Classifying Images and Text`_
    by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine.
    It's a supervised multimodal bitransformer model that fuses information from text and other image encoders,
    and obtain state-of-the-art performance on various multimodal classification benchmark tasks.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Supervised Multimodal Bitransformers for Classifying Images and Text`:
        https://github.com/facebookresearch/mmbt

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.MMBTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
        transformer (:class: `~nn.Module`): A text transformer that is used by MMBT.
            It should have embeddings, encoder, and pooler attributes.
        encoder (:class: `~nn.Module`): Encoder for the second modality.
            It should take in a batch of modal inputs and return k, n dimension embeddings.
"""


def add_start_docstrings(*docstr):

    def docstring_decorator(fn):
        fn.__doc__ = ''.join(docstr) + (fn.__doc__ if fn.__doc__ is not None else '')
        return fn
    return docstring_decorator


@add_start_docstrings("""MMBT Model with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output)""", MMBT_START_DOCSTRING, MMBT_INPUTS_DOCSTRING)
class MMBTForClassification(nn.Module):
    """
            **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
                Labels for computing the sequence classification/regression loss.
                Indices should be in ``[0, ..., config.num_labels - 1]``.
                If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
                If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

        Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
            **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Classification (or regression if config.num_labels==1) loss.
            **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
                list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
                of shape ``(batch_size, sequence_length, hidden_size)``:
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            **attentions**: (`optional`, returned when ``config.output_attentions=True``)
                list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

        Examples::

            # For example purposes. Not runnable.
            transformer = BertModel.from_pretrained('bert-base-uncased')
            encoder = ImageEncoder(args)
            model = MMBTForClassification(config, transformer, encoder)
            outputs = model(input_modal, input_ids, labels=labels)
            loss, logits = outputs[:2]
        """

    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.num_labels = config.num_labels
        self.mmbt = MMBTModel(config, transformer, encoder)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_modal, input_ids=None, modal_start_tokens=None, modal_end_tokens=None, attention_mask=None, token_type_ids=None, modal_token_type_ids=None, position_ids=None, modal_position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.mmbt(input_modal=input_modal, input_ids=input_ids, modal_start_tokens=modal_start_tokens, modal_end_tokens=modal_end_tokens, attention_mask=attention_mask, token_type_ids=token_type_ids, modal_token_type_ids=modal_token_type_ids, position_ids=position_ids, modal_position_ids=modal_position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


def _get_least_common_mult_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == 'lsh':
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == 'local':
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(['lsh', 'local']):
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError("Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {}. Select attn layer types from ['lsh', 'local'] only.".format(config.attn_layers))


class AxialPositionEmbeddings(nn.Module):
    """Constructs axial position embeddings. Useful for very long input
    sequences to save memory and time.
    """

    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.dropout = config.hidden_dropout_prob
        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)
        self.weights = nn.ParameterList()
        assert sum(self.axial_pos_embds_dim) == config.hidden_size, 'Make sure that config.axial_pos_embds factors: {} sum to config.hidden_size: {}'.format(self.axial_pos_embds_dim, config.hidden_size)
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)
            self.weights.append(nn.Parameter(torch.ones(ax_shape, dtype=torch.float32)))

    def forward(self, position_ids):
        batch_size = position_ids.shape[0]
        sequence_length = position_ids.shape[1]
        broadcasted_weights = [weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights]
        if self.training is True:
            assert reduce(mul, self.axial_pos_shape) == sequence_length, 'If training, make sure that config.axial_pos_shape factors: {} multiply to sequence length. Got prod({}) != sequence_length: {}. You might want to consider padding your sequence length to {} or changing config.axial_pos_shape.'.format(self.axial_pos_shape, self.axial_pos_shape, sequence_length, reduce(mul, self.axial_pos_shape))
            if self.dropout > 0:
                weights = torch.cat(broadcasted_weights, dim=-1)
                transposed_weights = weights.transpose(2, 1)
                dropped_transposed_weights = nn.functional.dropout2d(transposed_weights, p=self.dropout, training=self.training)
                dropped_weights = dropped_transposed_weights.transpose(2, 1)
                position_encodings = torch.reshape(dropped_weights, (batch_size, sequence_length, -1))
            else:
                position_encodings = torch.cat([torch.reshape(weight, (batch_size, sequence_length, -1)) for weight in broadcasted_weights], dim=-1)
        else:
            assert reduce(mul, self.axial_pos_shape) >= sequence_length, 'Make sure that config.axial_pos_shape factors: {} multiply at least to max(sequence_length, least_common_mult_chunk_length): max({}, {})'.format(self.axial_pos_shape, sequence_length, self.least_common_mult_chunk_length)
            position_encodings = torch.cat(broadcasted_weights, dim=-1)
            position_encodings = position_encodings.view(batch_size, -1, position_encodings.shape[-1])[:, :sequence_length]
        return position_encodings


class PositionEmbeddings(nn.Module):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`.
    """

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        position_embeddings = nn.functional.dropout(position_embeddings, p=self.dropout, training=self.training)
        return position_embeddings


class ReformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.hidden_dropout_prob
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        assert position_ids.shape[-1] <= self.max_position_embeddings, 'Sequence Length: {} has to be larger equal than config.max_position_embeddings: {}'.format(position_ids.shape[-1], self.max_position_embeddings)
        embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        return embeddings


class EfficientAttentionMixin:
    """
    A few utilities for nn.Modules in Reformer, to be used as a mixin.
    """

    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        """ Used to implement attention between consecutive chunks.

            Args:
                vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
                num_chunks_before: chunks before current chunk to include in attention
                num_chunks_after: chunks after current chunk to include in attention

            Returns:
                tensor of shape [num_chunks, N * chunk_length, ...], where
                N = (1 + num_chunks_before + num_chunks_after).
        """
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors
        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        return torch.cat(slices, dim=3)

    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        """
            splits hidden_size dim into attn_head_size and num_attn_heads
        """
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        """
            merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))

    def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        """
            splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        """
        batch_size = vectors.shape[0]
        split_dim_shape = batch_size, num_attn_heads, dim_factor_1, dim_factor_2
        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError('Input vector rank should be one of [3, 4], but is: {}'.format(len(vectors.shape)))


LSHSelfAttentionOutput = namedtuple('LSHSelfAttentionOutput', ['hidden_states', 'attention_probs', 'buckets'])


class ReverseSort(Function):
    """
        After chunked attention is applied which sorted clusters,
        original ordering has to be restored.
        Since customized backward function is used for Reformer,
        the gradients of the output vectors have to be explicitely
        sorted here.
    """

    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx, num_hashes):
        with torch.no_grad():
            ctx.sorted_bucket_idx = sorted_bucket_idx
            ctx.num_hashes = num_hashes
            expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
            out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
            logits = torch.gather(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits):
        sorted_bucket_idx = ctx.sorted_bucket_idx
        num_hashes = ctx.num_hashes
        grad_logits_shape = grad_logits.shape
        grad_out_vectors_shape = grad_out_vectors.shape
        grad_logits = grad_logits.view(grad_logits_shape[:2] + (num_hashes, -1))
        grad_out_vectors = grad_out_vectors.view(grad_out_vectors_shape[:2] + (num_hashes, -1) + grad_out_vectors_shape[-1:])
        sorted_bucket_idx = torch.reshape(sorted_bucket_idx, sorted_bucket_idx.shape[:2] + (num_hashes, -1))
        expanded_sort_indices = sorted_bucket_idx.unsqueeze(-1).expand(grad_out_vectors.shape)
        grad_out_vectors = torch.gather(grad_out_vectors, 3, expanded_sort_indices)
        grad_logits = torch.gather(grad_logits, 3, sorted_bucket_idx)
        grad_logits = torch.reshape(grad_logits, grad_logits_shape)
        grad_out_vectors = torch.reshape(grad_out_vectors, grad_out_vectors_shape)
        return grad_out_vectors, grad_logits, None, None, None


class LSHSelfAttention(nn.Module, EfficientAttentionMixin):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = config.lsh_num_chunks_before
        self.num_chunks_after = config.lsh_num_chunks_after
        self.hash_seed = config.hash_seed
        self.is_decoder = config.is_decoder
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.lsh_attention_probs_dropout_prob
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size
        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.register_buffer('self_mask_value_float16', torch.tensor(-1000.0))
        self.register_buffer('self_mask_value_float32', torch.tensor(-100000.0))
        self.register_buffer('mask_value_float16', torch.tensor(-10000.0))
        self.register_buffer('mask_value_float32', torch.tensor(-1000000000.0))

    def forward(self, hidden_states, attention_mask=None, head_mask=None, num_hashes=None, do_output_attentions=False, buckets=None, **kwargs):
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes
        query_key_vectors = self.query_key(hidden_states)
        value_vectors = self.value(hidden_states)
        del hidden_states
        query_key_vectors = self._split_hidden_size_dim(query_key_vectors, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._split_hidden_size_dim(value_vectors, self.num_attention_heads, self.attention_head_size)
        assert query_key_vectors.shape[-1] == self.attention_head_size, 'last dim of query_key_vectors is {} but should be {}.'.format(query_key_vectors.shape[-1], self.attention_head_size)
        assert value_vectors.shape[-1] == self.attention_head_size, 'last dim of value_vectors is {} but should be {}.'.format(value_vectors.shape[-1], self.attention_head_size)
        if self.num_buckets is None:
            self._set_num_buckets(sequence_length)
        if buckets is None:
            buckets = self._hash_vectors(query_key_vectors, num_hashes)
        assert int(buckets.shape[-1]) == num_hashes * sequence_length, 'last dim of buckets is {}, but should be {}'.format(buckets.shape[-1], num_hashes * sequence_length)
        sorted_bucket_idx, undo_sorted_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(sequence_length, buckets, num_hashes)
        sorted_bucket_idx = sorted_bucket_idx % sequence_length
        query_key_vectors = self._gather_by_expansion(query_key_vectors, sorted_bucket_idx, num_hashes)
        value_vectors = self._gather_by_expansion(value_vectors, sorted_bucket_idx, num_hashes)
        query_key_vectors = self._split_seq_length_dim_to(query_key_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._split_seq_length_dim_to(value_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size)
        if self.chunk_length is None:
            assert self.num_chunks_before == 0 and self.num_chunks_after == 0, 'If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and `config.num_chunks_before` are set to 0.'
        key_vectors = self._len_and_dim_norm(query_key_vectors)
        out_vectors, logits, attention_probs = self._attend(query_vectors=query_key_vectors, key_vectors=key_vectors, value_vectors=value_vectors, sorted_bucket_idx=sorted_bucket_idx, attention_mask=attention_mask, head_mask=head_mask)
        del query_key_vectors, key_vectors, value_vectors
        out_vectors, logits = ReverseSort.apply(out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx, self.num_hashes)
        if num_hashes > 1:
            out_vectors = self._split_seq_length_dim_to(out_vectors, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size)
            logits = self._split_seq_length_dim_to(logits, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size).unsqueeze(-1)
            probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
            out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
            del probs_vectors
        del logits
        assert out_vectors.shape == (batch_size, self.num_attention_heads, sequence_length, self.attention_head_size), 'out_vectors have be of shape `[batch_size, config.num_attention_heads, sequence_length, config.attention_head_size]`.'
        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)
        if do_output_attentions is False:
            attention_probs = ()
        return LSHSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs, buckets=buckets)

    def _hash_vectors(self, vectors, num_hashes):
        batch_size = vectors.shape[0]
        if isinstance(self.num_buckets, int):
            assert self.num_buckets % 2 == 0, 'There should be an even number of bucktes, but `self.num_bucktes`: {}'.format(self.num_buckets)
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert bucket_factor % 2 == 0, 'The number of buckets should be even, but `num_bucket`: {}'.format(bucket_factor)
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor
        vectors = vectors.detach()
        if self.hash_seed is not None:
            torch.manual_seed(self.hash_seed)
        rotations_shape = self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        rotated_vectors = torch.einsum('bmtd,mdhr->bmhtr', vectors, random_rotations)
        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[..., cur_sum:cur_sum + bucket_factor // 2]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = torch.cat([rotated_vectors_factor, -rotated_vectors_factor], dim=-1)
                if buckets is None:
                    buckets = torch.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + cur_product * torch.argmax(rotated_vectors_factor, dim=-1)
                cur_product = cur_product * bucket_factor
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        with torch.no_grad():
            batch_size = buckets.shape[0]
            orig_indices = torch.arange(num_hashes * sequence_length, device=buckets.device).view(1, 1, -1)
            orig_indices = orig_indices.expand(batch_size, self.num_attention_heads, orig_indices.shape[-1])
            scaled_buckets = sequence_length * buckets + orig_indices % sequence_length
            scaled_buckets = scaled_buckets.detach()
            sorted_bucket_idx = torch.argsort(scaled_buckets, dim=-1)
            indices = torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device).view(1, 1, -1).expand(sorted_bucket_idx.shape)
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _set_num_buckets(self, sequence_length):
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        num_buckets = 2 ** num_buckets_pow_2
        num_buckets_limit = 2 * max(int((self.max_position_embeddings // self.chunk_length) ** 0.5), self.chunk_length)
        if num_buckets > num_buckets_limit:
            num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]
        logger.warning('config.num_buckets is not set. Setting config.num_buckets to {}...'.format(num_buckets))
        self.config.num_buckets = num_buckets
        self.num_buckets = num_buckets

    def _attend(self, query_vectors, key_vectors, value_vectors, sorted_bucket_idx, attention_mask, head_mask):
        key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
        value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        del query_vectors, key_vectors
        query_bucket_idx = self._split_seq_length_dim_to(sorted_bucket_idx, -1, self.chunk_length, self.num_attention_heads)
        key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        if query_key_dots.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32
        mask = self._compute_attn_mask(query_bucket_idx, key_value_bucket_idx, attention_mask)
        if mask is not None:
            query_key_dots = torch.where(mask, query_key_dots, mask_value)
        del mask
        self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2))
        query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)
        del self_mask
        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        attention_probs = torch.exp(query_key_dots - logits)
        del query_key_dots
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        out_vectors = torch.matmul(attention_probs, value_vectors)
        del value_vectors
        logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
        out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        return out_vectors, logits, attention_probs

    def _compute_attn_mask(self, query_indices, key_indices, attention_mask):
        mask = None
        if self.is_decoder:
            mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2))
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
            key_attn_mask = torch.gather(attention_mask, -1, key_indices)
            query_attn_mask = torch.gather(attention_mask, -1, query_indices)
            attn_mask = query_attn_mask.unsqueeze(-1) * key_attn_mask.unsqueeze(-2)
            del query_attn_mask, key_attn_mask, attention_mask
            if mask is not None:
                mask = mask * attn_mask
            else:
                mask = attn_mask
        return mask

    def _len_and_dim_norm(self, vectors):
        """
            length and attention head size dim normalization
        """
        vectors = self._len_norm(vectors)
        vectors = vectors * torch.rsqrt(torch.tensor(self.attention_head_size, device=vectors.device, dtype=vectors.dtype))
        return vectors

    def _len_norm(self, x, epsilon=1e-06):
        """
            length normalization
        """
        variance = torch.mean(x ** 2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
            expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)


LocalSelfAttentionOutput = namedtuple('LocalSelfAttentionOutput', ['hidden_states', 'attention_probs'])


class LocalSelfAttention(nn.Module, EfficientAttentionMixin):

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.chunk_length = config.local_attn_chunk_length
        self.num_chunks_before = config.local_num_chunks_before
        self.num_chunks_after = config.local_num_chunks_after
        self.is_decoder = config.is_decoder
        self.pad_token_id = config.pad_token_id
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.dropout = config.local_attention_probs_dropout_prob
        self.register_buffer('mask_value_float16', torch.tensor(-10000.0))
        self.register_buffer('mask_value_float32', torch.tensor(-1000000000.0))

    def forward(self, hidden_states, attention_mask=None, head_mask=None, do_output_attentions=False, **kwargs):
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)
        query_vectors = self._split_hidden_size_dim(query_vectors, self.num_attention_heads, self.attention_head_size)
        key_vectors = self._split_hidden_size_dim(key_vectors, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._split_hidden_size_dim(value_vectors, self.num_attention_heads, self.attention_head_size)
        assert query_vectors.shape[-1] == self.attention_head_size, 'last dim of query_key_vectors is {} but should be {}.'.format(query_vectors.shape[-1], self.attention_head_size)
        assert key_vectors.shape[-1] == self.attention_head_size, 'last dim of query_key_vectors is {} but should be {}.'.format(key_vectors.shape[-1], self.attention_head_size)
        assert value_vectors.shape[-1] == self.attention_head_size, 'last dim of query_key_vectors is {} but should be {}.'.format(value_vectors.shape[-1], self.attention_head_size)
        if self.chunk_length is None:
            assert self.num_chunks_before == 0 and self.num_chunks_after == 0, 'If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and `config.num_chunks_before` are set to 0.'
        key_vectors = key_vectors / torch.sqrt(torch.tensor(self.attention_head_size, device=key_vectors.device, dtype=key_vectors.dtype))
        query_vectors = self._split_seq_length_dim_to(query_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size)
        key_vectors = self._split_seq_length_dim_to(key_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._split_seq_length_dim_to(value_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size)
        indices = torch.arange(sequence_length, device=query_vectors.device).repeat(batch_size, self.num_attention_heads, 1)
        query_indices = self._split_seq_length_dim_to(indices, -1, self.chunk_length, self.num_attention_heads)
        key_indices = self._split_seq_length_dim_to(indices, -1, self.chunk_length, self.num_attention_heads)
        key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
        value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)
        key_indices = self._look_adjacent(key_indices, self.num_chunks_before, self.num_chunks_after)
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        del query_vectors, key_vectors
        mask = self._compute_attn_mask(query_indices, key_indices, attention_mask, query_key_dots.shape)
        if mask is not None:
            if query_key_dots.dtype == torch.float16:
                mask_value = self.mask_value_float16.half()
            else:
                mask_value = self.mask_value_float32
            query_key_dots = torch.where(mask, query_key_dots, mask_value)
        del mask
        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        attention_probs = torch.exp(query_key_dots - logits)
        del logits
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        out_vectors = torch.matmul(attention_probs, value_vectors)
        del value_vectors
        out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        assert out_vectors.shape == (batch_size, self.num_attention_heads, sequence_length, self.attention_head_size)
        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)
        if do_output_attentions is False:
            attention_probs = ()
        return LocalSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs)

    def _compute_attn_mask(self, query_indices, key_indices, attention_mask, query_key_dots_shape):
        mask = None
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, :]
            attention_mask = self._split_seq_length_dim_to(attention_mask, -1, self.chunk_length, 1)
            attention_mask_key = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)
        if self.is_decoder is True:
            mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2))
        if attention_mask is not None:
            attn_mask = (attention_mask.unsqueeze(-1) * attention_mask_key.unsqueeze(-2)).expand(query_key_dots_shape)
            if mask is not None:
                mask = mask * attn_mask
            else:
                mask = attn_mask
        return mask


class ReformerSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        all_head_size = config.num_attention_heads * config.attention_head_size
        self.dropout = config.hidden_dropout_prob
        self.dense = nn.Linear(all_head_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


AttentionOutput = namedtuple('AttentionOutput', ['hidden_states', 'attention_probs', 'buckets'])


class ReformerAttention(nn.Module):

    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn_layers = config.attn_layers
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == 'lsh':
            self.self_attention = LSHSelfAttention(config)
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == 'local':
            self.self_attention = LocalSelfAttention(config)
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == set(['lsh', 'local']):
            if self.attn_layers[self.layer_id] == 'lsh':
                self.self_attention = LSHSelfAttention(config)
            else:
                self.self_attention = LocalSelfAttention(config)
        else:
            raise NotImplementedError("Only attn layer types 'lsh' and 'local' exist, but got `config.attn_layers`: {}. Select attn layer types from ['lsh', 'local'] only.".format(self.attn_layers))
        self.output = ReformerSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, num_hashes=None, do_output_attentions=False, buckets=None):
        hidden_states = self.layer_norm(hidden_states)
        self_attention_outputs = self.self_attention(hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask, num_hashes=num_hashes, do_output_attentions=do_output_attentions, buckets=buckets)
        attention_output = self.output(self_attention_outputs.hidden_states)
        if hasattr(self_attention_outputs, 'buckets'):
            buckets = self_attention_outputs.buckets
        else:
            buckets = None
        return AttentionOutput(hidden_states=attention_output, attention_probs=self_attention_outputs.attention_probs, buckets=buckets)


class ReformerFeedForwardDense(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act
        self.dense = nn.Linear(config.hidden_size, config.feed_forward_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class ReformerFeedForwardOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.dense = nn.Linear(config.feed_forward_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


def apply_chunking_to_forward(chunk_size: int, chunk_dim: int, forward_fn: Callable[..., torch.Tensor], *input_tensors) ->torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension `chunk_dim`.
    It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the
    same result as not applying it.

    Args:
        chunk_size: int - the chunk size of a chunked tensor. `num_chunks` = `len(input_tensors[0]) / chunk_size`
        chunk_dim: int - the dimension over which the input_tensors should be chunked
        forward_fn: fn - the forward fn of the model
        input_tensors: tuple(torch.Tensor) - the input tensors of `forward_fn` which are chunked
    Returns:
        a Tensor with the same shape the foward_fn would have given if applied


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.chunk_size_lm_head, self.seq_len_dim, self.forward_chunk, hidden_states)
    """
    assert len(input_tensors) > 0, '{} has to be a tuple/list of tensors'.format(input_tensors)
    tensor_shape = input_tensors[0].shape
    assert all(input_tensor.shape == tensor_shape for input_tensor in input_tensors), 'All input tenors have to be of the same shape'
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(input_tensors), 'forward_chunk_fn expects {} arguments, but only {} input tensors are given'.format(num_args_in_forward_chunk_fn, len(input_tensors))
    if chunk_size > 0:
        assert input_tensors[0].shape[chunk_dim] % chunk_size == 0, 'The dimension to be chunked {} has to be a multiple of the chunk size {}'.format(input_tensors[0][chunk_dim], chunk_size)
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        return torch.cat(output_chunks, dim=chunk_dim)
    return forward_fn(*input_tensors)


class ChunkReformerFeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = ReformerFeedForwardDense(config)
        self.output = ReformerFeedForwardOutput(config)

    def forward(self, attention_output):
        return apply_chunking_to_forward(self.chunk_size_feed_forward, self.seq_len_dim, self.forward_chunk, attention_output)

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        return self.output(hidden_states)


ReformerBackwardOutput = namedtuple('ReformerBackwardOutput', ['attn_output', 'hidden_states', 'grad_attn_output', 'grad_hidden_states'])


ReformerOutput = namedtuple('ReformerOutput', ['hidden_states', 'attn_output', 'attention_probs', 'buckets'])


class ReformerLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = ReformerAttention(config, layer_id)
        self.attention_seed = None
        self.feed_forward_seed = None
        self.feed_forward = ChunkReformerFeedForward(config)

    def _init_attention_seed(self):
        """
            This function sets a new seed for the
            attention layer to make dropout deterministic
            for both forward calls: 1 normal forward
            call and 1 forward call in backward
            to recalculate activations.
        """
        if next(self.parameters()).device.type == 'cuda':
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
            torch.manual_seed(self.attention_seed)
        else:
            self.attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
            This function sets a new seed for the
            feed forward layer to make dropout deterministic
            for both forward calls: 1 normal forward
            call and 1 forward call in backward
            to recalculate activations.
        """
        if next(self.parameters()).device.type == 'cuda':
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
            torch.manual_seed(self.feed_forward_seed)
        else:
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(self.feed_forward_seed)

    def forward(self, prev_attn_output, hidden_states, attention_mask=None, head_mask=None, num_hashes=None, do_output_attentions=False):
        with torch.no_grad():
            self._init_attention_seed()
            attn_outputs = self.attention(hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask, num_hashes=num_hashes, do_output_attentions=do_output_attentions)
            attn_output = attn_outputs.hidden_states
            attn_output = prev_attn_output + attn_output
            del prev_attn_output
            self._init_feed_forward_seed()
            hidden_states = hidden_states + self.feed_forward(attn_output)
        return ReformerOutput(attn_output=attn_output, hidden_states=hidden_states, attention_probs=attn_outputs.attention_probs, buckets=attn_outputs.buckets)

    def backward_pass(self, next_attn_output, hidden_states, grad_attn_output, grad_hidden_states, attention_mask=None, head_mask=None, buckets=None):
        with torch.enable_grad():
            next_attn_output.requires_grad = True
            torch.manual_seed(self.feed_forward_seed)
            res_hidden_states = self.feed_forward(next_attn_output)
            res_hidden_states.backward(grad_hidden_states, retain_graph=True)
        with torch.no_grad():
            hidden_states = hidden_states - res_hidden_states
            del res_hidden_states
            grad_attn_output = grad_attn_output + next_attn_output.grad
            next_attn_output.grad = None
        with torch.enable_grad():
            hidden_states.requires_grad = True
            torch.manual_seed(self.attention_seed)
            output = self.attention(hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask, buckets=buckets).hidden_states
            output.backward(grad_attn_output, retain_graph=True)
        with torch.no_grad():
            attn_output = next_attn_output - output
            del output, next_attn_output
            grad_hidden_states = grad_hidden_states + hidden_states.grad
            hidden_states.grad = None
            hidden_states = hidden_states.detach()
        return ReformerBackwardOutput(attn_output=attn_output, hidden_states=hidden_states, grad_attn_output=grad_attn_output, grad_hidden_states=grad_hidden_states)


ReformerEncoderOutput = namedtuple('ReformerEncoderOutput', ['hidden_states', 'all_hidden_states', 'all_attentions'])


class _ReversibleFunction(Function):
    """
    To prevent PyTorch from performing the usual backpropagation,
    a customized backward function is implemented here. This way
    it is made sure that no memory expensive activations are
    saved during the forward pass.
    This function is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
    """

    @staticmethod
    def forward(ctx, hidden_states, layers, attention_mask, head_mask, num_hashes, all_hidden_states, all_attentions, do_output_hidden_states, do_output_attentions):
        all_buckets = ()
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)
        for layer, layer_head_mask in zip(layers, head_mask):
            if do_output_hidden_states is True:
                all_hidden_states.append(hidden_states)
            layer_outputs = layer(prev_attn_output=attn_output, hidden_states=hidden_states, attention_mask=attention_mask, head_mask=layer_head_mask, num_hashes=num_hashes, do_output_attentions=do_output_attentions)
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            all_buckets = all_buckets + (layer_outputs.buckets,)
            if do_output_attentions:
                all_attentions.append(layer_outputs.attention_probs)
        if do_output_hidden_states is True:
            all_hidden_states.append(hidden_states)
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)
        attn_output, hidden_states = ctx.saved_tensors
        output = ReformerBackwardOutput(attn_output=attn_output, hidden_states=hidden_states, grad_attn_output=grad_attn_output, grad_hidden_states=grad_hidden_states)
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states
        layers = ctx.layers
        all_buckets = ctx.all_buckets
        head_mask = ctx.head_mask
        attention_mask = ctx.attention_mask
        for idx, layer in enumerate(layers[::-1]):
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]
            output = layer.backward_pass(next_attn_output=output.attn_output, hidden_states=output.hidden_states, grad_attn_output=output.grad_attn_output, grad_hidden_states=output.grad_hidden_states, head_mask=head_mask[len(layers) - idx - 1], attention_mask=attention_mask, buckets=buckets)
        assert all_buckets == (), 'buckets have to be empty after backpropagation'
        grad_hidden_states = torch.cat([output.grad_attn_output, output.grad_hidden_states], dim=-1)
        return grad_hidden_states, None, None, None, None, None, None, None, None


class ReformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.layers = nn.ModuleList([ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, num_hashes=None, do_output_hidden_states=False, do_output_attentions=False):
        all_hidden_states = []
        all_attentions = []
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(hidden_states, self.layers, attention_mask, head_mask, num_hashes, all_hidden_states, all_attentions, do_output_hidden_states, do_output_attentions)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return ReformerEncoderOutput(hidden_states=hidden_states, all_hidden_states=all_hidden_states, all_attentions=all_attentions)


class ReformerOnlyLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.chunk_size_lm_head = config.chunk_size_lm_head
        self.decoder = nn.Linear(2 * config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.chunk_size_lm_head, self.seq_len_dim, self.forward_chunk, hidden_states)

    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        return super().forward(input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, inv_lang_adapter=None, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        if inv_lang_adapter:
            x = inv_lang_adapter(x, rev=True)
        x = self.decoder(x)
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class T5LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)
        if self.weight.dtype == torch.float16:
            x = x
        return self.weight * x


class T5DenseReluDense(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        h = self.wi(hidden_states)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class T5LayerFF(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = T5DenseReluDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output


class T5Config(PretrainedConfig):
    """
        :class:`~transformers.T5Config` is the configuration class to store the configuration of a
        `T5Model`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `T5Model`.
            d_model: Size of the encoder layers and the pooler layer. `d_model` can also accesed via the property `hidden_size`.
            num_layers: Number of hidden layers in the Transformer encoder. `num_layers` can also be accessed via the property `num_hidden_layers`.
            num_heads: Number of attention heads for each attention layer in
                the Transformer encoder. `num_heads` can also be accessed via the property `num_attention_heads`.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            n_positions: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048). `n_positions` can also be accessed via the property `max_position_embeddings'.
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `T5Model`.
            initializer_factor: A factor for initializing all weight matrices (should be kept to 1.0, used for initialization testing).
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    model_type = 't5'

    def __init__(self, vocab_size=32128, n_positions=512, d_model=512, d_kv=64, d_ff=2048, num_layers=6, num_heads=8, relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon=1e-06, initializer_factor=1.0, is_encoder_decoder=True, pad_token_id=0, eos_token_id=1, **kwargs):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, **kwargs)
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers


class T5Attention(nn.Module):

    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.output_attentions = config.output_attentions
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, self.d_kv)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position, bidirectional=not self.is_decoder, num_buckets=self.relative_attention_num_buckets)
        rp_bucket = rp_bucket
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, input, mask=None, kv=None, position_bias=None, past_key_value_state=None, head_mask=None, query_length=None, use_cache=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        bs, qlen, dim = input.size()
        if past_key_value_state is not None:
            assert self.is_decoder is True, 'Encoder cannot cache past key value states'
            assert len(past_key_value_state) == 2, 'past_key_value_state should have 2 past states: keys and values. Got {} past states'.format(len(past_key_value_state))
            real_qlen = qlen + past_key_value_state[0].shape[2] if query_length is None else query_length
        else:
            real_qlen = qlen
        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)
        q = shape(self.q(input))
        if kv is None:
            k = shape(self.k(input))
            v = shape(self.v(input))
        elif past_key_value_state is None:
            k = v = kv
            k = shape(self.k(k))
            v = shape(self.v(v))
        if past_key_value_state is not None:
            if kv is None:
                k_, v_ = past_key_value_state
                k = torch.cat([k_, k], dim=2)
                v = torch.cat([v_, v], dim=2)
            else:
                k, v = past_key_value_state
        if self.is_decoder and use_cache is True:
            present_key_value_state = (k, v),
        else:
            present_key_value_state = None,
        scores = torch.einsum('bnqd,bnkd->bnqk', q, k)
        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError('No position_bias provided and no weights to compute position_bias')
            position_bias = self.compute_bias(real_qlen, klen)
            if past_key_value_state is not None:
                position_bias = position_bias[:, :, -1:, :]
            if mask is not None:
                position_bias = position_bias + mask
        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        context = self.o(context)
        outputs = (context,) + present_key_value_state
        if self.output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


class T5LayerSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, head_mask=None, past_key_value_state=None, use_cache=False):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(norm_x, mask=attention_mask, position_bias=position_bias, head_mask=head_mask, past_key_value_state=past_key_value_state, use_cache=use_cache)
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, kv, attention_mask=None, position_bias=None, head_mask=None, past_key_value_state=None, use_cache=False, query_length=None):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(norm_x, mask=attention_mask, kv=kv, position_bias=position_bias, head_mask=head_mask, past_key_value_state=past_key_value_state, use_cache=use_cache, query_length=query_length)
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class T5Block(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerFF(config))

    def forward(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, head_mask=None, past_key_value_state=None, use_cache=False):
        if past_key_value_state is not None:
            assert self.is_decoder, 'Only decoder can use `past_key_value_states`'
            expected_num_past_key_value_states = 2 if encoder_hidden_states is None else 4
            error_message = 'There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states'.format(expected_num_past_key_value_states, '2 (past / key) for cross attention' if expected_num_past_key_value_states == 4 else '', len(past_key_value_state))
            assert len(past_key_value_state) == expected_num_past_key_value_states, error_message
            self_attn_past_key_value_state = past_key_value_state[:2]
            cross_attn_past_key_value_state = past_key_value_state[2:]
        else:
            self_attn_past_key_value_state, cross_attn_past_key_value_state = None, None
        self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias, head_mask=head_mask, past_key_value_state=self_attn_past_key_value_state, use_cache=use_cache)
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]
        if self.is_decoder and encoder_hidden_states is not None:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
            cross_attention_outputs = self.layer[1](hidden_states, kv=encoder_hidden_states, attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias, head_mask=head_mask, past_key_value_state=cross_attn_past_key_value_state, query_length=query_length, use_cache=use_cache)
            hidden_states = cross_attention_outputs[0]
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        hidden_states = self.layer[-1](hidden_states)
        outputs = hidden_states,
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs


class PositionalEmbedding(nn.Module):

    def __init__(self, demb):
        super().__init__()
        self.demb = demb
        inv_freq = 1 / 10000 ** (torch.arange(0.0, demb, 2.0) / demb)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-05):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.CoreNet = nn.Sequential(nn.Linear(d_model, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_inner, d_model), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)
        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, r_r_bias=None, r_w_bias=None, output_attentions=False, layer_norm_epsilon=1e-05):
        super().__init__()
        self.output_attentions = output_attentions
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm
        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)
        x = x_padded[1:].view_as(x)
        return x

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
        rw_head_q = w_head_q + self.r_w_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and torch.sum(attn_mask).item():
            attn_mask = attn_mask == 1
            if attn_mask.dim() == 2:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], -1e+30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -1e+30).type_as(attn_score)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            outputs = [w + attn_out]
        else:
            outputs = [self.layer_norm(w + attn_out)]
        if self.output_attentions:
            outputs.append(attn_prob)
        return outputs


class RelPartialLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-05, **kwargs):
        super().__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'), layer_norm_epsilon=layer_norm_epsilon)

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None):
        attn_outputs = self.dec_attn(dec_inp, r, attn_mask=dec_attn_mask, mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output] + attn_outputs[1:]
        return outputs


class AdaptiveEmbedding(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super().__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])
                emb_flat.index_copy_(0, indices_i, emb_i)
            embed_shape = inp.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)
        embed.mul_(self.emb_scale)
        return embed


class ProjectedAdaptiveLogSoftmax(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super().__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)
            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))
                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))
        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
        return logit

    def forward(self, hidden, labels=None, keep_order=False):
        """
            Params:
                hidden :: [len*bsz x d_proj]
                labels :: [len*bsz]
            Return:
                if labels is None:
                    out :: [len*bsz x n_tokens] log probabilities of tokens over the vocabulary
                else:
                    out :: [(len-1)*bsz] Negative log likelihood
            We could replace this implementation by the native PyTorch one
            if their's had an option to set bias on all clusters in the native one.
            here: https://github.com/pytorch/pytorch/blob/dbe6a7a9ff1a364a8706bf5df58a1ca96d2fd9da/torch/nn/modules/adaptive.py#L138
        """
        if labels is not None:
            hidden = hidden[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            hidden = hidden.view(-1, hidden.size(-1))
            labels = labels.view(-1)
            if hidden.size(0) != labels.size(0):
                raise RuntimeError('Input and labels should have the same size in the batch dimension.')
        else:
            hidden = hidden.view(-1, hidden.size(-1))
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
            if labels is not None:
                out = -F.log_softmax(logit, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            else:
                out = F.log_softmax(logit, dim=-1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)
            if labels is None:
                out = hidden.new_empty((head_logit.size(0), self.n_token))
            else:
                out = torch.zeros_like(labels, dtype=hidden.dtype, device=hidden.device)
            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                if labels is not None:
                    mask_i = (labels >= l_idx) & (labels < r_idx)
                    indices_i = mask_i.nonzero().squeeze()
                    if indices_i.numel() == 0:
                        continue
                    target_i = labels.index_select(0, indices_i) - l_idx
                    head_logprob_i = head_logprob.index_select(0, indices_i)
                    hidden_i = hidden.index_select(0, indices_i)
                else:
                    hidden_i = hidden
                if i == 0:
                    if labels is not None:
                        logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    cluster_prob_idx = self.cutoffs[0] + i - 1
                    if labels is not None:
                        logprob_i = head_logprob_i[:, cluster_prob_idx] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        logprob_i = head_logprob[:, cluster_prob_idx, None] + tail_logprob_i
                        out[:, l_idx:r_idx] = logprob_i
                if labels is not None:
                    if hasattr(self, 'keep_order') and self.keep_order or keep_order:
                        out.index_copy_(0, indices_i, -logprob_i)
                    else:
                        out[offset:offset + logprob_i.size(0)].copy_(-logprob_i)
                    offset += logprob_i.size(0)
        return out

    def log_prob(self, hidden):
        """ Computes log probabilities for all :math:`n\\_classes`
        From: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.py
        Args:
            hidden (Tensor): a minibatch of examples
        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= n\\_classes`, where :math:`n\\_classes` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.
        Shape:
            - Input: :math:`(N, in\\_features)`
            - Output: :math:`(N, n\\_classes)`
        """
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
            return F.log_softmax(logit, dim=-1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            out = hidden.new_empty((head_logit.size(0), self.n_token))
            head_logprob = F.log_softmax(head_logit, dim=1)
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                start_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]
                if i == 0:
                    out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    logprob_i = head_logprob[:, -i] + tail_logprob_i
                    out[:, start_idx, stop_idx] = logprob_i
            return out


class BeamHypotheses(object):

    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1000000000.0

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]


TF2_WEIGHTS_NAME = 'tf_model.h5'


TF_WEIGHTS_NAME = 'model.ckpt'


WEIGHTS_NAME = 'pytorch_model.bin'


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) ->Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            return True
        if len(tokens) > len(prev_input_ids):
            return False
        if prev_tokens[-len(tokens):] == tokens:
            return True
        else:
            return False
    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []
        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, 'Banned words token sequences {} cannot have an empty list'.format(bad_words_ids)
            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                continue
            banned_tokens_slice.append(banned_token_seq[-1])
        banned_tokens.append(banned_tokens_slice)
    return banned_tokens


def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) ->None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])
    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def top_k_top_p_filtering(logits: Tensor, top_k: int=0, top_p: float=1.0, filter_value: float=-float('Inf'), min_tokens_to_keep: int=1) ->Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class PoolerStartLogits(nn.Module):
    """ Compute SQuAD start_logits from sequence hidden states. """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.dense(hidden_states).squeeze(-1)
        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerEndLogits(nn.Module):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
        """ Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to hidden_states
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
                Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions)
            start_states = start_states.expand(-1, slen, -1)
        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)
        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerAnswerClass(nn.Module):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, start_states=None, start_positions=None, cls_index=None):
        """
        Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to ``hidden_states``.
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span.
            **cls_index**: torch.LongTensor of shape ``(batch_size,)``
                position of the CLS token. If None, take the last token.

            note(Original repo):
                no dependency on end_feature so that we can obtain one single `cls_logits`
                for each sample
        """
        hsz = hidden_states.shape[-1]
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)
        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)
        else:
            cls_token_state = hidden_states[:, -1, :]
        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)
        return x


class SQuADHead(nn.Module):
    """ A SQuAD head inspired by XLNet.

    Parameters:
        config (:class:`~transformers.XLNetConfig`): Model configuration class with all the parameters of the model.

    Inputs:
        **hidden_states**: ``torch.FloatTensor`` of shape ``(batch_size, seq_len, hidden_size)``
            hidden states of sequence tokens
        **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the first token for the labeled span.
        **end_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the last token for the labeled span.
        **cls_index**: torch.LongTensor of shape ``(batch_size,)``
            position of the CLS token. If None, take the last token.
        **is_impossible**: ``torch.LongTensor`` of shape ``(batch_size,)``
            Whether the question has a possible answer in the paragraph or not.
        **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
            Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
            1.0 means token should be masked.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned if both ``start_positions`` and ``end_positions`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
            Indices for the top config.start_n_top start token possibilities (beam-search).
        **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size,)``
            Log probabilities for the ``is_impossible`` label of the answers.
    """

    def __init__(self, config):
        super().__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def forward(self, hidden_states, start_positions=None, end_positions=None, cls_index=None, is_impossible=None, p_mask=None):
        outputs = ()
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        if start_positions is not None and end_positions is not None:
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if cls_index is not None and is_impossible is not None:
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)
                total_loss += cls_loss * 0.5
            outputs = (total_loss,) + outputs
        else:
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)
            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)
            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)
            start_states = torch.einsum('blh,bl->bh', hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)
            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs
        return outputs


class SequenceSummary(nn.Module):
    """ Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' or another string => add an activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.summary_type = getattr(config, 'summary_type', 'last')
        if self.summary_type == 'attn':
            raise NotImplementedError
        self.summary = Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)
        activation_string = getattr(config, 'summary_activation', None)
        self.activation: Callable = get_activation(activation_string) if activation_string else Identity()
        self.first_dropout = Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)
        self.last_dropout = Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states, cls_index=None):
        """ hidden_states: float Tensor in shape [bsz, ..., seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2] - 1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            output = hidden_states.gather(-2, cls_index).squeeze(-2)
        elif self.summary_type == 'attn':
            raise NotImplementedError
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super().__init__()
        self.dropout = config.dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = gelu if config.gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class XLMConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.XLMModel`.
        It is used to instantiate an XLM model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the `xlm-mlm-en-2048 <https://huggingface.co/xlm-mlm-en-2048>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

        Args:
            vocab_size (:obj:`int`, optional, defaults to 30145):
                Vocabulary size of the XLM model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.XLMModel`.
            emb_dim (:obj:`int`, optional, defaults to 2048):
                Dimensionality of the encoder layers and the pooler layer.
            n_layer (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            n_head (:obj:`int`, optional, defaults to 16):
                Number of attention heads for each attention layer in the Transformer encoder.
            dropout (:obj:`float`, optional, defaults to 0.1):
                The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_dropout (:obj:`float`, optional, defaults to 0.1):
                The dropout probability for the attention mechanism
            gelu_activation (:obj:`boolean`, optional, defaults to :obj:`True`):
                The non-linear activation function (function or string) in the
                encoder and pooler. If set to `True`, "gelu" will be used instead of "relu".
            sinusoidal_embeddings (:obj:`boolean`, optional, defaults to :obj:`False`):
                Whether to use sinusoidal positional embeddings instead of absolute positional embeddings.
            causal (:obj:`boolean`, optional, defaults to :obj:`False`):
                Set this to `True` for the model to behave in a causal manner.
                Causal models use a triangular attention mask in order to only attend to the left-side context instead
                if a bidirectional context.
            asm (:obj:`boolean`, optional, defaults to :obj:`False`):
                Whether to use an adaptive log softmax projection layer instead of a linear layer for the prediction
                layer.
            n_langs (:obj:`int`, optional, defaults to 1):
                The number of languages the model handles. Set to 1 for monolingual models.
            use_lang_emb (:obj:`boolean`, optional, defaults to :obj:`True`)
                Whether to use language embeddings. Some models use additional language embeddings, see
                `the multilingual models page <http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings>`__
                for information on how to use them.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            embed_init_std (:obj:`float`, optional, defaults to 2048^-0.5):
                The standard deviation of the truncated_normal_initializer for
                initializing the embedding matrices.
            init_std (:obj:`int`, optional, defaults to 50257):
                The standard deviation of the truncated_normal_initializer for
                initializing all weight matrices except the embedding matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            bos_index (:obj:`int`, optional, defaults to 0):
                The index of the beginning of sentence token in the vocabulary.
            eos_index (:obj:`int`, optional, defaults to 1):
                The index of the end of sentence token in the vocabulary.
            pad_index (:obj:`int`, optional, defaults to 2):
                The index of the padding token in the vocabulary.
            unk_index (:obj:`int`, optional, defaults to 3):
                The index of the unknown token in the vocabulary.
            mask_index (:obj:`int`, optional, defaults to 5):
                The index of the masking token in the vocabulary.
            is_encoder(:obj:`boolean`, optional, defaults to :obj:`True`):
                Whether the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
            summary_type (:obj:`string`, optional, defaults to "first"):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLMForSequenceClassification`.
                Is one of the following options:

                - 'last' => take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLMForSequenceClassification`.
                Add a projection after the vector extraction
            summary_activation (:obj:`string` or :obj:`None`, optional, defaults to :obj:`None`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLMForSequenceClassification`.
                'tanh' => add a tanh activation to the output, Other => no activation.
            summary_proj_to_labels (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLMForSequenceClassification`.
                If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_first_dropout (:obj:`float`, optional, defaults to 0.1):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLMForSequenceClassification`.
                Add a dropout before the projection and activation
            start_n_top (:obj:`int`, optional, defaults to 5):
                Used in the SQuAD evaluation script for XLM and XLNet.
            end_n_top (:obj:`int`, optional, defaults to 5):
                Used in the SQuAD evaluation script for XLM and XLNet.
            mask_token_id (:obj:`int`, optional, defaults to 0):
                Model agnostic parameter to identify masked tokens when generating text in an MLM context.
            lang_id (:obj:`int`, optional, defaults to 1):
                The ID of the language used by the model. This parameter is used when generating
                text in a given language.

        Example::

            from transformers import XLMConfig, XLMModel

            # Initializing a XLM configuration
            configuration = XLMConfig()

            # Initializing a model from the configuration
            model = XLMModel(configuration)

            # Accessing the model configuration
            configuration = model.config
    """
    model_type = 'xlm'

    def __init__(self, vocab_size=30145, emb_dim=2048, n_layers=12, n_heads=16, dropout=0.1, attention_dropout=0.1, gelu_activation=True, sinusoidal_embeddings=False, causal=False, asm=False, n_langs=1, use_lang_emb=True, max_position_embeddings=512, embed_init_std=2048 ** -0.5, layer_norm_eps=1e-12, init_std=0.02, bos_index=0, eos_index=1, pad_index=2, unk_index=3, mask_index=5, is_encoder=True, summary_type='first', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, start_n_top=5, end_n_top=5, mask_token_id=0, lang_id=0, pad_token_id=2, bos_token_id=0, **kwargs):
        """Constructs XLMConfig.
        """
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.gelu_activation = gelu_activation
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.causal = causal
        self.asm = asm
        self.n_langs = n_langs
        self.use_lang_emb = use_lang_emb
        self.layer_norm_eps = layer_norm_eps
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.mask_index = mask_index
        self.is_encoder = is_encoder
        self.max_position_embeddings = max_position_embeddings
        self.embed_init_std = embed_init_std
        self.init_std = init_std
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.mask_token_id = mask_token_id
        self.lang_id = lang_id
        if 'n_words' in kwargs:
            self.n_words = kwargs['n_words']

    @property
    def n_words(self):
        return self.vocab_size

    @n_words.setter
    def n_words(self, value):
        self.vocab_size = value

    @property
    def hidden_size(self):
        return self.emb_dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers


XLM_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        langs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            A parallel sequence of tokens to be used to indicate the language of each token in the input.
            Indices are languages ids which can be obtained from the language names by using two conversion mappings
            provided in the configuration of the model (only provided for multilingual models).
            More precisely, the `language name -> language id` mapping is in `model.config.lang2id` (dict str -> int) and
            the `language id -> language name` mapping is `model.config.id2lang` (dict int -> str).

            See usage examples detailed in the `multilingual documentation <https://huggingface.co/transformers/multilingual.html>`__.
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        lengths (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Length of each sentence that can be used to avoid performing attention on padding token indices.
            You can also use `attention_mask` for the same result (see above), kept here for compatbility.
            Indices selected in ``[0, ..., input_ids.size(-1)]``:
        cache (:obj:`Dict[str, torch.FloatTensor]`, `optional`, defaults to :obj:`None`):
            dictionary with ``torch.FloatTensor`` that contains pre-computed
            hidden-states (key and values in the attention blocks) as computed by the model
            (see `cache` output below). Can be used to speed up sequential decoding.
            The dictionary object will be modified in-place during the forward pass to add newly computed hidden-states.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


XLM_START_DOCSTRING = """

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.XLMConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


def add_start_docstrings_to_callable(*docstr):

    def docstring_decorator(fn):
        class_name = ':class:`~transformers.{}`'.format(fn.__qualname__.split('.')[0])
        intro = '   The {} forward method, overrides the :func:`__call__` special method.'.format(class_name)
        note = '\n\n    .. note::\n        Although the recipe for forward pass needs to be defined within\n        this function, one should call the :class:`Module` instance afterwards\n        instead of this since the former takes care of running the\n        pre and post processing steps while the latter silently ignores them.\n        '
        fn.__doc__ = intro + note + ''.join(docstr) + (fn.__doc__ if fn.__doc__ is not None else '')
        return fn
    return docstring_decorator


def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    if padding_mask is not None:
        mask = padding_mask
    else:
        assert lengths.max().item() <= slen
        mask = alen < lengths[:, None]
    bs = lengths.size(0)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)
    return mask, attn_mask


class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config):
        super().__init__()
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        dim = config.emb_dim
        if config.asm is False:
            self.proj = nn.Linear(dim, config.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(in_features=dim, n_classes=config.n_words, cutoffs=config.asm_cutoffs, div_value=config.asm_div_value, head_bias=True)

    def forward(self, x, y=None):
        """ Compute the loss, and optionally the scores.
        """
        outputs = ()
        if self.asm is False:
            scores = self.proj(x)
            outputs = (scores,) + outputs
            if y is not None:
                loss = F.cross_entropy(scores.view(-1, self.n_words), y.view(-1), reduction='elementwise_mean')
                outputs = (loss,) + outputs
        else:
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs
        return outputs


XLNetLayerNorm = nn.LayerNorm


class XLNetRelativeAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        if config.d_model % config.n_head != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.d_model, config.n_head))
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / config.d_head ** 0.5
        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape
        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))
        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape
        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None, head_mask=None):
        """Core relative positional attention operations."""
        ac = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_w_bias, k_head_h)
        bd = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum('ijbs,ibns->bnij', seg_mat, ef)
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum('ijbn->bnij', attn_mask)
            else:
                attn_score = attn_score - 1e+30 * torch.einsum('ijbn->bnij', attn_mask)
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * torch.einsum('ijbn->bnij', head_mask)
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, v_head_h)
        if self.output_attentions:
            return attn_vec, torch.einsum('bnij->ijbn', attn_prob)
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.o)
        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def forward(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        if g is not None:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            attn_vec_h = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)
            if self.output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h
            output_h = self.post_attention(h, attn_vec_h)
            q_head_g = torch.einsum('ibh,hnd->ibnd', g, self.q)
            if target_mapping is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)
                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
                attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)
                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
            output_g = self.post_attention(g, attn_vec_g)
            if self.output_attentions:
                attn_prob = attn_prob_h, attn_prob_g
        else:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)
            attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)
            if self.output_attentions:
                attn_vec, attn_prob = attn_vec
            output_h = self.post_attention(h, attn_vec)
            output_g = None
        outputs = output_h, output_g
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class XLNetFeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        outputs = self.rel_attn(output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=mems, target_mapping=target_mapping, head_mask=head_mask)
        output_h, output_g = outputs[:2]
        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)
        outputs = (output_h, output_g) + outputs[2:]
        return outputs


class XLNetConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.XLNetModel`.
        It is used to instantiate an XLNet model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the `xlnet-large-cased <https://huggingface.co/xlnet-large-cased>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

        Args:
            vocab_size (:obj:`int`, optional, defaults to 32000):
                Vocabulary size of the XLNet model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.XLNetModel`.
            d_model (:obj:`int`, optional, defaults to 1024):
                Dimensionality of the encoder layers and the pooler layer.
            n_layer (:obj:`int`, optional, defaults to 24):
                Number of hidden layers in the Transformer encoder.
            n_head (:obj:`int`, optional, defaults to 16):
                Number of attention heads for each attention layer in the Transformer encoder.
            d_inner (:obj:`int`, optional, defaults to 4096):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            ff_activation (:obj:`string`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            untie_r (:obj:`boolean`, optional, defaults to :obj:`True`):
                Untie relative position biases
            attn_type (:obj:`string`, optional, defaults to "bi"):
                The attention type used by the model. Set 'bi' for XLNet, 'uni' for Transformer-XL.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            dropout (:obj:`float`, optional, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            mem_len (:obj:`int` or :obj:`None`, optional, defaults to :obj:`None`):
                The number of tokens to cache. The key/value pairs that have already been pre-computed
                in a previous forward pass won't be re-computed. See the
                `quickstart <https://huggingface.co/transformers/quickstart.html#using-the-past>`__
                for more information.
            reuse_len (:obj:`int` or :obj:`None`, optional, defaults to :obj:`None`):
                The number of tokens in the current batch to be cached and reused in the future.
            bi_data (:obj:`boolean`, optional, defaults to :obj:`False`):
                Whether to use bidirectional input pipeline. Usually set to `True` during
                pretraining and `False` during finetuning.
            clamp_len (:obj:`int`, optional, defaults to -1):
                Clamp all relative distances larger than clamp_len.
                Setting this attribute to -1 means no clamping.
            same_length (:obj:`boolean`, optional, defaults to :obj:`False`):
                Whether to use the same attention length for each token.
            summary_type (:obj:`string`, optional, defaults to "last"):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                Is one of the following options:
                    - 'last' => take the last token hidden state (like XLNet)
                    - 'first' => take the first token hidden state (like Bert)
                    - 'mean' => take the mean of all tokens hidden states
                    - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                    - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                Add a projection after the vector extraction
            summary_activation (:obj:`string` or :obj:`None`, optional, defaults to :obj:`None`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                'tanh' => add a tanh activation to the output, Other => no activation.
            summary_proj_to_labels (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_last_dropout (:obj:`float`, optional, defaults to 0.1):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                Add a dropout after the projection and activation
            start_n_top (:obj:`int`, optional, defaults to 5):
                Used in the SQuAD evaluation script for XLM and XLNet.
            end_n_top (:obj:`int`, optional, defaults to 5):
                Used in the SQuAD evaluation script for XLM and XLNet.

        Example::

            from transformers import XLNetConfig, XLNetModel

            # Initializing a XLNet configuration
            configuration = XLNetConfig()

            # Initializing a model from the configuration
            model = XLNetModel(configuration)

            # Accessing the model configuration
            configuration = model.config
    """
    model_type = 'xlnet'

    def __init__(self, vocab_size=32000, d_model=1024, n_layer=24, n_head=16, d_inner=4096, ff_activation='gelu', untie_r=True, attn_type='bi', initializer_range=0.02, layer_norm_eps=1e-12, dropout=0.1, mem_len=None, reuse_len=None, bi_data=False, clamp_len=-1, same_length=False, summary_type='last', summary_use_proj=True, summary_activation='tanh', summary_last_dropout=0.1, start_n_top=5, end_n_top=5, pad_token_id=5, bos_token_id=1, eos_token_id=2, **kwargs):
        """Constructs XLNetConfig.
        """
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        assert d_model % n_head == 0
        self.d_head = d_model // n_head
        self.ff_activation = ff_activation
        self.d_inner = d_inner
        self.untie_r = untie_r
        self.attn_type = attn_type
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    @property
    def max_position_embeddings(self):
        return -1

    @property
    def n_token(self):
        return self.vocab_size

    @n_token.setter
    def n_token(self, value):
        self.vocab_size = value

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


XLNET_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems
            given to this model should not be passed as input ids as they have already been computed.
            `use_cache` has to be set to `True` to make use of `mems`.
        perm_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to indicate the attention pattern for each input token with values selected in ``[0, 1]``:
            If ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
            if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.
            If None, each token attends to all the others (full bidirectional attention).
            Only used during pretraining (to define factorization order) or for sequential decoding (generation).
        target_mapping (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to indicate the output tokens to use.
            If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
            Only used during pretraining for partial prediction or for sequential decoding (generation).
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token. The classifier token should be represented by a ``2``.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        input_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Negative of `attention_mask`, i.e. with 0 for real tokens and 1 for padding.
            Kept for compatibility with the original code base.
            You can only uses one of `input_mask` and `attention_mask`
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are MASKED, ``0`` for tokens that are NOT MASKED.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        use_cache (:obj:`bool`):
            If `use_cache` is True, `mems` are returned and can be used to speed up decoding (see `mems`). Defaults to `True`.
"""


XLNET_START_DOCSTRING = """

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.XLNetConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    """ A map of modules from TF to PyTorch.
        I use a map to keep the PyTorch model as
        identical to the original PyTorch model as possible.
    """
    tf_to_pt_map = {}
    if hasattr(model, 'transformer'):
        if hasattr(model, 'lm_loss'):
            tf_to_pt_map['model/lm_loss/bias'] = model.lm_loss.bias
        if hasattr(model, 'sequence_summary') and 'model/sequnece_summary/summary/kernel' in tf_weights:
            tf_to_pt_map['model/sequnece_summary/summary/kernel'] = model.sequence_summary.summary.weight
            tf_to_pt_map['model/sequnece_summary/summary/bias'] = model.sequence_summary.summary.bias
        if hasattr(model, 'logits_proj') and config.finetuning_task is not None and 'model/regression_{}/logit/kernel'.format(config.finetuning_task) in tf_weights:
            tf_to_pt_map['model/regression_{}/logit/kernel'.format(config.finetuning_task)] = model.logits_proj.weight
            tf_to_pt_map['model/regression_{}/logit/bias'.format(config.finetuning_task)] = model.logits_proj.bias
        model = model.transformer
    tf_to_pt_map.update({'model/transformer/word_embedding/lookup_table': model.word_embedding.weight, 'model/transformer/mask_emb/mask_emb': model.mask_emb})
    for i, b in enumerate(model.layer):
        layer_str = 'model/transformer/layer_%d/' % i
        tf_to_pt_map.update({(layer_str + 'rel_attn/LayerNorm/gamma'): b.rel_attn.layer_norm.weight, (layer_str + 'rel_attn/LayerNorm/beta'): b.rel_attn.layer_norm.bias, (layer_str + 'rel_attn/o/kernel'): b.rel_attn.o, (layer_str + 'rel_attn/q/kernel'): b.rel_attn.q, (layer_str + 'rel_attn/k/kernel'): b.rel_attn.k, (layer_str + 'rel_attn/r/kernel'): b.rel_attn.r, (layer_str + 'rel_attn/v/kernel'): b.rel_attn.v, (layer_str + 'ff/LayerNorm/gamma'): b.ff.layer_norm.weight, (layer_str + 'ff/LayerNorm/beta'): b.ff.layer_norm.bias, (layer_str + 'ff/layer_1/kernel'): b.ff.layer_1.weight, (layer_str + 'ff/layer_1/bias'): b.ff.layer_1.bias, (layer_str + 'ff/layer_2/kernel'): b.ff.layer_2.weight, (layer_str + 'ff/layer_2/bias'): b.ff.layer_2.bias})
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    tf_to_pt_map.update({'model/transformer/r_r_bias': r_r_list, 'model/transformer/r_w_bias': r_w_list, 'model/transformer/r_s_bias': r_s_list, 'model/transformer/seg_embed': seg_embed_list})
    return tf_to_pt_map


def load_tf_weights_in_xlnet(model, config, tf_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)
    for name, pointer in tf_to_pt_map.items():
        logger.info('Importing {}'.format(name))
        if name not in tf_weights:
            logger.info('{} not in tf pre-trained weights, skipping'.format(name))
            continue
        array = tf_weights[name]
        if 'kernel' in name and ('ff' in name or 'summary' in name or 'logit' in name):
            logger.info('Transposing')
            array = np.transpose(array)
        if isinstance(pointer, list):
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += p_i.shape, arr_i.shape
                    raise
                logger.info('Initialize PyTorch weight {} for layer {}'.format(name, i))
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += pointer.shape, array.shape
                raise
            logger.info('Initialize PyTorch weight {}'.format(name))
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)
    logger.info('Weights not copied to PyTorch model: {}'.format(', '.join(tf_weights.keys())))
    return model


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def set_cuda(var, cuda):
    if cuda:
        return var
    return var


class CRFLoss(nn.Module):
    """
    Calculate log-space crf loss, given unary potentials, a transition matrix
    and gold tag sequences.
    """

    def __init__(self, num_tag, batch_average=True):
        super().__init__()
        self._transitions = nn.Parameter(torch.zeros(num_tag, num_tag))
        self._batch_average = batch_average

    def forward(self, inputs, masks, tag_indices):
        """
        inputs: batch_size x seq_len x num_tags
        masks: batch_size x seq_len
        tag_indices: batch_size x seq_len
        @return:
            loss: CRF negative log likelihood on all instances.
            transitions: the transition matrix
        """
        self.bs, self.sl, self.nc = inputs.size()
        unary_scores = self.crf_unary_score(inputs, masks, tag_indices)
        binary_scores = self.crf_binary_score(inputs, masks, tag_indices)
        log_norm = self.crf_log_norm(inputs, masks, tag_indices)
        log_likelihood = unary_scores + binary_scores - log_norm
        loss = torch.sum(-log_likelihood)
        if self._batch_average:
            loss = loss / self.bs
        else:
            total = masks.eq(0).sum()
            loss = loss / (total + 1e-08)
        return loss, self._transitions

    def crf_unary_score(self, inputs, masks, tag_indices):
        """
        @return:
            unary_scores: batch_size
        """
        flat_inputs = inputs.view(self.bs, -1)
        flat_tag_indices = tag_indices + set_cuda(torch.arange(self.sl).long().unsqueeze(0) * self.nc, tag_indices.is_cuda)
        unary_scores = torch.gather(flat_inputs, 1, flat_tag_indices).view(self.bs, -1)
        unary_scores.masked_fill_(masks, 0)
        return unary_scores.sum(dim=1)

    def crf_binary_score(self, inputs, masks, tag_indices):
        """
        @return:
            binary_scores: batch_size
        """
        nt = tag_indices.size(-1) - 1
        start_indices = tag_indices[:, :nt]
        end_indices = tag_indices[:, 1:]
        flat_transition_indices = start_indices * self.nc + end_indices
        flat_transition_indices = flat_transition_indices.view(-1)
        flat_transition_matrix = self._transitions.view(-1)
        binary_scores = torch.gather(flat_transition_matrix, 0, flat_transition_indices).view(self.bs, -1)
        score_masks = masks[:, 1:]
        binary_scores.masked_fill_(score_masks, 0)
        return binary_scores.sum(dim=1)

    def crf_log_norm(self, inputs, masks, tag_indices):
        """
        Calculate the CRF partition in log space for each instance, following:
            http://www.cs.columbia.edu/~mcollins/fb.pdf
        @return:
            log_norm: batch_size
        """
        start_inputs = inputs[:, 0, :]
        rest_inputs = inputs[:, 1:, :]
        rest_masks = masks[:, 1:]
        alphas = start_inputs
        trans = self._transitions.unsqueeze(0)
        for i in range(rest_inputs.size(1)):
            transition_scores = alphas.unsqueeze(2) + trans
            new_alphas = rest_inputs[:, i, :] + log_sum_exp(transition_scores, dim=1)
            m = rest_masks[:, i].unsqueeze(1).expand_as(new_alphas)
            new_alphas.masked_scatter_(m, alphas.masked_select(m))
            alphas = new_alphas
        log_norm = log_sum_exp(alphas, dim=1)
        return log_norm


INFINITY_NUMBER = 65504


class BasicAttention(nn.Module):
    """
    A basic MLP attention layer.
    """

    def __init__(self, dim):
        super(BasicAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        target = self.linear_in(input)
        source = self.linear_c(context.contiguous().view(-1, dim)).view(batch_size, source_len, dim)
        attn = target.unsqueeze(1).expand_as(context) + source
        attn = self.tanh(attn)
        attn = self.linear_v(attn.view(-1, dim)).view(batch_size, source_len)
        if mask is not None:
            attn.masked_fill_(mask, -INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)
        attn = torch.bmm(context, target).squeeze(2)
        if mask is not None:
            assert mask.size() == attn.size(), 'Mask size must match the attention size!'
            attn.masked_fill_(mask, -INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LinearAttention(nn.Module):
    """ A linear attention form, inspired by BiDAF:
        a = W (u; v; u o v)
    """

    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.linear = nn.Linear(dim * 3, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)
        v = context.contiguous().view(-1, dim)
        attn_in = torch.cat((u, v, u.mul(v)), 1)
        attn = self.linear(attn_in).view(batch_size, source_len)
        if mask is not None:
            assert mask.size() == attn.size(), 'Mask size must match the attention size!'
            attn.masked_fill_(mask, -INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        attn3 = attn.view(batch_size, 1, source_len)
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class DeepAttention(nn.Module):
    """ A deep attention form, invented by Robert:
        u = ReLU(Wx)
        v = ReLU(Wy)
        a = V.(u o v)
    """

    def __init__(self, dim):
        super(DeepAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)
        u = self.relu(self.linear_in(u))
        v = self.relu(self.linear_in(context.contiguous().view(-1, dim)))
        attn = self.linear_v(u.mul(v)).view(batch_size, source_len)
        if mask is not None:
            assert mask.size() == attn.size(), 'Mask size must match the attention size!'
            attn.masked_fill_(mask, -INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        attn3 = attn.view(batch_size, 1, source_len)
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LSTMAttention(nn.Module):
    """A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='soft'):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        if attn_type == 'soft':
            self.attention_layer = SoftDotAttention(hidden_size)
        elif attn_type == 'mlp':
            self.attention_layer = BasicAttention(hidden_size)
        elif attn_type == 'linear':
            self.attention_layer = LinearAttention(hidden_size)
        elif attn_type == 'deep':
            self.attention_layer = DeepAttention(hidden_size)
        else:
            raise Exception('Unsupported LSTM attention type: {}'.format(attn_type))

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        if self.batch_first:
            input = input.transpose(0, 1)
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.lstm_cell(input[i], hidden)
            hy, cy = hidden
            h_tilde, alpha = self.attention_layer(hy, ctx, mask=ctx_mask)
            output.append(h_tilde)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, hidden


EOS_ID = 3


PAD_ID = 0


SOS_ID = 2


class Beam:
    """
     Adapted and modified from the OpenNMT project.

     Class for managing the internals of the beam search process.


             hyp1-hyp1---hyp1 -hyp1
                     \\             /
             hyp2 \\-hyp2 /-hyp2hyp2
                                   /                   hyp3-hyp3---hyp3 -hyp3
             ========================

     Takes care of beams, back pointers, and scores.
    """

    def __init__(self, size, cuda=False):
        self.size = size
        self.done = False
        self.tt = torch.cuda if cuda else torch
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []
        self.prevKs = []
        self.nextYs = [self.tt.LongTensor(size).fill_(PAD_ID)]
        self.nextYs[0][0] = SOS_ID
        self.copy = []

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.nextYs[-1]

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prevKs[-1]

    def advance(self, wordLk, copy_indices=None):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `copy_indices` - copy indices (K x ctx_len)

        Returns: True if beam search is complete.
        """
        if self.done:
            return True
        numWords = wordLk.size(1)
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        if copy_indices is not None:
            self.copy.append(copy_indices.index_select(0, prevK))
        if self.nextYs[-1][0] == EOS_ID:
            self.done = True
            self.allScores.append(self.scores)
        return self.done

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """
        Walk back to construct the full hypothesis.

        Parameters:

             * `k` - the position in the beam to construct.

         Returns: The hypothesis
        """
        hyp = []
        cpy = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            if len(self.copy) > 0:
                cpy.append(self.copy[j][k])
            k = self.prevKs[j][k]
        hyp = hyp[::-1]
        cpy = cpy[::-1]
        for i, cidx in enumerate(cpy):
            if cidx >= 0:
                hyp[i] = -(cidx + 1)
        return hyp


EMB_INIT_RANGE = 1.0


def prune_hyp(hyp):
    """
    Prune a decoded hypothesis
    """
    if EOS_ID in hyp:
        idx = hyp.index(EOS_ID)
        return hyp[:idx]
    else:
        return hyp


class Seq2SeqModel(nn.Module):
    """
    A complete encoder-decoder model, with optional attention.
    """

    def __init__(self, args, emb_matrix=None, use_cuda=False, training_mode=False):
        super().__init__()
        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']
        self.hidden_dim = args['hidden_dim']
        self.nlayers = args['num_layers']
        self.emb_dropout = args.get('emb_dropout', 0.0)
        self.dropout = args['dropout']
        self.pad_token = PAD_ID
        self.max_dec_len = args['max_dec_len']
        self.use_cuda = use_cuda
        self.training_mode = training_mode
        self.top = args.get('top', 10000000000.0)
        self.args = args
        self.emb_matrix = emb_matrix
        self.num_directions = 2
        self.enc_hidden_dim = self.hidden_dim // 2
        self.dec_hidden_dim = self.hidden_dim
        self.use_pos = args.get('pos', False)
        self.pos_dim = args.get('pos_dim', 0)
        self.pos_vocab_size = args.get('pos_vocab_size', 0)
        self.pos_dropout = args.get('pos_dropout', 0)
        self.edit = args.get('edit', False)
        self.num_edit = args.get('num_edit', 0)
        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.drop = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_token)
        self.encoder = nn.LSTM(self.emb_dim, self.enc_hidden_dim, self.nlayers, bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.decoder = LSTMAttention(self.emb_dim, self.dec_hidden_dim, batch_first=True, attn_type=self.args['attn_type'])
        self.dec2vocab = nn.Linear(self.dec_hidden_dim, self.vocab_size)
        if self.use_pos and self.pos_dim > 0:
            self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_dim, self.pad_token)
            self.pos_drop = nn.Dropout(self.pos_dropout)
        if self.edit:
            edit_hidden = self.hidden_dim // 2
            self.edit_clf = nn.Sequential(nn.Linear(self.hidden_dim, edit_hidden), nn.ReLU(), nn.Linear(edit_hidden, self.num_edit))
        self.SOS_tensor = torch.LongTensor([SOS_ID])
        self.SOS_tensor = self.SOS_tensor if self.use_cuda else self.SOS_tensor
        self.init_weights()

    def init_weights(self):
        init_range = EMB_INIT_RANGE
        if self.emb_matrix is not None:
            if isinstance(self.emb_matrix, np.ndarray):
                self.emb_matrix = torch.from_numpy(self.emb_matrix)
            assert self.emb_matrix.size() == (self.vocab_size, self.emb_dim), 'Input embedding matrix must match size: {} x {}'.format(self.vocab_size, self.emb_dim)
            self.embedding.weight.data.copy_(self.emb_matrix)
        else:
            self.embedding.weight.data.uniform_(-init_range, init_range)
        if self.use_pos:
            self.pos_embedding.weight.data.uniform_(-init_range, init_range)

    def cuda(self):
        super()
        self.use_cuda = True

    def cpu(self):
        super().cpu()
        self.use_cuda = False

    def zero_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = torch.zeros(self.encoder.num_layers * 2, batch_size, self.enc_hidden_dim, requires_grad=False)
        c0 = torch.zeros(self.encoder.num_layers * 2, batch_size, self.enc_hidden_dim, requires_grad=False)
        if self.use_cuda:
            if self.training_mode:
                return h0, c0
            else:
                return h0.half(), c0.half()
        return h0, c0

    def encode(self, enc_inputs, lens):
        """ Encode source sequence. """
        self.h0, self.c0 = self.zero_state(enc_inputs)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(enc_inputs, lens, batch_first=True)
        packed_h_in, (hn, cn) = self.encoder(packed_inputs, (self.h0, self.c0))
        h_in, _ = nn.utils.rnn.pad_packed_sequence(packed_h_in, batch_first=True)
        hn = torch.cat((hn[-1], hn[-2]), 1)
        cn = torch.cat((cn[-1], cn[-2]), 1)
        return h_in, (hn, cn)

    def decode(self, dec_inputs, hn, cn, ctx, ctx_mask=None):
        """ Decode a step, based on context encoding and source context states."""
        dec_hidden = hn, cn
        h_out, dec_hidden = self.decoder(dec_inputs, dec_hidden, ctx, ctx_mask)
        h_out_reshape = h_out.contiguous().view(h_out.size(0) * h_out.size(1), -1)
        decoder_logits = self.dec2vocab(h_out_reshape)
        decoder_logits = decoder_logits.view(h_out.size(0), h_out.size(1), -1)
        log_probs = self.get_log_prob(decoder_logits)
        return log_probs, dec_hidden

    def forward(self, src, src_mask, tgt_in, pos=None):
        batch_size = src.size(0)
        enc_inputs = self.emb_drop(self.embedding(src))
        dec_inputs = self.emb_drop(self.embedding(tgt_in))
        if self.use_pos:
            assert pos is not None, 'Missing POS input for seq2seq lemmatizer.'
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(0).long().sum(1))
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)
        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None
        log_probs, _ = self.decode(dec_inputs, hn, cn, h_in, src_mask)
        return log_probs, edit_logits

    def get_log_prob(self, logits):
        logits_reshape = logits.view(-1, self.vocab_size)
        log_probs = F.log_softmax(logits_reshape, dim=1)
        if logits.dim() == 2:
            return log_probs
        return log_probs.view(logits.size(0), logits.size(1), logits.size(2))

    def predict_greedy(self, src, src_mask, pos=None):
        """ Predict with greedy decoding. """
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, 'Missing POS input for seq2seq lemmatizer.'
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(PAD_ID).long().sum(1))
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)
        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None
        dec_inputs = self.embedding(self.SOS_tensor)
        dec_inputs = dec_inputs.expand(batch_size, dec_inputs.size(0), dec_inputs.size(1))
        done = [(False) for _ in range(batch_size)]
        total_done = 0
        max_len = 0
        output_seqs = [[] for _ in range(batch_size)]
        while total_done < batch_size and max_len < self.max_dec_len:
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask)
            assert log_probs.size(1) == 1, 'Output must have 1-step of output.'
            _, preds = log_probs.squeeze(1).max(1, keepdim=True)
            dec_inputs = self.embedding(preds)
            max_len += 1
            for i in range(batch_size):
                if not done[i]:
                    token = preds.data[i][0].item()
                    if token == EOS_ID:
                        done[i] = True
                        total_done += 1
                    else:
                        output_seqs[i].append(token)
        return output_seqs, edit_logits

    def predict(self, src, src_mask, pos=None, beam_size=5):
        """ Predict with beam search. """
        if beam_size == 1:
            return self.predict_greedy(src, src_mask, pos=pos)
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, 'Missing POS input for seq2seq lemmatizer.'
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(PAD_ID).long().sum(1))
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)
        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None
        with torch.no_grad():
            h_in = h_in.data.repeat(beam_size, 1, 1)
            src_mask = src_mask.repeat(beam_size, 1)
            hn = hn.data.repeat(beam_size, 1)
            cn = cn.data.repeat(beam_size, 1)
        beam = [Beam(beam_size, self.use_cuda) for _ in range(batch_size)]

        def update_state(states, idx, positions, beam_size):
            """ Select the states according to back pointers. """
            for e in states:
                br, d = e.size()
                s = e.contiguous().view(beam_size, br // beam_size, d)[:, idx]
                s.data.copy_(s.data.index_select(0, positions))
        for i in range(self.max_dec_len):
            dec_inputs = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1, 1)
            dec_inputs = self.embedding(dec_inputs)
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask)
            log_probs = log_probs.view(beam_size, batch_size, -1).transpose(0, 1).contiguous()
            done = []
            for b in range(batch_size):
                is_done = beam[b].advance(log_probs.data[b])
                if is_done:
                    done += [b]
                update_state((hn, cn), b, beam[b].get_current_origin(), beam_size)
            if len(done) == batch_size:
                break
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[0]]
            k = ks[0]
            hyp = beam[b].get_hyp(k)
            hyp = prune_hyp(hyp)
            hyp = [i.item() for i in hyp]
            all_hyp += [hyp]
        return all_hyp, edit_logits


ADAPTER_HUB_URL = 'https://raw.githubusercontent.com/Adapter-Hub/Hub/master/dist/'


ADAPTER_HUB_CONFIG_FILE = ADAPTER_HUB_URL + 'architectures.json'


def download_cached(url, checksum=None, checksum_algo='sha1', cache_dir=None, force_extract=False, **kwargs):
    if not cache_dir:
        cache_dir = ADAPTER_CACHE
    if isinstance(url, Path):
        url = str(url)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if is_remote_url(url):
        output_path = get_from_cache(url, cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError("Unable to parse '{}' as a URL".format(url))
    if not output_path:
        return None
    if checksum and checksum_algo:
        h = hashlib.new(checksum_algo)
        with open(output_path, 'rb') as f:
            h.update(f.read())
        calculated_checksum = h.hexdigest()
        if calculated_checksum != checksum.lower():
            raise EnvironmentError("Failed to verify checksum of '{}'".format(output_path))
    if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
        return output_path
    output_dir, output_file = os.path.split(output_path)
    output_extract_dir_name = output_file.replace('.', '-') + '-extracted'
    output_path_extracted = os.path.join(output_dir, output_extract_dir_name)
    if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
        return output_path_extracted
    lock_path = output_path + '.lock'
    with FileLock(lock_path):
        shutil.rmtree(output_path_extracted, ignore_errors=True)
        os.makedirs(output_path_extracted)
        if is_zipfile(output_path):
            with ZipFile(output_path, 'r') as zip_file:
                for file in zip_file.namelist():
                    if basename(file):
                        file_data = zip_file.read(file)
                        with open(join(output_path_extracted, basename(file)), 'wb') as f:
                            f.write(file_data)
        elif tarfile.is_tarfile(output_path):
            tar_file = tarfile.open(output_path)
            tar_file.extractall(output_path_extracted)
            tar_file.close()
        else:
            raise EnvironmentError('Archive format of {} could not be identified'.format(output_path))
    return output_path_extracted


def resolve_adapter_config(config: Union[dict, str], local_map=None, try_loading_from_hub=True, **kwargs) ->dict:
    """Resolves a given adapter configuration specifier to a full configuration dictionary.

    Args:
        config (Union[dict, str]): The configuration to resolve. Can be either:
            - a dictionary: returned without further action
            - an identifier string available in local_map
            - the path to a file containing a full adapter configuration
            - an identifier string available in Adapter-Hub

    Returns:
        dict: The resolved adapter configuration dictionary.
    """
    if isinstance(config, Mapping):
        return config
    if local_map and config in local_map:
        return local_map[config]
    if isfile(config):
        with open(config, 'r') as f:
            loaded_config = json.load(f)
            if 'config' in loaded_config:
                return loaded_config['config']
            else:
                return loaded_config
    if try_loading_from_hub:
        index_file = download_cached(ADAPTER_HUB_CONFIG_FILE, **kwargs)
        if not index_file:
            raise EnvironmentError('Unable to load adapter hub index file. The file might be temporarily unavailable.')
        with open(index_file, 'r') as f:
            config_index = json.load(f)
        if config in config_index:
            return config_index[config]
    raise ValueError("Could not identify '{}' as a valid adapter configuration.".format(config))


BERT_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, embedding_dim)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        training (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether to activate dropout modules (if set to :obj:`True`) during training or to de-activate them
            (if set to :obj:`False`) for evaluation.
"""


BERT_START_DOCSTRING = """
    This model is a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ sub-class.
    Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

            - having all inputs as keyword arguments (like PyTorch models), or
            - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :obj:`tf.keras.Model.fit()` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with input_ids only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({'input_ids': input_ids, 'token_type_ids': token_type_ids})`

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


DEFAULT_ADAPTER_CONFIG = 'pfeiffer'


ADAPTERFUSION_CONFIG_NAME = 'adapter_fusion_config.json'


ADAPTERFUSION_WEIGHTS_NAME = 'pytorch_model_adapter_fusion.bin'


class WeightsLoaderHelper:
    """
    A class providing helper methods for saving and loading module weights.
    """

    def __init__(self, model, weights_name, config_name):
        self.model = model
        self.weights_name = weights_name
        self.config_name = config_name

    def state_dict(self, filter_func):
        return {k: v for k, v in self.model.state_dict().items() if filter_func(k)}

    def rename_state_dict(self, state_dict, rename_func):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = rename_func(k)
            new_state_dict[new_k] = v
        return new_state_dict

    def save_weights_config(self, save_directory, config, meta_dict=None):
        if meta_dict:
            for k, v in meta_dict.items():
                if k not in config:
                    config[k] = v
        output_config_file = join(save_directory, self.config_name)
        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, sort_keys=True)
        logger.info('Configuration saved in {}'.format(output_config_file))

    def save_weights(self, save_directory, filter_func):
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), 'Saving path should be a directory where the module weights can be saved.'
        state_dict = self.state_dict(filter_func)
        output_file = join(save_directory, self.weights_name)
        torch.save(state_dict, output_file)
        logger.info('Module weights saved in {}'.format(output_file))

    def load_weights_config(self, save_directory):
        config_file = join(save_directory, self.config_name)
        logger.info('Loading module configuration from {}'.format(config_file))
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        return loaded_config

    @staticmethod
    def _load_module_state_dict(module, state_dict, start_prefix=''):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(module, prefix=start_prefix)
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(module.__class__.__name__, '\n\t'.join(error_msgs)))
        return missing_keys, unexpected_keys

    def load_weights(self, save_directory, filter_func, rename_func=None, loading_info=None, in_base_model=False):
        weights_file = join(save_directory, self.weights_name)
        try:
            state_dict = torch.load(weights_file, map_location='cpu')
        except Exception:
            raise OSError('Unable to load weights from pytorch checkpoint file. ')
        if rename_func:
            state_dict = self.rename_state_dict(state_dict, rename_func)
        logger.info('Loading module weights from {}'.format(weights_file))
        start_prefix = ''
        model_to_load = self.model
        has_prefix_module = any(s.startswith(self.model.base_model_prefix) for s in state_dict.keys())
        if not hasattr(self.model, self.model.base_model_prefix) and has_prefix_module:
            start_prefix = self.model.base_model_prefix + '.'
        if in_base_model and hasattr(self.model, self.model.base_model_prefix) and not has_prefix_module:
            model_to_load = self.model.base_model
        missing_keys, unexpected_keys = self._load_module_state_dict(model_to_load, state_dict, start_prefix=start_prefix)
        missing_keys = [k for k in missing_keys if filter_func(k)]
        if len(missing_keys) > 0:
            logger.info('Some module weights could not be found in loaded weights file: {}'.format(', '.join(missing_keys)))
        if len(unexpected_keys) > 0:
            logger.info('Some weights of the state_dict could not be loaded into model: {}'.format(', '.join(unexpected_keys)))
        if isinstance(loading_info, dict):
            if 'missing_keys' not in loading_info:
                loading_info['missing_keys'] = []
            if 'unexpected_keys' not in loading_info:
                loading_info['unexpected_keys'] = []
            loading_info['missing_keys'].extend(missing_keys)
            loading_info['unexpected_keys'].extend(unexpected_keys)
        return missing_keys, unexpected_keys


def build_full_config(adapter_config, model_config, **kwargs):
    config_dict = {'model_type': model_config.model_type, 'hidden_size': model_config.hidden_size}
    config_dict.update(kwargs)
    if is_dataclass(adapter_config):
        config_dict['config'] = adapter_config.to_dict()
    else:
        config_dict['config'] = adapter_config
    return config_dict


class WeightsLoader(ABC):
    """
    An abstract class providing basic methods for saving and loading weights of a model.
    Extend this class to build custom module weight loaders.
    """

    def __init__(self, model, weights_name, config_name):
        self.model = model
        self.weights_helper = WeightsLoaderHelper(model, weights_name, config_name)

    @abstractmethod
    def filter_func(self, name: str) ->Callable[[str], bool]:
        """The callable returned by this method is used to extract the module weights to be saved or loaded
        based on their names.

        Args:
            name (str): An identifier of the weights to be saved.

        Returns:
            Callable[str, bool]: A function that takes the fully qualified name of a module parameter and returns
                                a boolean value that specifies whether this parameter should be extracted.
        """
        pass

    @abstractmethod
    def rename_func(self, old_name: str, new_name: str) ->Callable[[str], str]:
        """The callable returned by this method is used to optionally rename the module weights after loading.

        Args:
            old_name (str): The string identifier of the weights as loaded from file.
            new_name (str): The new string identifier to which the weights should be renamed.

        Returns:
            Callable[str, str]: A function that takes the fully qualified name of a module parameter and returns
                                a new fully qualified name.
        """
        pass

    def save(self, save_directory, name, **kwargs):
        """Saves the module config and weights into the given directory.
        Override this method for additional saving actions.

        Args:
            save_directory (str): The directory to save the weights in.
            name (str): An identifier of the weights to be saved. The details are specified by the implementor.
        """
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), 'Saving path should be a directory where weights and configuration can be saved.'
        config_dict = build_full_config(None, self.model.config, model_name=self.model.model_name, name=name, model_class=self.model.__class__.__name__)
        meta_dict = kwargs.pop('meta_dict', None)
        self.weights_helper.save_weights_config(save_directory, config_dict, meta_dict=meta_dict)
        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(self, save_directory, load_as=None, loading_info=None, **kwargs) ->Tuple[str, str]:
        """Loads the module weights from the given directory.
        Override this method for additional loading actions. If adding the loaded weights
        to the model passed to the loader class requires adding additional modules, this method should also perform the
        architectural changes to the model.

        Args:
            save_directory (str): The directory from where to load the weights.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
                             and the name of the loaded weights.
        """
        if not exists(join(save_directory, self.weights_helper.weights_name)):
            raise ValueError('Loading path should be a directory where the weights are saved.')
        config = self.weights_helper.load_weights_config(save_directory)
        filter_func = self.filter_func(config['name'])
        if load_as:
            rename_func = self.rename_func(config['name'], load_as)
        else:
            rename_func = None
        self.weights_helper.load_weights(save_directory, filter_func, rename_func=rename_func, loading_info=loading_info)
        return save_directory, load_as or config['name']


class AdapterFusionLoader(WeightsLoader):
    """
    A class providing methods for saving and loading AdapterFusion modules from the file system.

    """

    def __init__(self, model, error_on_missing=True):
        super().__init__(model, ADAPTERFUSION_WEIGHTS_NAME, ADAPTERFUSION_CONFIG_NAME)
        self.error_on_missing = error_on_missing

    def filter_func(self, adapter_fusion_name):
        return lambda x: 'adapter_fusion_layer.{}'.format(adapter_fusion_name) in x

    def rename_func(self, old_name, new_name):
        return lambda k: k.replace('adapter_fusion_layer.{}'.format(old_name), 'adapter_fusion_layer.{}'.format(new_name))

    def save(self, save_directory: str, name: str):
        """Saves a AdapterFusion module into the given directory.

        Args:
            save_directory (str): The directory to save the weights in.
            name (str, optional): The AdapterFusion name.
        """
        if hasattr(self.model.config, 'adapter_fusion_models'):
            if name not in self.model.config.adapter_fusion_models:
                if self.error_on_missing:
                    raise ValueError(f"Unknown AdapterFusion '{name}'.")
                else:
                    logger.debug(f"No AdapterFusion with name '{name}' available.")
                    return
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), 'Saving path should be a directory where the head can be saved.'
        adapter_fusion_config = self.model.config.adapter_fusion
        config_dict = build_full_config(adapter_fusion_config, self.model.config, name=name, model_name=self.model.model_name, model_class=self.model.__class__.__name__)
        self.weights_helper.save_weights_config(save_directory, config_dict)
        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(self, save_directory, load_as=None, loading_info=None):
        """Loads a AdapterFusion module from the given directory.

        Args:
            save_directory (str): The directory from where to load the weights.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
                             and the name of the loaded weights.
        """
        if not exists(join(save_directory, ADAPTERFUSION_WEIGHTS_NAME)):
            if self.error_on_missing:
                raise ValueError('Loading path should be a directory where AdapterFusion is saved.')
            else:
                logger.debug("No matching adapter fusion found in '{}'".format(save_directory))
                return None, None
        config = self.weights_helper.load_weights_config(save_directory)
        if not hasattr(self.model.config, 'adapter_fusion_models'):
            self.model.config.adapter_fusion_models = []
        adapter_fusion_name = load_as or config['name']
        if adapter_fusion_name in self.model.config.adapter_fusion_models:
            logger.warning("Overwriting existing adapter fusion module '{}'".format(adapter_fusion_name))
        self.model.add_fusion(adapter_fusion_name, config['config'])
        filter_func = self.filter_func(adapter_fusion_name)
        if load_as:
            rename_func = self.rename_func(config['name'], load_as)
        else:
            rename_func = None
        self.weights_helper.load_weights(save_directory, filter_func, rename_func=rename_func, loading_info=loading_info)
        return save_directory, adapter_fusion_name


ADAPTER_IDENTIFIER_PATTERN = '[0-9a-zA-Z\\-_\\/@]{2,}'


ADAPTER_HUB_INDEX_FILE = ADAPTER_HUB_URL + 'index_{}/{}.json'


def _dict_extract(d, primary_key, secondary_key=None):
    for k, v in d.items():
        if k == primary_key:
            if secondary_key:
                if secondary_key in v.keys():
                    yield v[secondary_key]
            else:
                for k, v in v.items():
                    yield v
        else:
            for k, v in v.items():
                if k == primary_key:
                    yield v


def _get_matching_version(config_entry, org):
    if org:
        return config_entry['versions'].get(org, None)
    elif len(config_entry['versions']) == 1:
        return list(config_entry['versions'].values())[0]
    elif 'default' in config_entry:
        return config_entry['default']
    else:
        raise ValueError('Multiple adapters with this name are available for this config.')


def _split_identifier(identifier):
    task, subtask, org_name = None, None, None
    identifier = identifier.split('@')
    if len(identifier) > 1:
        org_name = identifier[1]
    identifier = identifier[0].split('/')
    if len(identifier) > 1:
        subtask = identifier[1]
    task = identifier[0]
    return task, subtask, org_name


ADAPTER_CONFIG_HASH_IGNORE = []


def _minimize_dict(d):
    if isinstance(d, Mapping):
        return {k: _minimize_dict(v) for k, v in d.items() if v}
    else:
        return d


def get_adapter_config_hash(config, length=16):
    """Calculates the hash of a given adapter configuration which is used to identify this configuration.

    Returns:
        str: The resulting hash of the given config dict.
    """
    minimized_config = _minimize_dict({k: v for k, v in config.items() if k not in ADAPTER_CONFIG_HASH_IGNORE})
    dict_str = json.dumps(minimized_config, sort_keys=True)
    h = hashlib.sha1()
    h.update(dict_str.encode(encoding='utf-8'))
    return h.hexdigest()[:length]


def find_in_index(identifier: str, adapter_type: AdapterType, model_name: str, adapter_config: Optional[dict]=None, strict: bool=False, index_file: str=None) ->Optional[str]:
    if not index_file:
        index_file = download_cached(ADAPTER_HUB_INDEX_FILE.format(adapter_type, model_name))
    if not index_file:
        raise EnvironmentError('Unable to load adapter hub index file. The file might be temporarily unavailable.')
    with open(index_file, 'r') as f:
        adapter_index = json.load(f)
    task, subtask, org = _split_identifier(identifier)
    entries = list(_dict_extract(adapter_index, task, subtask))
    if not entries:
        return None
    elif len(entries) == 1:
        index_entry = entries[0]
    else:
        raise ValueError("Found multiple possible adapters matching '{}'.".format(identifier))
    if adapter_config:
        config_hash = get_adapter_config_hash(adapter_config)
        if config_hash in index_entry:
            hub_entry = _get_matching_version(index_entry[config_hash], org)
            if hub_entry:
                logger.info('Found matching adapter at: {}'.format(hub_entry))
            return hub_entry
    if not adapter_config or not strict:
        if 'default' in index_entry:
            logger.info('No exactly matching adapter config found for this specifier, falling back to default.')
            return index_entry['default']
        elif len(index_entry) == 1:
            logger.info('Only one configuration available for this adapter, using default.')
            config_entry = list(index_entry.values())[0]
            return _get_matching_version(config_entry, org)
    raise ValueError("No adapter '{}' found for the current model or configuration.".format(identifier))


def get_checksum(file_entry: dict):
    for algo in hashlib.algorithms_guaranteed:
        if algo in file_entry:
            return algo, file_entry[algo]


def urljoin(*args):
    return '/'.join([s.strip('/') for s in args])


def http_get_json(url):
    if not urlparse(url).netloc:
        url = urljoin(ADAPTER_HUB_URL, url)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise EnvironmentError('Failed to get file {}'.format(url))


def pull_from_hub(specifier: str, adapter_type: AdapterType, model_name: str, adapter_config: Optional[Union[dict, str]]=None, version: str=None, strict: bool=False, **kwargs) ->str:
    """Downloads a pre-trained adapter module from Adapter-Hub

    Args:
        specifier (str): A string specifying the adapter to be loaded.
        adapter_type (AdapterType): The adapter type.
        model_name (str): The identifier of the pre-trained model for which to load an adapter.
        adapter_config (Union[dict, str], optional): The configuration of the adapter to be loaded.
        version (str, optional): The version of the adapter to be loaded. Defaults to None.
        strict (bool, optional): If set to True, only allow adapters exactly matching the given config to be loaded. Defaults to False.

    Returns:
        str: The local path to which the adapter has been downloaded.
    """
    if not adapter_type:
        raise ValueError('Adapter type must be specified.')
    elif not model_name:
        raise ValueError('Unable to resolve adapter without the name of a model. Please specify model_name.')
    if adapter_config:
        adapter_config = resolve_adapter_config(adapter_config)
    hub_entry_url = find_in_index(specifier, adapter_type, model_name, adapter_config=adapter_config, strict=strict)
    if not hub_entry_url:
        raise EnvironmentError("No adapter with name '{}' was found in the adapter index.".format(specifier))
    hub_entry = http_get_json(hub_entry_url)
    if not version:
        version = hub_entry['default_version']
    elif version not in hub_entry['files']:
        logger.warn("Version '{}' of adapter '{}' not found. Falling back to default.".format(version, specifier))
        version = hub_entry['default_version']
    file_entry = hub_entry['files'][version]
    logger.info('Resolved adapter files at {}.'.format(file_entry['url']))
    checksum_algo, checksum = get_checksum(file_entry)
    download_path = download_cached(file_entry['url'], checksum=checksum, checksum_algo=checksum_algo, **kwargs)
    if not download_path:
        raise EnvironmentError('Unable to load file from {}. The file might be unavailable.'.format(file_entry['url']))
    return download_path


def resolve_adapter_path(adapter_name_or_path, adapter_type: AdapterType=AdapterType.text_task, model_name: str=None, adapter_config: Union[dict, str]=None, version: str=None, **kwargs) ->str:
    """Resolves the path to a pre-trained adapter module.
    Note: If attempting to resolve an adapter from the Hub, adapter_config, adapter_type and model_name must be present.

    Args:
        adapter_name_or_path (str): Can be either:
            - the path to a folder in the file system containing the adapter configuration and weights
            - an url pointing to a zip folder containing the adapter configuration and weights
            - a specifier matching a pre-trained adapter uploaded to Adapter-Hub
        adapter_type (AdapterType, optional): The adapter type.
        model_name (str, optional): The identifier of the pre-trained model for which to load an adapter.
        adapter_config (Union[dict, str], optional): The configuration of the adapter to be loaded.
        version (str, optional): The version of the adapter to be loaded. Defaults to None.

    Returns:
        str: The local path from where the adapter module can be loaded.
    """
    if is_remote_url(adapter_name_or_path):
        resolved_folder = download_cached(adapter_name_or_path, **kwargs)
        if not resolved_folder:
            raise EnvironmentError('Unable to load file from {}. The file might be unavailable.'.format(resolved_folder))
        return resolved_folder
    elif isdir(adapter_name_or_path):
        if isfile(join(adapter_name_or_path, WEIGHTS_NAME)) and isfile(join(adapter_name_or_path, CONFIG_NAME)):
            return adapter_name_or_path
        else:
            raise EnvironmentError('No file {} or no file {} found in directory {}'.format(WEIGHTS_NAME, CONFIG_NAME, adapter_name_or_path))
    elif re.fullmatch(ADAPTER_IDENTIFIER_PATTERN, adapter_name_or_path):
        if not adapter_type:
            adapter_type = AdapterType.text_task
        return pull_from_hub(adapter_name_or_path, adapter_type, model_name, adapter_config=adapter_config, version=version, **kwargs)
    else:
        raise ValueError('Unable to identify {} as a valid module location.'.format(adapter_name_or_path))


class AdapterLoader(WeightsLoader):
    """
    A class providing methods for saving and loading adapter modules from the Hub, the filesystem or a remote url.

    Model classes passed to this loader must implement the `ModelAdaptersMixin` class.
    """

    def __init__(self, model, adapter_type=None):
        super().__init__(model, WEIGHTS_NAME, CONFIG_NAME)
        self.adapter_type = adapter_type

    @property
    def config(self):
        return self.model.config.adapters.get_config(self.adapter_type)

    def filter_func(self, adapter_name):
        if self.adapter_type == AdapterType.text_lang:
            return lambda x: '{}_adapters.{}'.format(self.adapter_type, adapter_name) in x or 'invertible_lang_adapters.{}'.format(adapter_name) in x
        elif AdapterType.has(self.adapter_type):
            return lambda x: '{}_adapters.{}'.format(self.adapter_type, adapter_name) in x
        else:
            raise ValueError('Invalid adapter type {}'.format(self.adapter_type))

    def rename_func(self, old_name, new_name):
        return lambda k: k.replace('_adapters.{}'.format(old_name), '_adapters.{}'.format(new_name))

    def save(self, save_directory, name, meta_dict=None):
        """Saves an adapter and its configuration file to a directory, so that it can be reloaded
        using the `load()` method.

        Args:
            save_directory (str): a path to a directory where the adapter will be saved
            task_name (str): the name of the adapter to be saved
        """
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), 'Saving path should be a directory where adapter and configuration can be saved.'
        assert name in self.model.config.adapters.adapters, 'No adapter of this type with the given name is part of this model.'
        adapter_config, adapter_type = self.model.config.adapters.get(name, return_type=True)
        if self.adapter_type:
            assert adapter_type == self.adapter_type, 'Saved adapter has to be a {} adapter.'.format(self.adapter_type)
        else:
            self.adapter_type = adapter_type
        config_dict = build_full_config(adapter_config, self.model.config, type=adapter_type, model_name=self.model.model_name, name=name, model_class=self.model.__class__.__name__)
        self.weights_helper.save_weights_config(save_directory, config_dict, meta_dict=meta_dict)
        filter_func = self.filter_func(config_dict['name'])
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(self, adapter_name_or_path, config=None, version=None, model_name=None, load_as=None, loading_info=None, **kwargs):
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (str, optional): The requested configuration of the adapter.
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
             saved will be used.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
                             and the name of the loaded weights.
        """
        config = config or self.config
        requested_config = AdapterConfig.load(config) if config else None
        model_name = self.model.model_name or model_name
        resolved_folder = resolve_adapter_path(adapter_name_or_path, self.adapter_type, model_name, adapter_config=requested_config, version=version, **kwargs)
        config = self.weights_helper.load_weights_config(resolved_folder)
        if self.adapter_type:
            assert config['type'] == self.adapter_type, 'Loaded adapter has to be a {} adapter.'.format(self.adapter_type)
        else:
            self.adapter_type = config['type']
        adapter_name = load_as or config['name']
        if adapter_name not in self.model.config.adapters.adapters:
            self.model.add_adapter(adapter_name, config['type'], config=config['config'])
        else:
            logger.warning("Overwriting existing adapter '{}'.".format(adapter_name))
        filter_func = self.filter_func(adapter_name)
        rename_func = self.rename_func(config['name'], adapter_name)
        self.weights_helper.load_weights(resolved_folder, filter_func, rename_func=rename_func, loading_info=loading_info, in_base_model=True)
        return resolved_folder, adapter_name


DEFAULT_ADAPTERFUSION_CONFIG = 'dynamic'


class ModelAdaptersMixin(ABC):
    """Mixin for transformer models adding support for loading/ saving adapters."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model_name = None
        self._active_adapter_names = None

    @abstractmethod
    def add_adapter(self, adapter_name: str, adapter_type: AdapterType, config=None):
        """Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            adapter_type (AdapterType): The adapter type.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:
                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        pass

    @abstractmethod
    def train_adapter(self, adapter_names: list):
        """Sets the model into mode for training the given adapters.
        """
        pass

    @abstractmethod
    def train_fusion(self, adapter_names: list):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names.
        """
        pass

    def has_adapters(self, adapter_type=None):
        if not adapter_type:
            return len(self.config.adapters.adapters) > 0
        else:
            return len(self.config.adapters.adapter_list(adapter_type)) > 0

    @property
    def active_adapters(self):
        return self.base_model._active_adapter_names

    def set_active_adapters(self, adapter_names: list):
        """Sets the adapter modules to be used by default in every forward pass.
        This setting can be overriden by passing the `adapter_names` parameter in the `foward()` pass.
        If no adapter with the given name is found, no module of the respective type will be activated.

        Args:
            adapter_names (list): The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        adapter_names = parse_adapter_names(adapter_names)
        new_adapter_names = []
        for stack in adapter_names:
            new_adapter_names.append([])
            for adapter_name in stack:
                if adapter_name in self.config.adapters.adapters:
                    new_adapter_names[-1].append(adapter_name)
                else:
                    logger.info("No adapter with name '{}' available. Skipping.".format(adapter_name))
        if len(new_adapter_names[0]) == 0:
            new_adapter_names = None
        self.base_model._active_adapter_names = new_adapter_names

    def set_adapter_config(self, adapter_type: AdapterType, adapter_config):
        """Sets the adapter configuration of the specified adapter type.

        Args:
            adapter_type (AdapterType): The adapter type.
            adapter_config (str or dict): adapter configuration, can be either:
                - a string identifying a pre-defined adapter configuration
                - a dictionary representing the adapter configuration
                - the path to a file containing the adapter configuration
        """
        if AdapterType.has(adapter_type):
            self.config.adapters.set_config(adapter_type, adapter_config)
        else:
            raise ValueError('Invalid adapter type {}'.format(adapter_type))

    def set_adapter_fusion_config(self, adapter_fusion_config, override_kwargs=None):
        """Sets the adapter fusion configuration.

        Args:
            adapter_fusion_config (str or dict): adapter fusion configuration, can be either:
                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
        """
        if override_kwargs is None:
            override_kwargs = {}
        if isinstance(adapter_fusion_config, str) and adapter_fusion_config in ADAPTERFUSION_CONFIG_MAP:
            self.config.adapter_fusion = AdapterFusionConfig.load(adapter_fusion_config, **override_kwargs)
        elif isinstance(adapter_fusion_config, Mapping):
            self.config.adapter_fusion = adapter_fusion_config
        else:
            raise ValueError('Invalid adapter type {}'.format(adapter_fusion_config))

    def add_fusion(self, adapter_names, adapter_fusion_config=None, override_kwargs=None):
        """Adds AdapterFusion to the model with alll the necessary configurations and weight initializations

        Args:
            adapter_names: a list of adapter names which should be fused
            adapter_fusion_config (str or dict): adapter fusion configuration, can be either:
                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
            override_kwargs: dictionary items for values which should be overwritten in the default AdapterFusion configuration
        """
        if not hasattr(self.config, 'adapter_fusion'):
            if override_kwargs is None:
                override_kwargs = {}
            if adapter_fusion_config is not None:
                self.set_adapter_fusion_config(adapter_fusion_config, **override_kwargs)
            else:
                self.set_adapter_fusion_config(DEFAULT_ADAPTERFUSION_CONFIG)
        elif hasattr(self.config, 'adapter_fusion') and adapter_fusion_config is not None:
            raise Warning('An AdapterFusion config has already been set and will NOT be overwritten')
        if not hasattr(self.config, 'adapter_fusion_models'):
            self.config.adapter_fusion_models = []
        if isinstance(adapter_names, list):
            adapter_fusion_name = ','.join(adapter_names)
        else:
            adapter_fusion_name = adapter_names
        if adapter_fusion_name not in self.config.adapter_fusion_models:
            self.config.adapter_fusion_models.append(adapter_fusion_name)
            self.base_model.add_fusion_layer(adapter_names)

    def save_adapter(self, save_directory: str, adapter_name: str, meta_dict: dict=None, custom_weights_loaders: Optional[List[WeightsLoader]]=None):
        """Saves an adapter and its configuration file to a directory so that it can be shared
        or reloaded using `load_adapter()`.

        Args:
            save_directory (str): Path to a directory where the adapter should be saved.
            adapter_name (str): Name of the adapter to be saved.

        Raises:
            ValueError: If the given adapter name is invalid.
        """
        adapter_type = self.config.adapters.get_type(adapter_name)
        if adapter_type:
            loader = AdapterLoader(self, adapter_type)
            loader.save(save_directory, adapter_name, meta_dict)
            if custom_weights_loaders:
                for weights_loader in custom_weights_loaders:
                    weights_loader.save(save_directory, adapter_name)
        else:
            raise ValueError("Could not resolve '{}' to a valid adapter name.".format(adapter_name))

    def save_adapter_fusion(self, save_directory: str, adapter_names: list, custom_weights_loaders: Optional[List[WeightsLoader]]=None):
        """Saves an adapter and its configuration file to a directory so that it can be shared
        or reloaded using `load_adapter()`.

        Args:
            save_directory (str): Path to a directory where the adapter should be saved.
            adapter_name (str): Name of the adapter to be saved.

        Raises:
            ValueError: If the given adapter name is invalid.
        """
        loader = AdapterFusionLoader(self)
        loader.save(save_directory, adapter_names)
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_names)

    def load_adapter(self, adapter_name_or_path: str, adapter_type: AdapterType=None, config: Union[dict, str]=None, version: str=None, model_name: str=None, load_as: str=None, custom_weights_loaders: Optional[List[WeightsLoader]]=None, **kwargs) ->str:
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            adapter_type (AdapterType, optional): The type of adapter to be loaded. If not specified, text_task will be
                    used for adapters loaded from the Hub.
            config (dict or str, optional): The requested configuration of the adapter.
                If not specified, will be either:
                - the default adapter config for the requested adapter if specified
                - the global default adapter config
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                    saved will be used.

        Returns:
            str: The name with which the adapter was added to the model.
        """
        if AdapterType.has(adapter_type) or not adapter_type:
            loader = AdapterLoader(self, adapter_type)
            load_dir, load_name = loader.load(adapter_name_or_path, config, version, model_name, load_as, **kwargs)
            if custom_weights_loaders:
                for weights_loader in custom_weights_loaders:
                    weights_loader.load(load_dir, load_as=load_as, loading_info=kwargs.get('loading_info', None))
            return load_name
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def load_adapter_fusion(self, adapter_fusion_name_or_path: str, load_as: str=None, custom_weights_loaders: Optional[List[WeightsLoader]]=None, **kwargs) ->str:
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_fusion_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter fusion module to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (dict or str, optional): The requested configuration of the adapter fusion.
                If not specified, will be either:
                - the default adapter config for the requested adapter fusion if specified
                - the global default adapter fusion config
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                    saved will be used.

        Returns:
            str: The name with which the adapter was added to the model.
        """
        loader = AdapterFusionLoader(self)
        load_dir, load_name = loader.load(adapter_fusion_name_or_path, load_as)
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(load_dir, load_as=load_as, loading_info=kwargs.get('loading_info', None))
        return load_name

    def save_all_adapters(self, save_directory: str, meta_dict: dict=None, custom_weights_loaders: Optional[List[WeightsLoader]]=None):
        """Saves all adapters of this model together with their configuration
        to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
        """
        for name in self.config.adapters.adapters:
            adapter_config, adapter_type = self.config.adapters.get(name, return_type=True)
            h = get_adapter_config_hash(adapter_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({'config_id': h})
            else:
                meta_dict = {'config_id': h}
            self.save_adapter(save_path, name, meta_dict=meta_dict, custom_weights_loaders=custom_weights_loaders)

    def save_all_adapter_fusions(self, save_directory: str, meta_dict: dict=None, custom_weights_loaders: Optional[List[WeightsLoader]]=None):
        """Saves all adapters of this model together with their configuration
        to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
        """
        for name in self.config.adapter_fusion_models:
            adapter_fusion_config = self.config.adapter_fusion
            h = get_adapter_config_hash(adapter_fusion_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({'config_id': h})
            else:
                meta_dict = {'config_id': h}
            self.save_adapter_fusion(save_path, name, custom_weights_loaders=custom_weights_loaders)

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model.
        """
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        self.model_freezed = freeze


def flatten_adapter_names(adapter_names: list) ->List[str]:
    if not isinstance(adapter_names, list):
        return [adapter_names]
    elif not isinstance(adapter_names[0], list):
        return adapter_names
    else:
        return list(itertools.chain(*adapter_names))


class BertModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the BertModel module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        self.invertible_lang_adapters = nn.ModuleDict(dict())
        for language in self.config.adapters.adapter_list(AdapterType.text_lang):
            self.encoder.add_adapter(language, AdapterType.text_lang)
            self.add_invertible_lang_adapter(language)
        for task in self.config.adapters.adapter_list(AdapterType.text_task):
            self.encoder.add_adapter(task, AdapterType.text_task)
        if hasattr(self.config, 'fusion_models'):
            for fusion_adapter_names in self.config.fusion_models:
                self.add_fusion_layer(fusion_adapter_names)

    def train_adapter(self, adapter_names: list):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_names_flat = flatten_adapter_names(adapter_names)
        self.encoder.enable_adapters(adapter_names_flat, True, False)
        for adapter_name in adapter_names_flat:
            if adapter_name in self.invertible_lang_adapters:
                for param in self.invertible_lang_adapters[adapter_name].parameters():
                    param.requires_grad = True
        self.set_active_adapters(adapter_names)

    def train_fusion(self, adapter_names: list):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_names_flat = flatten_adapter_names(adapter_names)
        self.encoder.enable_adapters(adapter_names_flat, False, True)
        self.set_active_adapters(adapter_names)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType, config=None):
        """Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            adapter_type (AdapterType): The adapter type.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:
                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        if not AdapterType.has(adapter_type):
            raise ValueError('Invalid adapter type {}'.format(adapter_type))
        if not self.config.adapters.get_config(adapter_type):
            self.config.adapters.set_config(adapter_type, config or DEFAULT_ADAPTER_CONFIG)
        self.config.adapters.add(adapter_name, adapter_type, config=config)
        self.encoder.add_adapter(adapter_name, adapter_type)
        if adapter_type == AdapterType.text_lang:
            self.add_invertible_lang_adapter(adapter_name)

    def add_invertible_lang_adapter(self, language):
        if language in self.invertible_lang_adapters:
            raise ValueError(f"Model already contains an adapter module for '{language}'.")
        inv_adap_config = self.config.adapters.get(language)['invertible_adapter']
        if inv_adap_config['block_type'] == 'nice':
            inv_adap = NICECouplingBlock([[self.config.hidden_size]], non_linearity=inv_adap_config['non_linearity'], reduction_factor=inv_adap_config['reduction_factor'])
        elif inv_adap_config['block_type'] == 'glow':
            inv_adap = GLOWCouplingBlock([[self.config.hidden_size]], non_linearity=inv_adap_config['non_linearity'], reduction_factor=inv_adap_config['reduction_factor'])
        else:
            raise ValueError(f"Invalid invertible adapter type '{inv_adap_config['block_type']}'.")
        self.invertible_lang_adapters[language] = inv_adap
        self.invertible_lang_adapters[language].apply(Adapter.init_bert_weights)

    def get_invertible_lang_adapter(self, language):
        if language in self.invertible_lang_adapters:
            return self.invertible_lang_adapters[language]
        else:
            return None

    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_attention_layer"""
        self.encoder.add_fusion_layer(adapter_names)


class BertConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.BertModel`.
        It is used to instantiate an BERT model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.


        Args:
            vocab_size (:obj:`int`, optional, defaults to 30522):
                Vocabulary size of the BERT model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.BertModel`.
            hidden_size (:obj:`int`, optional, defaults to 768):
                Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (:obj:`int`, optional, defaults to 3072):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size (:obj:`int`, optional, defaults to 2):
                The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.

        Example::

            from transformers import BertModel, BertConfig

            # Initializing a BERT bert-base-uncased style configuration
            configuration = BertConfig()

            # Initializing a model from the bert-base-uncased style configuration
            model = BertModel(configuration)

            # Accessing the model configuration
            configuration = model.config
    """
    model_type = 'bert'

    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        adapter_config_dict = kwargs.pop('adapters', None)
        if adapter_config_dict:
            self.adapters = ModelAdaptersConfig(**adapter_config_dict)
        else:
            self.adapters = ModelAdaptersConfig()


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer', 'AdamWeightDecayOptimizer_1', 'global_step'] for n in name):
            logger.info('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info('Skipping {}'.format('/'.join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        logger.info('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model


ROBERTA_START_DOCSTRING = """

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


class RobertaConfig(BertConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel`.
        It is used to instantiate an RoBERTa model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

        The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`.
        It reuses the same defaults. Please check the parent class for more information.

        Example::

            from transformers import RobertaConfig, RobertaModel

            # Initializing a RoBERTa configuration
            configuration = RobertaConfig()

            # Initializing a model from the configuration
            model = RobertaModel(configuration)

            # Accessing the model configuration
            configuration = model.config
    """
    model_type = 'roberta'

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig.
        """
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class XLMRobertaConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    model_type = 'xlm-roberta'


XLM_ROBERTA_START_DOCSTRING = """

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.XLMRobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


def word_lens_to_idxs_fast(token_lens):
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([(i + offset) for i in range(token_len)] + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


class Base_Model(nn.Module):

    def __init__(self, config, task_name):
        super().__init__()
        self.config = config
        self.task_name = task_name
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.xlmr = XLMRobertaModel.from_pretrained(config.embedding_name, cache_dir=os.path.join(config._cache_dir, config.embedding_name), output_hidden_states=True)
        self.xlmr_dropout = nn.Dropout(p=config.embedding_dropout)
        task_config = AdapterConfig.load('pfeiffer', reduction_factor=6 if config.embedding_name == 'xlm-roberta-base' else 4)
        self.xlmr.add_adapter(task_name, AdapterType.text_task, config=task_config)
        self.xlmr.train_adapter([task_name])
        self.xlmr.set_active_adapters([task_name])

    def encode(self, piece_idxs, attention_masks):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]
        wordpiece_reprs = xlmr_outputs[:, 1:-1, :]
        wordpiece_reprs = self.xlmr_dropout(wordpiece_reprs)
        return wordpiece_reprs

    def encode_words(self, piece_idxs, attention_masks, word_lens):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]
        cls_reprs = xlmr_outputs[:, 0, :].unsqueeze(1)
        idxs, masks, token_num, token_len = word_lens_to_idxs_fast(word_lens)
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.xlmr_dim) + 1
        masks = xlmr_outputs.new(masks).unsqueeze(-1)
        xlmr_outputs = torch.gather(xlmr_outputs, 1, idxs) * masks
        xlmr_outputs = xlmr_outputs.view(batch_size, token_num, token_len, self.xlmr_dim)
        xlmr_outputs = xlmr_outputs.sum(2)
        return xlmr_outputs, cls_reprs

    def forward(self, batch):
        raise NotImplementedError


class Multilingual_Embedding(Base_Model):

    def __init__(self, config, model_name='embedding'):
        super(Multilingual_Embedding, self).__init__(config, task_name=model_name)

    def get_tokenizer_inputs(self, batch):
        wordpiece_reprs = self.encode(piece_idxs=batch.piece_idxs, attention_masks=batch.attention_masks)
        return wordpiece_reprs

    def get_tagger_inputs(self, batch):
        word_reprs, cls_reprs = self.encode_words(piece_idxs=batch.piece_idxs, attention_masks=batch.attention_masks, word_lens=batch.word_lens)
        return word_reprs, cls_reprs


class Deep_Biaffine(nn.Module):
    """
    implemented based on the paper https://arxiv.org/abs/1611.01734
    """

    def __init__(self, in_dim1, in_dim2, hidden_dim, output_dim):
        super().__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ffn1 = nn.Sequential(nn.Linear(in_dim1, hidden_dim), nn.ReLU(), nn.Dropout(0.5))
        self.ffn2 = nn.Sequential(nn.Linear(in_dim2, hidden_dim), nn.ReLU(), nn.Dropout(0.5))
        self.pairwise_weight = nn.Parameter(torch.Tensor(in_dim1 + 1, in_dim2 + 1, output_dim))
        self.pairwise_weight.data.zero_()

    def forward(self, x1, x2):
        h1 = self.ffn1(x1)
        h2 = self.ffn2(x2)
        g1 = torch.cat([h1, h1.new_ones(*h1.size()[:-1], 1)], len(h1.size()) - 1)
        g2 = torch.cat([h2, h2.new_ones(*h2.size()[:-1], 1)], len(h2.size()) - 1)
        g1_size = g1.size()
        g2_size = g2.size()
        g1_w = torch.mm(g1.view(-1, g1_size[-1]), self.pairwise_weight.view(-1, (self.in_dim2 + 1) * self.output_dim))
        g2 = g2.transpose(1, 2)
        g1_w_g2 = g1_w.view(g1_size[0], g1_size[1] * self.output_dim, g2_size[2]).bmm(g2)
        g1_w_g2 = g1_w_g2.view(g1_size[0], g1_size[1], self.output_dim, g2_size[1]).transpose(2, 3)
        return g1_w_g2


class Linears(nn.Module):

    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias) for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


def viterbi_decode(scores, transition_params):
    """
    Decode a tag sequence with viterbi algorithm.
    scores: seq_len x num_tags (numpy array)
    transition_params: num_tags x num_tags (numpy array)
    @return:
        viterbi: a list of tag ids with highest score
        viterbi_score: the highest score
    """
    trellis = np.zeros_like(scores)
    backpointers = np.zeros_like(scores, dtype=np.int32)
    trellis[0] = scores[0]
    for t in range(1, scores.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = scores[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)
    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


class NERClassifier(nn.Module):

    def __init__(self, config, language):
        super().__init__()
        self.config = config
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.entity_label_stoi = config.ner_vocabs[language]
        self.entity_label_itos = {i: s for s, i in self.entity_label_stoi.items()}
        self.entity_label_num = len(self.entity_label_stoi)
        self.entity_label_ffn = Linears([self.xlmr_dim, config.hidden_num, self.entity_label_num], dropout_prob=config.linear_dropout, bias=config.linear_bias, activation=config.linear_activation)
        self.crit = CRFLoss(self.entity_label_num)
        if not config.training:
            self.initialized_weights = self.state_dict()
            self.pretrained_ner_weights = torch.load(os.path.join(self.config._cache_dir, self.config.embedding_name, language, '{}.ner.mdl'.format(language)), map_location=self.config.device)['adapters']
            for name, value in self.pretrained_ner_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value
            self.load_state_dict(self.initialized_weights)
            None

    def forward(self, batch, word_reprs):
        batch_size, _, _ = word_reprs.size()
        logits = self.entity_label_ffn(word_reprs)
        loss, trans = self.crit(logits, batch.word_mask, batch.entity_label_idxs)
        return loss

    def predict(self, batch, word_reprs):
        batch_size, _, _ = word_reprs.size()
        logits = self.entity_label_ffn(word_reprs)
        _, trans = self.crit(logits, batch.word_mask, batch.entity_label_idxs)
        trans = trans.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :batch.word_num[i]], trans)
            tags = [self.entity_label_itos[t] for t in tags]
            tag_seqs += [tags]
        return tag_seqs


DEPREL = 'deprel'


FEATS = 'feats'


UPOS = 'upos'


XPOS = 'xpos'


lang2treebank = {'afrikaans': 'UD_Afrikaans-AfriBooms', 'ancient-greek-perseus': 'UD_Ancient_Greek-Perseus', 'ancient-greek': 'UD_Ancient_Greek-PROIEL', 'arabic': 'UD_Arabic-PADT', 'armenian': 'UD_Armenian-ArmTDP', 'basque': 'UD_Basque-BDT', 'belarusian': 'UD_Belarusian-HSE', 'bulgarian': 'UD_Bulgarian-BTB', 'catalan': 'UD_Catalan-AnCora', 'chinese': 'UD_Simplified_Chinese-GSDSimp', 'traditional-chinese': 'UD_Chinese-GSD', 'classical-chinese': 'UD_Classical_Chinese-Kyoto', 'croatian': 'UD_Croatian-SET', 'czech-cac': 'UD_Czech-CAC', 'czech-cltt': 'UD_Czech-CLTT', 'czech-fictree': 'UD_Czech-FicTree', 'czech': 'UD_Czech-PDT', 'danish': 'UD_Danish-DDT', 'dutch': 'UD_Dutch-Alpino', 'dutch-lassysmall': 'UD_Dutch-LassySmall', 'english': 'UD_English-EWT', 'english-gum': 'UD_English-GUM', 'english-lines': 'UD_English-LinES', 'english-partut': 'UD_English-ParTUT', 'estonian': 'UD_Estonian-EDT', 'estonian-ewt': 'UD_Estonian-EWT', 'finnish-ftb': 'UD_Finnish-FTB', 'finnish': 'UD_Finnish-TDT', 'french': 'UD_French-GSD', 'french-partut': 'UD_French-ParTUT', 'french-sequoia': 'UD_French-Sequoia', 'french-spoken': 'UD_French-Spoken', 'galician': 'UD_Galician-CTG', 'galician-treegal': 'UD_Galician-TreeGal', 'german': 'UD_German-GSD', 'german-hdt': 'UD_German-HDT', 'greek': 'UD_Greek-GDT', 'hebrew': 'UD_Hebrew-HTB', 'hindi': 'UD_Hindi-HDTB', 'hungarian': 'UD_Hungarian-Szeged', 'indonesian': 'UD_Indonesian-GSD', 'irish': 'UD_Irish-IDT', 'italian': 'UD_Italian-ISDT', 'italian-partut': 'UD_Italian-ParTUT', 'italian-postwita': 'UD_Italian-PoSTWITA', 'italian-twittiro': 'UD_Italian-TWITTIRO', 'italian-vit': 'UD_Italian-VIT', 'japanese': 'UD_Japanese-GSD', 'kazakh': 'UD_Kazakh-KTB', 'korean': 'UD_Korean-GSD', 'korean-kaist': 'UD_Korean-Kaist', 'kurmanji': 'UD_Kurmanji-MG', 'latin': 'UD_Latin-ITTB', 'latin-perseus': 'UD_Latin-Perseus', 'latin-proiel': 'UD_Latin-PROIEL', 'latvian': 'UD_Latvian-LVTB', 'lithuanian': 'UD_Lithuanian-ALKSNIS', 'lithuanian-hse': 'UD_Lithuanian-HSE', 'marathi': 'UD_Marathi-UFAL', 'norwegian-nynorsk': 'UD_Norwegian_Nynorsk-Nynorsk', 'norwegian-nynorsklia': 'UD_Norwegian_Nynorsk-NynorskLIA', 'norwegian-bokmaal': 'UD_Norwegian-Bokmaal', 'old-french': 'UD_Old_French-SRCMF', 'old-russian': 'UD_Old_Russian-TOROT', 'persian': 'UD_Persian-Seraji', 'polish-lfg': 'UD_Polish-LFG', 'polish': 'UD_Polish-PDB', 'portuguese': 'UD_Portuguese-Bosque', 'portuguese-gsd': 'UD_Portuguese-GSD', 'romanian-nonstandard': 'UD_Romanian-Nonstandard', 'romanian': 'UD_Romanian-RRT', 'russian-gsd': 'UD_Russian-GSD', 'russian': 'UD_Russian-SynTagRus', 'russian-taiga': 'UD_Russian-Taiga', 'scottish-gaelic': 'UD_Scottish_Gaelic-ARCOSG', 'serbian': 'UD_Serbian-SET', 'slovak': 'UD_Slovak-SNK', 'slovenian': 'UD_Slovenian-SSJ', 'slovenian-sst': 'UD_Slovenian-SST', 'spanish': 'UD_Spanish-AnCora', 'spanish-gsd': 'UD_Spanish-GSD', 'swedish-lines': 'UD_Swedish-LinES', 'swedish': 'UD_Swedish-Talbanken', 'tamil': 'UD_Tamil-TTB', 'telugu': 'UD_Telugu-MTG', 'turkish': 'UD_Turkish-IMST', 'ukrainian': 'UD_Ukrainian-IU', 'urdu': 'UD_Urdu-UDTB', 'uyghur': 'UD_Uyghur-UDT', 'vietnamese': 'UD_Vietnamese-VLSP', 'vietnamese-vtb': 'UD_Vietnamese-VTB', 'customized': 'UD_Customized', 'customized-mwt': 'UD_Customized-MWT', 'customized-ner': 'UD_Customized-NER', 'customized-mwt-ner': 'UD_Customized-MWT-NER'}


treebank2lang = {v: k for k, v in lang2treebank.items()}


class PosDepClassifier(nn.Module):

    def __init__(self, config, treebank_name):
        super().__init__()
        self.config = config
        self.vocabs = config.vocabs[treebank_name]
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.upos_embedding = nn.Embedding(num_embeddings=len(self.vocabs[UPOS]), embedding_dim=50)
        self.upos_ffn = nn.Linear(self.xlmr_dim, len(self.vocabs[UPOS]))
        self.xpos_ffn = nn.Linear(self.xlmr_dim + 50, len(self.vocabs[XPOS]))
        self.feats_ffn = nn.Linear(self.xlmr_dim, len(self.vocabs[FEATS]))
        self.down_dim = self.xlmr_dim // 4
        self.down_project = nn.Linear(self.xlmr_dim, self.down_dim)
        self.unlabeled = Deep_Biaffine(self.down_dim, self.down_dim, self.down_dim, 1)
        self.deprel = Deep_Biaffine(self.down_dim, self.down_dim, self.down_dim, len(self.vocabs[DEPREL]))
        self.criteria = torch.nn.CrossEntropyLoss()
        if not config.training:
            self.initialized_weights = self.state_dict()
            language = treebank2lang[treebank_name]
            self.pretrained_tagger_weights = torch.load(os.path.join(self.config._cache_dir, self.config.embedding_name, language, '{}.tagger.mdl'.format(language)), map_location=self.config.device)['adapters']
            for name, value in self.pretrained_tagger_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value
            self.load_state_dict(self.initialized_weights)
            None

    def forward(self, batch, word_reprs, cls_reprs):
        upos_scores = self.upos_ffn(word_reprs)
        upos_scores = upos_scores.view(-1, len(self.vocabs[UPOS]))
        xpos_reprs = torch.cat([word_reprs, self.upos_embedding(batch.upos_ids)], dim=2)
        xpos_scores = self.xpos_ffn(xpos_reprs)
        xpos_scores = xpos_scores.view(-1, len(self.vocabs[XPOS]))
        feats_scores = self.feats_ffn(word_reprs)
        feats_scores = feats_scores.view(-1, len(self.vocabs[FEATS]))
        loss = self.criteria(upos_scores, batch.upos_type_idxs) + self.criteria(xpos_scores, batch.xpos_type_idxs) + self.criteria(feats_scores, batch.feats_type_idxs)
        dep_reprs = torch.cat([cls_reprs, word_reprs], dim=1)
        dep_reprs = self.down_project(dep_reprs)
        unlabeled_scores = self.unlabeled(dep_reprs, dep_reprs).squeeze(3)
        diag = torch.eye(batch.head_idxs.size(-1) + 1, dtype=torch.bool).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))
        unlabeled_scores = unlabeled_scores[:, 1:, :]
        unlabeled_scores = unlabeled_scores.masked_fill(batch.word_mask.unsqueeze(1), -float('inf'))
        unlabeled_target = batch.head_idxs.masked_fill(batch.word_mask[:, 1:], -100)
        loss += self.criteria(unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)), unlabeled_target.view(-1))
        deprel_scores = self.deprel(dep_reprs, dep_reprs)
        deprel_scores = deprel_scores[:, 1:]
        deprel_scores = torch.gather(deprel_scores, 2, batch.head_idxs.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, len(self.vocabs[DEPREL]))).view(-1, len(self.vocabs[DEPREL]))
        deprel_target = batch.deprel_idxs.masked_fill(batch.word_mask[:, 1:], -100)
        loss += self.criteria(deprel_scores.contiguous(), deprel_target.view(-1))
        return loss

    def predict(self, batch, word_reprs, cls_reprs):
        upos_scores = self.upos_ffn(word_reprs)
        predicted_upos = torch.argmax(upos_scores, dim=2)
        xpos_reprs = torch.cat([word_reprs, self.upos_embedding(predicted_upos)], dim=2)
        xpos_scores = self.xpos_ffn(xpos_reprs)
        predicted_xpos = torch.argmax(xpos_scores, dim=2)
        feats_scores = self.feats_ffn(word_reprs)
        predicted_feats = torch.argmax(feats_scores, dim=2)
        dep_reprs = torch.cat([cls_reprs, word_reprs], dim=1)
        dep_reprs = self.down_project(dep_reprs)
        unlabeled_scores = self.unlabeled(dep_reprs, dep_reprs).squeeze(3)
        diag = torch.eye(batch.head_idxs.size(-1) + 1, dtype=torch.bool).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))
        deprel_scores = self.deprel(dep_reprs, dep_reprs)
        dep_preds = []
        dep_preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
        dep_preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())
        return predicted_upos, predicted_xpos, predicted_feats, dep_preds


class TokenizerClassifier(nn.Module):

    def __init__(self, config, treebank_name):
        super().__init__()
        self.config = config
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.tokenizer_ffn = nn.Linear(self.xlmr_dim, 5)
        self.criteria = torch.nn.CrossEntropyLoss()
        if not config.training:
            language = treebank2lang[treebank_name]
            self.pretrained_tokenizer_weights = torch.load(os.path.join(self.config._cache_dir, self.config.embedding_name, language, '{}.tokenizer.mdl'.format(language)), map_location=self.config.device)['adapters']
            self.initialized_weights = self.state_dict()
            for name, value in self.pretrained_tokenizer_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value
            self.load_state_dict(self.initialized_weights)
            None

    def forward(self, wordpiece_reprs, batch):
        wordpiece_scores = self.tokenizer_ffn(wordpiece_reprs)
        wordpiece_scores = wordpiece_scores.view(-1, 5)
        token_type_idxs = batch.token_type_idxs.view(-1)
        loss = self.criteria(wordpiece_scores, token_type_idxs)
        return loss

    def predict(self, batch, wordpiece_reprs):
        wordpiece_scores = self.tokenizer_ffn(wordpiece_reprs)
        predicted_wordpiece_labels = torch.argmax(wordpiece_scores, dim=2)
        return predicted_wordpiece_labels, batch.wordpiece_ends, batch.paragraph_index


def SequenceLoss(vocab_size):
    weight = torch.ones(vocab_size)
    weight[PAD_ID] = 0
    crit = nn.NLLLoss(weight)
    return crit


class MixLoss(nn.Module):
    """
    A mixture of SequenceLoss and CrossEntropyLoss.
    Loss = SequenceLoss + alpha * CELoss
    """

    def __init__(self, vocab_size, alpha):
        super().__init__()
        self.seq_loss = SequenceLoss(vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()
        assert alpha >= 0
        self.alpha = alpha

    def forward(self, seq_inputs, seq_targets, class_inputs, class_targets):
        sl = self.seq_loss(seq_inputs, seq_targets)
        cel = self.ce_loss(class_inputs, class_targets)
        loss = sl + self.alpha * cel
        return loss


class MaxEntropySequenceLoss(nn.Module):
    """
    A max entropy loss that encourage the model to have large entropy,
    therefore giving more diverse outputs.

    Loss = NLLLoss + alpha * EntropyLoss
    """

    def __init__(self, vocab_size, alpha):
        super().__init__()
        weight = torch.ones(vocab_size)
        weight[PAD_ID] = 0
        self.nll = nn.NLLLoss(weight)
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        inputs: [N, C]
        targets: [N]
        """
        assert inputs.size(0) == targets.size(0)
        nll_loss = self.nll(inputs, targets)
        mask = targets.eq(PAD_ID).unsqueeze(1).expand_as(inputs)
        masked_inputs = inputs.clone().masked_fill_(mask, 0.0)
        p = torch.exp(masked_inputs)
        ent_loss = p.mul(masked_inputs).sum() / inputs.size(0)
        loss = nll_loss + self.alpha * ent_loss
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Adapter,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (AlbertAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, output_attentions=4, attention_probs_dropout_prob=0.5, layer_norm_eps=1)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (AlbertSOPHead,
     lambda: ([], {'config': _mock_config(classifier_dropout_prob=0.5, hidden_size=4, num_labels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Attention,
     lambda: ([], {'nx': 4, 'n_ctx': 4, 'config': _mock_config(n_head=4, output_attentions=4, attn_pdrop=0.5, resid_pdrop=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BartClassificationHead,
     lambda: ([], {'input_dim': 4, 'inner_dim': 4, 'num_classes': 4, 'pooler_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicAttention,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, output_attentions=4, attention_probs_dropout_prob=0.5, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOnlyNSPHead,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BertPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, output_attentions=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1D,
     lambda: ([], {'nf': 4, 'nx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeepAttention,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Deep_Biaffine,
     lambda: ([], {'in_dim1': 4, 'in_dim2': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (FFN,
     lambda: ([], {'config': _mock_config(dropout=0.5, dim=4, hidden_dim=4, activation='relu')}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LSHSelfAttention,
     lambda: ([], {'config': _mock_config(lsh_attn_chunk_length=4, num_hashes=4, num_buckets=4, lsh_num_chunks_before=4, lsh_num_chunks_after=4, hash_seed=4, is_decoder=4, max_position_embeddings=4, lsh_attention_probs_dropout_prob=0.5, num_attention_heads=4, attention_head_size=4, hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LearnedPositionalEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4, 'padding_idx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearAttention,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Linears,
     lambda: ([], {'dimensions': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LocalSelfAttention,
     lambda: ([], {'config': _mock_config(num_attention_heads=4, local_attn_chunk_length=4, local_num_chunks_before=4, local_num_chunks_after=4, is_decoder=4, pad_token_id=4, attention_head_size=4, hidden_size=4, local_attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LongformerClassificationHead,
     lambda: ([], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=0.5, num_labels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LongformerSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, output_attentions=4, attention_probs_dropout_prob=0.5, attention_window=[4, 4]), 'layer_id': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'n_heads': 4, 'dim': 4, 'config': _mock_config(output_attentions=4, attention_dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 1, 1, 4])], {}),
     False),
    (MultiHeadSelfAttention,
     lambda: ([], {'config': _mock_config(n_heads=4, dim=4, attention_dropout=0.5, output_attentions=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 1, 1, 4])], {}),
     False),
    (PoolerStartLogits,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionwiseFF,
     lambda: ([], {'d_model': 4, 'd_inner': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReformerFeedForwardOutput,
     lambda: ([], {'config': _mock_config(hidden_dropout_prob=0.5, feed_forward_size=4, hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RobertaClassificationHead,
     lambda: ([], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=0.5, num_labels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SQuADHead,
     lambda: ([], {'config': _mock_config(start_n_top=4, end_n_top=4, hidden_size=4, layer_norm_eps=1)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SelfAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (T5DenseReluDense,
     lambda: ([], {'config': _mock_config(d_model=4, d_ff=4, dropout_rate=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (T5LayerFF,
     lambda: ([], {'config': _mock_config(d_model=4, d_ff=4, dropout_rate=0.5, layer_norm_epsilon=1)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (T5LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (XLNetFeedForward,
     lambda: ([], {'config': _mock_config(d_model=4, layer_norm_eps=1, d_inner=4, dropout=0.5, ff_activation=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_nlp_uoregon_trankit(_paritybench_base):
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

