import sys
_module = sys.modules[__name__]
del sys
combine_docs = _module
run_swag = _module
run_qa = _module
trainer_qa = _module
utils_qa = _module
run_glue = _module
run_ner = _module
run_image_classification = _module
run_swag = _module
utils_qa = _module
run_clm = _module
run_mlm = _module
utils_qa = _module
run_summarization = _module
test_examples = _module
run_translation = _module
bettertransformer = _module
models = _module
base = _module
encoder_models = _module
transformation = _module
commands = _module
env = _module
export = _module
onnx = _module
optimum_cli = _module
configuration_utils = _module
exporters = _module
base = _module
config = _module
convert = _module
model_configs = _module
utils = _module
tasks = _module
fx = _module
optimization = _module
transformations = _module
quantization = _module
functions = _module
modeling_base = _module
configuration = _module
graph_transformations = _module
modeling_seq2seq = _module
onnxruntime = _module
graph = _module
io_binding = _module
io_binding_helper = _module
model = _module
modeling_decoder = _module
modeling_ort = _module
modeling_seq2seq = _module
preprocessors = _module
passes = _module
excluders = _module
fully_connected = _module
gelu = _module
layernorm = _module
runs = _module
calibrator = _module
trainer = _module
trainer_seq2seq = _module
training_args = _module
training_args_seq2seq = _module
utils = _module
pipelines = _module
quantization_base = _module
runs_base = _module
doc = _module
file_utils = _module
import_utils = _module
input_generators = _module
logging = _module
normalized_config = _module
preprocessing = _module
image_classification = _module
question_answering = _module
text_classification = _module
token_classification = _module
save_utils = _module
testing_utils = _module
version = _module
setup = _module
benchmark_bettertransformer = _module
benchmark_bettertransformer_vit = _module
test_transformers_optimum_examples_parity = _module
test_bettertransformer_audio = _module
test_bettertransformer_encoder = _module
test_bettertransformer_vision = _module
test_gpu = _module
testing_bettertransformer_utils = _module
test_cli = _module
exporters_utils = _module
test_exporters_onnx_cli = _module
test_onnx_config_loss = _module
test_onnx_export = _module
test_transformations = _module
test_quantization = _module
test_onnx_export_custom_module = _module
test_onnx_graph_transformations = _module
nightly_test_trainer = _module
test_modeling = _module
test_optimization = _module
test_utils = _module
test_configuration_utils = _module
test_modeling_base = _module
test_dummpy_input_generators = _module

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


from functools import partial


from itertools import chain


from typing import Optional


from typing import Union


import numpy as np


import torch


import collections


from typing import Tuple


from torchvision.transforms import CenterCrop


from torchvision.transforms import Compose


from torchvision.transforms import Normalize


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


import torch.nn as nn


from typing import TYPE_CHECKING


from copy import deepcopy


from typing import Dict


import copy


import enum


import inspect


import itertools


import re


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


from typing import Any


from typing import Callable


from typing import Iterable


from typing import List


from typing import Mapping


from inspect import signature


import random


import functools


import warnings


from typing import Type


from torch.fx.node import Argument


from torch.fx.node import Node


from torch.fx.node import Target


from torch.nn.intrinsic import _FusedModule


from torch.quantization.fx.graph_module import GraphModule


from torch.quantization.fx.graph_module import ObservedGraphModule


from torch.quantization.quantize_fx import Scope


from torch.quantization.quantize_fx import ScopeContextManager


from torch.quantization.quantize_fx import fuse_fx as orig_fuse_fx


from torch.quantization.quantize_fx import prepare_fx as orig_prepare_fx


from torch.quantization.quantize_fx import prepare_qat_fx as orig_prepare_qat_fx


from torch.nn import CrossEntropyLoss


import math


import time


import torch.distributed as dist


from torch import nn


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import RandomSampler


from torch.utils.data.distributed import DistributedSampler


from enum import Enum


from time import perf_counter_ns


from typing import Set


import tensorflow as tf


from torch.ao.quantization.quantize_fx import fuse_fx as orig_fuse_fx


from torch.ao.quantization.quantize_fx import prepare_fx as orig_prepare_fx


from torch.ao.quantization.quantize_fx import prepare_qat_fx as orig_prepare_qat_fx


KNOWN_ACTIVATION_ATTRIBUTES = ['hidden_act', 'activation', 'act_fn', 'activation_function']


KNOWN_NUM_LAYERS = ['num_hidden_layers', 'num_layers', 'encoder_layers', 'n_layers']


KNOWN_POS_EMB_ATTRIBUTES = ['position_embedding_type']


SUPPORTED_ACTIVATION_FUNCTIONS = ['gelu', 'relu', 'gelu_new']


USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS = ['quick_gelu']


class BetterTransformerBaseLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm_first = False
        self.use_gelu = False
        self.act_fn = None
        self.pos_emb_type = None
        self.num_heads = None
        self.embed_dim = None
        self.num_layers = None
        for attr in KNOWN_ACTIVATION_ATTRIBUTES:
            if hasattr(config, attr):
                self.act_fn = getattr(config, attr)
                break
        if self.act_fn is None and hasattr(self, '_get_activation_function'):
            self.act_fn = self._get_activation_function(config)
        for attr in KNOWN_POS_EMB_ATTRIBUTES:
            if hasattr(config, attr):
                self.pos_emb_type = getattr(config, attr)
                break
        for attr in KNOWN_NUM_LAYERS:
            if hasattr(config, attr):
                self.num_layers = getattr(config, attr)
                break

    def validate_bettertransformer(self):
        """
        A wrapper function to validate the `BetterTransformer` implementation. Implements most relevant checks
        that are present in: https://github.com/pytorch/pytorch/blob/0fc7de398636f4b53e6c3fde38b4e48a5ff5b37d/torch/nn/modules/transformer.py#L457-L475
        """
        if self.num_heads is None:
            raise ValueError('Number of heads not set for `BetterTransformer` integration.')
        if self.embed_dim is None:
            raise ValueError('Embedding dimension not set for `BetterTransformer` integration.')
        if self.norm2_eps is None or self.norm1_eps is None:
            raise ValueError('`norm2_eps` and `norm1_eps` not set for `BetterTransformer` integration.')
        if self.pos_emb_type is not None and self.pos_emb_type != 'absolute':
            raise ValueError(f'Positional embedding type {self.pos_emb_type} not supported for `BetterTransformer` integration')
        if self.norm1_eps != self.norm2_eps:
            raise ValueError('norm1_eps and norm2_eps must be equal for `BetterTransformer` integration.')
        if self.act_fn in USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS:
            logger.warning(f'Overridding {self.act_fn} activation with gelu. Use the transformed model at your own risk, the output logits could be significantly different.')
            self.act_fn = 'gelu'
        elif self.act_fn not in SUPPORTED_ACTIVATION_FUNCTIONS:
            raise ValueError(f'Activation function {self.act_fn} not supported for `BetterTransformer` integration.')
        self.use_gelu = self.act_fn == 'gelu' or self.act_fn == 'gelu_new'
        if self.num_heads % 2 == 1:
            raise ValueError(f'Number of heads {self.num_heads} is not supported for `BetterTransformer` integration. Number of heads must be even.')

    def forward_checker(self, *args, **kwargs):
        if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
            raise ValueError('Autocast is not supported for `BetterTransformer` integration.')
        if self.training:
            raise ValueError('Training is not supported for `BetterTransformer` integration.', ' Please use `model.eval()` before running the model.')


class AlbertLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, albert_layer, config):
        """
        A simple conversion of the ALBERT layer to its `BetterTransformer` implementation.

        Args:
            albert_layer (`torch.nn.Module`):
                The original ALBERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([albert_layer.attention.query.weight, albert_layer.attention.key.weight, albert_layer.attention.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([albert_layer.attention.query.bias, albert_layer.attention.key.bias, albert_layer.attention.value.bias]))
        self.out_proj_weight = albert_layer.attention.dense.weight
        self.out_proj_bias = albert_layer.attention.dense.bias
        self.linear1_weight = albert_layer.ffn.weight
        self.linear1_bias = albert_layer.ffn.bias
        self.linear2_weight = albert_layer.ffn_output.weight
        self.linear2_bias = albert_layer.ffn_output.bias
        self.norm1_eps = albert_layer.attention.LayerNorm.eps
        self.norm1_weight = albert_layer.attention.LayerNorm.weight
        self.norm1_bias = albert_layer.attention.LayerNorm.bias
        self.norm2_eps = albert_layer.full_layer_layer_norm.eps
        self.norm2_weight = albert_layer.full_layer_layer_norm.weight
        self.norm2_bias = albert_layer.full_layer_layer_norm.bias
        self.num_heads = albert_layer.attention.num_attention_heads
        self.embed_dim = albert_layer.attention.all_head_size
        self.is_last_layer = False
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        if hidden_states.is_nested:
            attention_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
            hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
            attention_mask = None
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        if hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0)
        return hidden_states,


class BertLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, bert_layer, config):
        """
        A simple conversion of the BERT layer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([bert_layer.attention.self.query.weight, bert_layer.attention.self.key.weight, bert_layer.attention.self.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([bert_layer.attention.self.query.bias, bert_layer.attention.self.key.bias, bert_layer.attention.self.value.bias]))
        self.out_proj_weight = bert_layer.attention.output.dense.weight
        self.out_proj_bias = bert_layer.attention.output.dense.bias
        self.linear1_weight = bert_layer.intermediate.dense.weight
        self.linear1_bias = bert_layer.intermediate.dense.bias
        self.linear2_weight = bert_layer.output.dense.weight
        self.linear2_bias = bert_layer.output.dense.bias
        self.norm1_eps = bert_layer.attention.output.LayerNorm.eps
        self.norm1_weight = bert_layer.attention.output.LayerNorm.weight
        self.norm1_bias = bert_layer.attention.output.LayerNorm.bias
        self.norm2_eps = bert_layer.output.LayerNorm.eps
        self.norm2_weight = bert_layer.output.LayerNorm.weight
        self.norm2_bias = bert_layer.output.LayerNorm.bias
        self.num_heads = bert_layer.attention.self.num_attention_heads
        self.embed_dim = bert_layer.attention.self.all_head_size
        self.is_last_layer = False
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        if hidden_states.is_nested:
            attention_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
            hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
            attention_mask = None
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        if hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0)
        return hidden_states,


class BartEncoderLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, bart_layer, config):
        """
        A simple conversion of the `BartEncoderLayer` to its `BetterTransformer` implementation.

        Args:
            bart_layer (`torch.nn.Module`):
                The original `BartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([bart_layer.self_attn.q_proj.weight, bart_layer.self_attn.k_proj.weight, bart_layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([bart_layer.self_attn.q_proj.bias, bart_layer.self_attn.k_proj.bias, bart_layer.self_attn.v_proj.bias]))
        self.out_proj_weight = bart_layer.self_attn.out_proj.weight
        self.out_proj_bias = bart_layer.self_attn.out_proj.bias
        self.linear1_weight = bart_layer.fc1.weight
        self.linear1_bias = bart_layer.fc1.bias
        self.linear2_weight = bart_layer.fc2.weight
        self.linear2_bias = bart_layer.fc2.bias
        self.norm1_eps = bart_layer.self_attn_layer_norm.eps
        self.norm1_weight = bart_layer.self_attn_layer_norm.weight
        self.norm1_bias = bart_layer.self_attn_layer_norm.bias
        self.norm2_eps = bart_layer.final_layer_norm.eps
        self.norm2_weight = bart_layer.final_layer_norm.weight
        self.norm2_bias = bart_layer.final_layer_norm.bias
        self.num_heads = bart_layer.self_attn.num_heads
        self.embed_dim = bart_layer.self_attn.embed_dim
        self.is_last_layer = False
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, position_bias=None, *_, **__):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        if not hasattr(hidden_states, 'original_shape'):
            original_shape = hidden_states.shape
        else:
            original_shape = hidden_states.original_shape
        if hidden_states.is_nested:
            attention_mask = None
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask.squeeze(1)[:, 0]
            attention_mask = attention_mask.bool()
            attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
            hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
            attention_mask = None
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        if not self.is_last_layer:
            hidden_states.original_shape = original_shape
        elif hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        return hidden_states,


class MBartEncoderLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, mbart_layer, config):
        """
        A simple conversion of the `MBartEncoderLayer` to its `BetterTransformer` implementation.
        Args:
            mbart_layer (`torch.nn.Module`):
                The original `MBartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([mbart_layer.self_attn.q_proj.weight, mbart_layer.self_attn.k_proj.weight, mbart_layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([mbart_layer.self_attn.q_proj.bias, mbart_layer.self_attn.k_proj.bias, mbart_layer.self_attn.v_proj.bias]))
        self.out_proj_weight = mbart_layer.self_attn.out_proj.weight
        self.out_proj_bias = mbart_layer.self_attn.out_proj.bias
        self.linear1_weight = mbart_layer.fc1.weight
        self.linear1_bias = mbart_layer.fc1.bias
        self.linear2_weight = mbart_layer.fc2.weight
        self.linear2_bias = mbart_layer.fc2.bias
        self.norm1_eps = mbart_layer.self_attn_layer_norm.eps
        self.norm1_weight = mbart_layer.self_attn_layer_norm.weight
        self.norm1_bias = mbart_layer.self_attn_layer_norm.bias
        self.norm2_eps = mbart_layer.final_layer_norm.eps
        self.norm2_weight = mbart_layer.final_layer_norm.weight
        self.norm2_bias = mbart_layer.final_layer_norm.bias
        self.num_heads = mbart_layer.self_attn.num_heads
        self.embed_dim = mbart_layer.self_attn.embed_dim
        self.is_last_layer = False
        self.norm_first = True
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, position_bias=None, *_, **__):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        if not hasattr(hidden_states, 'original_shape'):
            original_shape = hidden_states.shape
        else:
            original_shape = hidden_states.original_shape
        if hidden_states.is_nested:
            attention_mask = None
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask.squeeze(1)[:, 0]
            attention_mask = attention_mask.bool()
            attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
            hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
            attention_mask = None
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        if not self.is_last_layer:
            hidden_states.original_shape = original_shape
        elif hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        return hidden_states,


class DistilBertLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, bert_layer, config):
        """
        A simple conversion of the Distill-BERTLayer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original Distill-BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([bert_layer.attention.q_lin.weight, bert_layer.attention.k_lin.weight, bert_layer.attention.v_lin.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([bert_layer.attention.q_lin.bias, bert_layer.attention.k_lin.bias, bert_layer.attention.v_lin.bias]))
        self.out_proj_weight = bert_layer.attention.out_lin.weight
        self.out_proj_bias = bert_layer.attention.out_lin.bias
        self.linear1_weight = bert_layer.ffn.lin1.weight
        self.linear1_bias = bert_layer.ffn.lin1.bias
        self.linear2_weight = bert_layer.ffn.lin2.weight
        self.linear2_bias = bert_layer.ffn.lin2.bias
        self.norm1_eps = bert_layer.sa_layer_norm.eps
        self.norm1_weight = bert_layer.sa_layer_norm.weight
        self.norm1_bias = bert_layer.sa_layer_norm.bias
        self.norm2_eps = bert_layer.output_layer_norm.eps
        self.norm2_weight = bert_layer.output_layer_norm.weight
        self.norm2_bias = bert_layer.output_layer_norm.bias
        self.num_heads = bert_layer.attention.n_heads
        self.embed_dim = bert_layer.attention.dim
        self.is_last_layer = False
        self.validate_bettertransformer()

    def forward(self, x, attn_mask, head_mask=None, output_attentions=None, *_):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        if x.is_nested:
            attn_mask = None
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn_mask = torch.reshape(attn_mask, (attn_mask.shape[0], attn_mask.shape[-1]))
            seqlen = attn_mask.shape[1]
            lengths = torch.sum(~attn_mask, 1)
            if not all([(l == seqlen) for l in lengths]):
                x = torch._nested_tensor_from_mask(x, attn_mask)
            attn_mask = None
        x = torch._transformer_encoder_layer_fwd(x, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attn_mask)
        if x.is_nested and self.is_last_layer:
            x = x.to_padded_tensor(0.0)
        return x,


class WhisperEncoderLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, whisper_layer, config):
        """
        A simple conversion of the WhisperEncoderLayer to its `BetterTransformer` implementation.

        Args:
            whisper_layer (`torch.nn.Module`):
                The original `WhisperEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([whisper_layer.self_attn.q_proj.weight, whisper_layer.self_attn.k_proj.weight, whisper_layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([whisper_layer.self_attn.q_proj.bias, torch.zeros_like(whisper_layer.self_attn.q_proj.bias), whisper_layer.self_attn.v_proj.bias]))
        self.out_proj_weight = whisper_layer.self_attn.out_proj.weight
        self.out_proj_bias = whisper_layer.self_attn.out_proj.bias
        self.linear1_weight = whisper_layer.fc1.weight
        self.linear1_bias = whisper_layer.fc1.bias
        self.linear2_weight = whisper_layer.fc2.weight
        self.linear2_bias = whisper_layer.fc2.bias
        self.norm1_eps = whisper_layer.self_attn_layer_norm.eps
        self.norm1_weight = whisper_layer.self_attn_layer_norm.weight
        self.norm1_bias = whisper_layer.self_attn_layer_norm.bias
        self.norm2_eps = whisper_layer.final_layer_norm.eps
        self.norm2_weight = whisper_layer.final_layer_norm.weight
        self.norm2_bias = whisper_layer.final_layer_norm.bias
        self.num_heads = whisper_layer.self_attn.num_heads
        self.embed_dim = whisper_layer.self_attn.embed_dim
        self.is_last_layer = False
        self.norm_first = True
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_, **__):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        attention_mask = None
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        if hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0)
        return hidden_states,


class ViTLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, vit_layer, config):
        """
        A simple conversion of the ViTLayer to its `BetterTransformer` implementation.

        Args:
            vit_layer (`torch.nn.Module`):
                The original `ViTLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([vit_layer.attention.attention.query.weight, vit_layer.attention.attention.key.weight, vit_layer.attention.attention.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([vit_layer.attention.attention.query.bias, vit_layer.attention.attention.key.bias, vit_layer.attention.attention.value.bias]))
        self.out_proj_weight = vit_layer.attention.output.dense.weight
        self.out_proj_bias = vit_layer.attention.output.dense.bias
        self.linear1_weight = vit_layer.intermediate.dense.weight
        self.linear1_bias = vit_layer.intermediate.dense.bias
        self.linear2_weight = vit_layer.output.dense.weight
        self.linear2_bias = vit_layer.output.dense.bias
        self.norm1_eps = vit_layer.layernorm_before.eps
        self.norm1_weight = vit_layer.layernorm_before.weight
        self.norm1_bias = vit_layer.layernorm_before.bias
        self.norm2_eps = vit_layer.layernorm_after.eps
        self.norm2_weight = vit_layer.layernorm_after.weight
        self.norm2_bias = vit_layer.layernorm_after.bias
        self.num_heads = vit_layer.attention.attention.num_attention_heads
        self.embed_dim = int(vit_layer.attention.attention.attention_head_size * self.num_heads)
        self.is_last_layer = False
        self.norm_first = True
        self.validate_bettertransformer()

    def forward(self, hidden_states, *_, **__):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        attention_mask = None
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        if hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0)
        return hidden_states,


class ViltLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, vilt_layer, config):
        """
        A simple conversion of the VilTLayer to its `BetterTransformer` implementation.

        Args:
            vilt_layer (`torch.nn.Module`):
                The original `VilTLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([vilt_layer.attention.attention.query.weight, vilt_layer.attention.attention.key.weight, vilt_layer.attention.attention.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([vilt_layer.attention.attention.query.bias, vilt_layer.attention.attention.key.bias, vilt_layer.attention.attention.value.bias]))
        self.out_proj_weight = vilt_layer.attention.output.dense.weight
        self.out_proj_bias = vilt_layer.attention.output.dense.bias
        self.linear1_weight = vilt_layer.intermediate.dense.weight
        self.linear1_bias = vilt_layer.intermediate.dense.bias
        self.linear2_weight = vilt_layer.output.dense.weight
        self.linear2_bias = vilt_layer.output.dense.bias
        self.norm1_eps = vilt_layer.layernorm_before.eps
        self.norm1_weight = vilt_layer.layernorm_before.weight
        self.norm1_bias = vilt_layer.layernorm_before.bias
        self.norm2_eps = vilt_layer.layernorm_after.eps
        self.norm2_weight = vilt_layer.layernorm_after.weight
        self.norm2_bias = vilt_layer.layernorm_after.bias
        self.num_heads = vilt_layer.attention.attention.num_attention_heads
        self.embed_dim = int(vilt_layer.attention.attention.attention_head_size * self.num_heads)
        self.is_last_layer = False
        self.norm_first = True
        self.validate_bettertransformer()

    def forward(self, hidden_states, *_, **__):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        attention_mask = None
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        if hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0)
        return hidden_states,


class Wav2Vec2EncoderLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, wav2vec2_layer, config):
        """
        A simple conversion of the Wav2Vec2EncoderLayer to its `BetterTransformer` implementation.

        Args:
            wav2vec2_layer (`torch.nn.Module`):
                The original `Wav2Vec2EncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([wav2vec2_layer.attention.q_proj.weight, wav2vec2_layer.attention.k_proj.weight, wav2vec2_layer.attention.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([wav2vec2_layer.attention.q_proj.bias, wav2vec2_layer.attention.k_proj.bias, wav2vec2_layer.attention.v_proj.bias]))
        self.out_proj_weight = wav2vec2_layer.attention.out_proj.weight
        self.out_proj_bias = wav2vec2_layer.attention.out_proj.bias
        self.linear1_weight = wav2vec2_layer.feed_forward.intermediate_dense.weight
        self.linear1_bias = wav2vec2_layer.feed_forward.intermediate_dense.bias
        self.linear2_weight = wav2vec2_layer.feed_forward.output_dense.weight
        self.linear2_bias = wav2vec2_layer.feed_forward.output_dense.bias
        self.norm1_eps = wav2vec2_layer.layer_norm.eps
        self.norm1_weight = wav2vec2_layer.layer_norm.weight
        self.norm1_bias = wav2vec2_layer.layer_norm.bias
        self.norm2_eps = wav2vec2_layer.final_layer_norm.eps
        self.norm2_weight = wav2vec2_layer.final_layer_norm.weight
        self.norm2_bias = wav2vec2_layer.final_layer_norm.bias
        self.num_heads = wav2vec2_layer.attention.num_heads
        self.embed_dim = wav2vec2_layer.attention.embed_dim
        self.is_last_layer = False
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, **__):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        if hidden_states.is_nested:
            attention_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask.squeeze(1)[:, 0]
            attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
            hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
            attention_mask = None
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        if hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0)
        return hidden_states,


class FSMTEncoderLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, fsmt_layer, config):
        """
        A simple conversion of the FSMT Encoder layer to its `BetterTransformer` implementation.

        Args:
            fsmt_layer (`torch.nn.Module`):
                The original FSMT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([fsmt_layer.self_attn.q_proj.weight, fsmt_layer.self_attn.k_proj.weight, fsmt_layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([fsmt_layer.self_attn.q_proj.bias, fsmt_layer.self_attn.k_proj.bias, fsmt_layer.self_attn.v_proj.bias]))
        self.out_proj_weight = fsmt_layer.self_attn.out_proj.weight
        self.out_proj_bias = fsmt_layer.self_attn.out_proj.bias
        self.linear1_weight = fsmt_layer.fc1.weight
        self.linear1_bias = fsmt_layer.fc1.bias
        self.linear2_weight = fsmt_layer.fc2.weight
        self.linear2_bias = fsmt_layer.fc2.bias
        self.norm1_eps = fsmt_layer.self_attn_layer_norm.eps
        self.norm1_weight = fsmt_layer.self_attn_layer_norm.weight
        self.norm1_bias = fsmt_layer.self_attn_layer_norm.bias
        self.norm2_eps = fsmt_layer.final_layer_norm.eps
        self.norm2_weight = fsmt_layer.final_layer_norm.weight
        self.norm2_bias = fsmt_layer.final_layer_norm.bias
        self.num_heads = fsmt_layer.self_attn.num_heads
        self.embed_dim = fsmt_layer.self_attn.embed_dim
        self.is_last_layer = False
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, position_bias=None, *_, **__):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        if not hasattr(hidden_states, 'original_shape'):
            original_shape = hidden_states.shape
        else:
            original_shape = hidden_states.original_shape
        if hidden_states.is_nested:
            attention_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
            if hidden_states.shape[0] != attention_mask.shape[0]:
                hidden_states = hidden_states.transpose(1, 0)
                original_shape = hidden_states.shape
            hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
            attention_mask = None
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        if not self.is_last_layer:
            hidden_states.original_shape = original_shape
        elif hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        return hidden_states, attention_mask


class CLIPLayerBetterTransformer(BetterTransformerBaseLayer):

    def __init__(self, layer, config):
        """
        A simple conversion of the CLIPEncoderLayer to its `BetterTransformer` implementation.

        **The implementation is valid only for the vision model, that does not use `causal_attention_mask`.**

        Args:
            layer (`torch.nn.Module`):
                The original `CLIPEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        self.in_proj_weight = nn.Parameter(torch.cat([layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([layer.self_attn.q_proj.bias, layer.self_attn.k_proj.bias, layer.self_attn.v_proj.bias]))
        self.out_proj_weight = layer.self_attn.out_proj.weight
        self.out_proj_bias = layer.self_attn.out_proj.bias
        self.linear1_weight = layer.mlp.fc1.weight
        self.linear1_bias = layer.mlp.fc1.bias
        self.linear2_weight = layer.mlp.fc2.weight
        self.linear2_bias = layer.mlp.fc2.bias
        self.norm1_eps = layer.layer_norm1.eps
        self.norm1_weight = layer.layer_norm1.weight
        self.norm1_bias = layer.layer_norm1.bias
        self.norm2_eps = layer.layer_norm2.eps
        self.norm2_weight = layer.layer_norm2.weight
        self.norm2_bias = layer.layer_norm2.bias
        self.num_heads = layer.self_attn.num_heads
        self.embed_dim = layer.self_attn.embed_dim
        self.is_last_layer = False
        self.norm_first = True
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_, **__):
        """
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        if attention_mask is not None:
            raise ValueError('Please do not use attention masks when using `BetterTransformer` converted vision models')
        hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        return hidden_states,

    def _get_activation_function(self, config: 'PretrainedConfig'):
        if hasattr(config, 'vision_config') and hasattr(config, 'text_config'):
            assert config.vision_config.hidden_act == config.text_config.hidden_act
            return config.vision_config.hidden_act
        else:
            return config.hidden_act


CONFIG_NAME = 'config.json'


FROM_PRETRAINED_START_DOCSTRING = """
    Instantiate a pretrained model from a pre-trained model configuration.

    Args:
        model_id (`Union[str, Path]`):
            Can be either:
                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing a model saved using [`~OptimizedModel.save_pretrained`],
                    e.g., `./my_model_directory/`.
        from_transformers (`bool`, *optional*, defaults to `False`):
            Defines whether the provided `model_id` contains a vanilla Transformers checkpoint.
        force_download (`bool`, *optional*, defaults to `True`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        use_auth_token (`Optional[str]`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        cache_dir (`Optional[str]`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        config (`Optional[transformers.PretrainedConfig]`, *optional*):
            The model configuration.
        local_files_only(`bool`, *optional*, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
"""

