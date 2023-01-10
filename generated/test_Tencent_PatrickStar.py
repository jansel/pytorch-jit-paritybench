import sys
_module = sys.modules[__name__]
del sys
master = _module
generate_res_table = _module
is_run_this_file = _module
process_logs = _module
data_loader = _module
eval_chunk_size = _module
huggingface_bert = _module
imdb_dataset = _module
model_builder = _module
huggingface_bert_moe = _module
moe_bert = _module
optimizations = _module
checkpoint = _module
global_opt_flags = _module
ls_hf_transformer_encoder_layer = _module
ps_tile_modeling_bert = _module
test_tiling = _module
tiling = _module
parse_args = _module
pretrain_demo = _module
ps_config = _module
simple_net = _module
train_simple_net = _module
patrickstar = _module
core = _module
chunk_data = _module
chunk_list = _module
chunk_tensor_index = _module
client = _module
comm = _module
const = _module
eviction_policy = _module
hook = _module
memory_cache = _module
memtracer = _module
memtracer = _module
metronome = _module
training_stage_mgr = _module
parameter = _module
preprocess = _module
tensor_stub = _module
torch_profiler_hook = _module
fp16 = _module
loss_scaler = _module
manager = _module
cuda_context = _module
runtime_config = _module
ops = _module
chunk_io_buff = _module
embedding = _module
fp16_cpu_adam = _module
op_builder = _module
builder = _module
cpu_adam = _module
profiler = _module
runtime = _module
checkpoint = _module
engine = _module
utils = _module
distributed = _module
global_timer = _module
helper = _module
logging = _module
memory = _module
memory_monitor = _module
model_size_calculator = _module
singleton_meta = _module
setup = _module
merge_checkpoint = _module
profile_visualizer = _module
common = _module
test_checkpoint = _module
test_chunk_data = _module
test_chunk_list = _module
test_chunk_tensor_index = _module
test_client = _module
test_ds_cpu_adam = _module
test_embedding_ops = _module
test_eviction_policy = _module
test_hf_checkpoint = _module
test_loss_scale = _module
test_memory_cache = _module
test_model_init = _module
test_optimizer_init = _module
test_torch_scope = _module
test_utils = _module

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


from torch.utils.data import SequentialSampler


import logging


import time


from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split


import warnings


from typing import Any


from typing import Iterable


from typing import List


from typing import Tuple


import math


from torch import nn


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


import copy


from math import floor as floor


import numpy as np


from torch.utils.checkpoint import checkpoint


from typing import Optional


from time import sleep


import functools


from copy import deepcopy


import torch.nn as nn


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


import itertools


import torch.distributed as dist


from torch.utils.cpp_extension import BuildExtension


import matplotlib.pyplot as plt


import matplotlib.patches as patches


from torch.multiprocessing import Process


from torch.nn import Embedding as TorchEmbedding


class BertSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        if self.is_decoder:
            past_key_value = key_layer, value_layer
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
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
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


def partition_uniform(num_items, num_parts):
    parts = [0] * (num_parts + 1)
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts
    chunksize = floor(num_items / num_parts)
    for p in range(num_parts):
        parts[p] = min(chunksize * p, num_items)
    parts[num_parts] = num_items
    return parts


def split_tensor_along_last_dim(tensor, partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension. Adapted from Megatron-LM.
    Arguments:
        tensor: input tensor.
        partitions: list of partition sizes to supply to torch.split
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    last_dim = tensor.dim() - 1
    tensor_list = torch.split(tensor, partitions, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


class TiledLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, in_splits=1, out_splits=1, input_is_already_split=False, combine_out_splits=True, linear_cls=torch.nn.Linear, init_linear=None, **kwargs):
        """A replacement for ``torch.nn.Linear`` that works with ZeRO-3 to reduce
        memory requirements via tiling.
        TiledLinear breaks the input and output dimensions of a linear layer
        into tiles that are processed in sequence. This class enables huge
        linear layers when combined with ZeRO-3 because inactive tiles can be
        partitioned and offloaded.
        .. note::
            We recommend using as few tiles as necessary. Tiling
            significantly reduces memory usage, but can reduce throughput
            for inexpensive layers. This due to the smaller kernels having
            less parallelism and lower arithmetic intensity, while
            introducing more frequent synchronization and communication.
        Args:
            in_features (int): See ``torch.nn.Linear``
            out_features (int): See ``torch.nn.Linear``
            bias (bool, optional): See ``torch.nn.Linear``
            in_splits (int, optional): The number of tiles along the input dimension. Defaults to 1.
            out_splits (int, optional): The number of tiles along the output dimension. Defaults to 1.
            input_is_already_split (bool, optional): If set to ``True``, assume that the ``input_`` in
                to ``forward()`` is already split into ``in_splits`` chunks. Defaults to ``False``.
            combine_out_splits (bool, optional): If set to ``False``, do not combine the ``out_splits`` outputs
                into a single tensor. Defaults to ``True``.
            linear_cls (class, optional): The underlying class to build individual tiles.
                Defaults to ``torch.nn.Linear``.
            init_linear (``torch.nn.Linear``, optional): If set, copy the parameters of
                ``init_linear``. Useful for debugging. Defaults to ``None``.
            kwargs (dict, optional): additional keyword arguments to provide to ``linear_cls()``.
        Raises:
            RuntimeError: ``in_splits`` must be within the range [1, in_features).
            RuntimeError: ``out_splits`` must be within the range of [1, out_features).
        """
        super().__init__()
        if in_splits < 1 or in_splits > in_features:
            raise RuntimeError('in splits must be in range [1, in_features].')
        if out_splits < 1 or out_splits > out_features:
            raise RuntimeError('out splits must be in range [1, out_features].')
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.out_splits = out_splits
        self.in_splits = in_splits
        self.input_is_already_split = input_is_already_split
        self.combine_out_splits = combine_out_splits
        self.in_parts = partition_uniform(num_items=in_features, num_parts=in_splits)
        self.out_parts = partition_uniform(num_items=out_features, num_parts=out_splits)
        assert len(self.out_parts) == out_splits + 1
        assert len(self.in_parts) == in_splits + 1
        assert self.out_parts[0] == 0
        assert self.out_parts[out_splits] == out_features
        assert self.in_parts[in_splits] == in_features
        self.linears = torch.nn.ModuleList()
        for out_id in range(out_splits):
            self.linears.append(torch.nn.ModuleList())
            local_out_dim = self.out_parts[out_id + 1] - self.out_parts[out_id]
            for in_id in range(in_splits):
                local_bias = bias if in_id == in_splits - 1 else False
                local_in_dim = self.in_parts[in_id + 1] - self.in_parts[in_id]
                local = linear_cls(local_in_dim, local_out_dim, bias=local_bias, **kwargs)
                self.linears[out_id].append(local)
        if init_linear is not None:
            self.copy_params_from(init_linear)

    @torch.no_grad()
    def copy_params_from(self, other):
        """Copy the weight and bias data from ``other``.
        This is especially useful for reproducible initialization and testing.
        Equivalent to:
        .. code-block:: python
            with torch.no_grad():
                self.weight.copy_(other.weight)
                if self.bias is not None:
                    self.bias.copy_(other.bias)
        .. note::
            If ZeRO-3 is enabled, this is a collective operation and the
            updated parameters of data-parallel rank 0 will be visible on all
            ranks. See :class:`deepspeed.zero.GatheredParameters` for more
            information.
        Args:
            other (``torch.nn.Linear``): the linear layer to copy from.
        """
        assert hasattr(other, 'weight')
        assert other.weight.size() == (self.out_features, self.in_features)
        if self.use_bias:
            assert hasattr(other, 'bias')
            assert other.bias is not None
            assert other.bias.size() == (self.out_features,)
        else:
            assert other.bias is None
        for row in range(self.out_splits):
            rstart = self.out_parts[row]
            rstop = self.out_parts[row + 1]
            for col in range(self.in_splits):
                cstart = self.in_parts[col]
                cstop = self.in_parts[col + 1]
                local = self.linears[row][col]
                global_weight = other.weight[rstart:rstop, cstart:cstop]
                local.weight.copy_(global_weight)
            if local.bias is not None:
                local.bias.data.copy_(other.bias[rstart:rstop].data)

    def forward(self, input_):
        if self.in_splits > 1 and not self.input_is_already_split:
            input_parts = partition_uniform(input_.shape[-1], self.in_splits)
            split_sizes = [(input_parts[p + 1] - input_parts[p]) for p in range(self.in_splits)]
            inputs = self._split_global_input(input_, split_sizes)
        elif self.in_splits > 1:
            inputs = input_
            assert len(inputs) == self.in_splits, f'Col splits {self.in_splits} does not match input splits {len(inputs)}'
        else:
            inputs = [input_]
        outputs = [None] * self.out_splits
        for out_id in range(self.out_splits):
            for in_id in range(self.in_splits):
                local_output = self.linears[out_id][in_id](inputs[in_id])
                outputs[out_id] = self._reduce_local_output(in_id=in_id, out_id=out_id, current_out=outputs[out_id], new_out=local_output)
        if self.combine_out_splits:
            return self._combine_output_splits(outputs)
        return outputs

    def _split_global_input(self, input, split_sizes):
        """Partition an input tensor along the last dimension, aligned with given splits.
        Subclasses should override this method to account for new input types.
        Args:
            input (List[Tensor]): The tensor to partition along the last dimension.
            split_sizes (List[int]): The size of each partition.
        Returns:
            List[Any]: A list of the chunks of ``input``.
        """
        return split_tensor_along_last_dim(input, split_sizes)

    def _reduce_local_output(self, in_id, out_id, current_out, new_out):
        """Reduce (sum) a new local result into the existing local results.
        Subclasses should override this method.
        For a given ``out_id``, this method is called ``in_id-1`` times. The first input
        split is a simple assignment.
        Args:
            in_id (int): The input split that produced ``new_out``.
            out_id (int): The output split that produced ``new_out``.
            current_out (Any): The reduced form of all previous ``out_id`` results.
            new_out (Any): The local result from forward (``in_id``, ``out_id``)e
        Returns:
            Any: The combined result of ``current_out`` and ``new_out``.
        """
        if current_out is None:
            return new_out.clone()
        else:
            return current_out + new_out

    def _combine_output_splits(self, outputs):
        """Join the splits of the output into a single result.
        Args:
            outputs (List[Any]): The reduced outputs for each output split.
        Returns:
            Any: The combined outputs.
        """
        assert len(outputs) == self.out_splits
        return torch.cat(outputs, dim=-1)


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        if global_opt_flags.USE_TILE:
            self.dense = TiledLinear(in_features=config.hidden_size, out_features=config.intermediate_size, linear_cls=nn.Linear, in_splits=1, out_splits=4)
        else:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        if global_opt_flags.USE_TILE:
            self.dense = TiledLinear(in_features=config.intermediate_size, out_features=config.hidden_size, linear_cls=nn.Linear, in_splits=4, out_splits=1)
        else:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = BertAttention(config, position_embedding_type='absolute')
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed,                        {self} has to be instantiated with cross-attention                             layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder(torch.nn.Module):

    def __init__(self, hidden_dim, is_ckp=False):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim))
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.is_ckp = is_ckp

    def forward(self, x):
        h2 = self.linear1(x)
        if self.is_ckp:
            h3 = checkpoint(self.linear3, h2)
        else:
            h3 = self.linear3(h2)
        h4 = self.linear4(h3)
        h5 = self.linear5(h4)
        return h5


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, seq_len, is_ckp=False, is_share_param=False):
        super(SimpleModel, self).__init__()
        config = BertConfig()
        config.vocab_size = 25
        config.max_position_embeddings = seq_len
        config.hidden_size = hidden_dim
        self.embeddings_1 = BertEmbeddings(config)
        self._is_share_param = is_share_param
        if is_share_param:
            self.embeddings_2 = self.embeddings_1
        else:
            self.embeddings_2 = BertEmbeddings(config)
        self.encoder = Encoder(hidden_dim, is_ckp)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        h1 = self.embeddings_1(x)
        h2 = self.embeddings_2(x)
        h3 = h1 + h2
        h3 = self.encoder(h3)
        return self.cross_entropy_loss(h3[:, 0], y)


class LoggerFactory:

    @staticmethod
    def create_logger(name=None, level=logging.WARNING):
        """create a logger
        Args:
            name (str): name of the logger
            level: level of logger
        Raises:
            ValueError is name is None
        """
        if name is None:
            raise ValueError('name for logger cannot be None')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger_.addHandler(RichHandler())
        return logger_


class _CopyInputToCPU(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        logger.debug(f'Copy input to cpu and {input_.dtype}.')
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        logger.debug('Copy grad_output to cuda.')
        return grad_output


def copy_to_cpu(input_):
    return _CopyInputToCPU.apply(input_)


class _CopyActToGPU(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_):
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        return input_

    @staticmethod
    def forward(ctx, input_):
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        logger.debug(f'Copy grad_output to cuda, input dtype {input_.dtype}.')
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.float()


def copy_to_gpu(input_):
    return _CopyActToGPU.apply(input_)


class Embedding(nn.Embedding):
    """CPU Embedding.

    If `use_cpu` is set, the embedding operations will
    be performed on CPU.
    """
    use_cpu = False
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cpu = Embedding.use_cpu
        Embedding.instances.append(self)

    def forward(self, input_):
        if self.use_cpu:
            input_ = copy_to_cpu(input_)
        else:
            input_ = copy_to_gpu(input_)
        output = super().forward(input_)
        if self.use_cpu:
            output = copy_to_gpu(output)
        return output


class DynamicLossScaler:
    """
    Class that manages dynamic loss scaling.  It is recommended to use :class:`DynamicLossScaler`
    indirectly, by supplying ``dynamic_loss_scale=True`` to the constructor of
    :class:`Fp16Optimizer`.  However, it's important to understand how :class:`DynamicLossScaler`
    operates, because the default options can be changed using the
    the ``dynamic_loss_args`` argument to :class:`Fp16Optimizer`'s constructor.
    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.  If overflowing gradients are
    encountered, :class:`DynamicLossScaler` informs :class:`Fp16Optimizer` that an overflow has
    occurred.
    :class:`Fp16Optimizer` then skips the update step for this particular iteration/minibatch,
    and :class:`DynamicLossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients detected,
    :class:`DynamicLossScaler` increases the loss scale once more.
    In this way :class:`DynamicLossScaler` attempts to "ride the edge" of
    always using the highest loss scale possible without incurring overflow.
    Args:
        init_scale (float, optional, default=2**32):  Initial loss scale attempted by :class:`DynamicLossScaler.`
        scale_factor (float, optional, default=2.0):  Factor used when adjusting the loss scale. If an overflow
        is encountered, the loss scale is readjusted to loss scale/``scale_factor``.
        If ``scale_window`` consecutive iterations take place without an overflow,
        the loss scale is readjusted to loss_scale*``scale_factor``.
        scale_window (int, optional, default=1000):  Number of consecutive iterations
        without an overflow to wait before increasing the loss scale.
    """

    def __init__(self, init_scale=2 ** 32, scale_factor=2.0, scale_window=1000, min_scale=1, delayed_shift=1, consecutive_hysteresis=False):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis

    def has_overflow(self, param):
        if DynamicLossScaler._has_inf_or_nan(param.grad):
            return True
        return False

    def _has_inf_or_nan(x):
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def update_scale(self, overflow):
        if not hasattr(self, 'min_scale'):
            self.min_scale = 1
        if not hasattr(self, 'delayed_shift'):
            self.delayed_shift = 1
        if not hasattr(self, 'cur_hysteresis'):
            self.cur_hysteresis = 1
        if not hasattr(self, 'consecutive_hysteresis'):
            self.consecutive_hysteresis = True
        if overflow:
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
            else:
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)


DEFAULT_TORCH_EXTENSION_PATH = '/tmp/torch_extensions'


END = '\x1b[0m'


YELLOW = '\x1b[93m'


WARNING = f'{YELLOW} [WARNING] {END}'


cuda_minor_mismatch_ok = {(10): ['10.0', '10.1', '10.2'], (11): ['11.0', '11.1', '11.2', '11.3']}


def installed_cuda_version():
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    assert cuda_home is not None, 'CUDA_HOME does not exist, unable to compile CUDA op(s)'
    output = subprocess.check_output([cuda_home + '/bin/nvcc', '-V'], universal_newlines=True)
    output_split = output.split()
    release_idx = output_split.index('release')
    release = output_split[release_idx + 1].replace(',', '').split('.')
    cuda_major, cuda_minor = release[:2]
    return int(cuda_major), int(cuda_minor)


def assert_no_cuda_mismatch():
    cuda_major, cuda_minor = installed_cuda_version()
    sys_cuda_version = f'{cuda_major}.{cuda_minor}'
    torch_cuda_version = '.'.join(torch.version.cuda.split('.')[:2])
    if sys_cuda_version != torch_cuda_version:
        if cuda_major in cuda_minor_mismatch_ok and sys_cuda_version in cuda_minor_mismatch_ok[cuda_major] and torch_cuda_version in cuda_minor_mismatch_ok[cuda_major]:
            None
            return
        raise Exception(f'Installed CUDA version {sys_cuda_version} does not match the version torch was compiled with {torch.version.cuda}, unable to compile cuda/cpp extensions without a matching cuda version.')


DEFAULT_COMPUTE_CAPABILITIES = '6.0;6.1;7.0'


def get_default_compute_capatabilities():
    compute_caps = DEFAULT_COMPUTE_CAPABILITIES
    import torch.utils.cpp_extension
    if torch.utils.cpp_extension.CUDA_HOME is not None and installed_cuda_version()[0] >= 11:
        if installed_cuda_version()[0] == 11 and installed_cuda_version()[1] == 0:
            compute_caps += ';8.0'
        else:
            compute_caps += ';8.0;8.6'
    return compute_caps


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


class CUDAContext(metaclass=SingletonMeta):

    def __init__(self):
        self.compute_stream = torch.cuda.current_stream()
        if get_world_size() == 1:
            self.copy_stream = torch.Stream()
        else:
            logger.warning('Asynchronized move will not be enabled for world size larger than 1')
            self.copy_stream = self.compute_stream


def get_local_world_size():
    global _local_world_size
    if _local_world_size is None:
        if torch.distributed.is_initialized():
            if 'LOCAL_WORLD_SIZE' in os.environ:
                _local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
            else:
                logger.warning("If you are training with multiple nodes, it's recommand to set LOCAL_WORLD_SIZE manually to make better use of CPU memory. Otherwise, get_world_size() is used instead.")
                _local_world_size = get_world_size()
        else:
            _local_world_size = 1
    return _local_world_size


def get_memory_info():
    try:
        mems = {}
        with open('/sys/fs/cgroup/memory/memory.meminfo', 'rb') as f:
            for line in f:
                fields = line.split()
                mems[fields[0]] = int(fields[1]) * 1024
        total = mems[b'MemTotal:']
        free = mems[b'MemFree:']
        cached = mems[b'Cached:']
        buffers = mems[b'Buffers:']
        used = total - free - cached - buffers
        if used < 0:
            used = total - free
        mem_info = ps_mem_info(total=total, free=free, cached=cached, buffers=buffers, used=used)
    except FileNotFoundError:
        mems = psutil.virtual_memory()
        mem_info = ps_mem_info(total=mems.total, free=mems.free, cached=mems.cached, buffers=mems.buffers, used=mems.used)
    return mem_info


def get_sys_memory_used(device):
    """
    Get the free memory info of device.
    Notice that for CPU, this function will return 1/N of the total free memory,
    where N is the world size.
    """
    if device.type == 'cuda':
        ret = torch.cuda.memory_allocated()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
    elif device.type == 'cpu':
        mem_info = get_memory_info()
        ret = mem_info.used / get_local_world_size()
    return ret


class AsyncMemoryMonitor:

    def __init__(self, power=3):
        """
        An Async Mem Monitor runing during computing.
        Sampling GPU memory usage of the current GPU dev
        at interval of 1/(10**power) sec.
        """
        self.keep_measuring = False

        def _set_cuda_device():
            torch.cuda.set_device(torch.cuda.current_device())
        self.executor = ThreadPoolExecutor(max_workers=1, initializer=_set_cuda_device)
        self.monitor_thread = None
        self.interval = 1 / 10 ** power

    def set_interval(self, power: int):
        self.interval = 1 / 10 ** power

    def start(self):
        self.keep_measuring = True
        self.monitor_thread = self.executor.submit(self._measure_usage)

    def finish(self):
        if self.keep_measuring is False:
            return 0
        self.keep_measuring = False
        max_usage = self.monitor_thread.result()
        self.monitor_thread = None
        return max_usage

    def _measure_usage(self):
        max_usage = 0
        dev = torch.device(f'cuda:{torch.cuda.current_device()}')
        while self.keep_measuring:
            max_usage = max(max_usage, get_sys_memory_used(dev))
            sleep(self.interval)
        return max_usage


class TrainingStageMgr:

    def __init__(self):
        """
        Tell us in which stage the training are. (FWD, BWD, ADAM)
        Also tell us whether in an warmup iteration.
        """
        self.training_phase = TrainingStage.UNSTART
        self.is_warmup = False


class Metronome(object):
    """
    A metronome for memory stats sampling.
    Use two indicators to tell us where the training is now
    One is moment, indicates the moment of one iteration.
    The other is training stage, indicates FWD/BWD/ADAM and is this iteration is
    a warmup iteration.

    It also contain the training stage information.
    """

    def __init__(self):
        self._moment = 0
        self._total_moment = None
        self.training_stage_mgr = TrainingStageMgr()

    def set_training_phase(self, phase):
        self.training_stage_mgr.training_phase = phase

    def set_warmup(self, flag):
        self.training_stage_mgr.is_warmup = flag

    def is_warmup(self):
        return self.training_stage_mgr.is_warmup

    def training_stage(self):
        return self.training_stage_mgr.training_phase

    def get_total_mom(self):
        assert self._total_moment is not None, 'Don not use get_total during warmup'
        return self._total_moment

    def tiktac(self):
        """
        The function should be called right before and after computing of an operator.
        """
        self._moment += 1

    def moment(self):
        return self._moment

    def reset(self):
        """
        The function is called after a trainig iteration is finished.
        """
        self._total_moment = self._moment
        self._moment = 0

    def next_moment(self):
        assert self._total_moment is not None
        return min(self._total_moment, self._moment + 1) % self._total_moment

    def prev_moment(self):
        assert self._total_moment is not None
        return max(0, self._moment - 1) % self._total_moment


def log_dist(message, ranks=[0], level=logging.INFO):
    """Log message when one of following condition meets
    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]
    Args:
        message (str)
        ranks (list)
        level (int)
    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or my_rank in set(ranks)
    if should_log:
        final_message = '[Rank {}] {}'.format(my_rank, message)
        logger.log(level, final_message)


class Profiler(metaclass=SingletonMeta):

    def __init__(self):
        self._nested_level = 0
        self.start_time = None
        self.warmup_finish_time = None
        self.end_time = None
        self.gpu_memory_used = []
        self.gpu_chunk_memory_used = []
        self.cpu_memory_used = []
        self.cpu_chunk_memory_used = []
        self.stage_convert_time = []
        self.chunk_life_cycle = {}

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()
        self._nested_level += 1

    def end(self):
        self._nested_level = max(0, self._nested_level - 1)
        if self._nested_level == 0:
            self.end_time = time.time()

    def started(self):
        return self._nested_level > 0

    def warmup_finish(self):
        if self.warmup_finish_time is None:
            self.warmup_finish_time = time.time()

    def state_dict(self):
        return {'start_time': self.start_time, 'end_time': self.end_time if self.end_time is not None else time.time(), 'warmup_finish_time': self.warmup_finish_time, 'gpu_memory_used': self.gpu_memory_used, 'gpu_chunk_memory_used': self.gpu_chunk_memory_used, 'cpu_memory_used': self.cpu_memory_used, 'cpu_chunk_memory_used': self.cpu_chunk_memory_used, 'stage_convert_time': self.stage_convert_time, 'chunk_life_cycle': self.chunk_life_cycle}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.state_dict(), f)


profiler = Profiler()


class RuntimeMemTracer(object):
    """Collecting memory statistics on CPU and GPU during training,
    to direct chunk moving.
    Glossary:
        Chunkable Memry: Memory can be used to store chunk.
    """

    def __init__(self, local_rank: int=0, config=None, with_mem_saving_comm: bool=False):
        self.local_rank = local_rank
        self.metronome = Metronome()
        self.gpu_chunk_available_mem = 0
        self.cpu_chunk_available_mem = 0
        self.gpu_chunk_used_mem = 0
        self.cpu_chunk_used_mem = 0
        self.cpu_chunk_used_mem_pinned = 0
        self.with_mem_saving_comm = with_mem_saving_comm
        if config is not None:
            self._overall_gpu_mem_ratio = config.get('overall_gpu_mem_ratio', 0.8)
            self._overall_cpu_mem_ratio = config.get('overall_cpu_mem_ratio', 0.8)
            self._margin_use_ratio = config.get('margin_use_ratio', 0.8)
            self.warmup_gpu_chunk_mem_ratio = config.get('warmup_gpu_chunk_mem_ratio', 0.1)
            self.use_fake_dist = config.get('use_fake_dist', False)
            self.with_static_partition = config.get('with_static_partition', False)
            self.use_async_mem_monitor = config.get('use_async_mem_monitor', False)
        else:
            self._overall_gpu_mem_ratio = 0.8
            self._overall_cpu_mem_ratio = 0.8
            self._margin_use_ratio = 0.8
            self.warmup_gpu_chunk_mem_ratio = 0.2
            self.use_fake_dist = False
            self.with_static_partition = False
            self.use_async_mem_monitor = True
        if self.use_async_mem_monitor:
            self.async_mem_monitor = AsyncMemoryMonitor()
        mem_info = get_memory_info()
        local_world_size = get_local_world_size()
        if self.use_fake_dist:
            self._overall_gpu_mem = torch.cuda.get_device_properties(0).total_memory * self._overall_gpu_mem_ratio / local_world_size
            self._overall_cpu_mem = mem_info.total * self._overall_cpu_mem_ratio / local_world_size
        else:
            self._overall_gpu_mem = torch.cuda.get_device_properties(self.local_rank).total_memory * self._overall_gpu_mem_ratio
            self._overall_cpu_mem = mem_info.total * self._overall_cpu_mem_ratio / local_world_size
        log_dist(f'Init Manager over all gpu mem {self._overall_gpu_mem / 1000000.0} MB, cpu mem {self._overall_cpu_mem / 1000000.0} MB')
        self.cpu_used_list = []
        self.cpu_chunk_used_list = []
        self.cpu_sys_used_list = []
        self.gpu_used_list = []
        self.gpu_chunk_used_list = []
        self.gpu_sys_used_list = []
        self._margin_chunk_num_for_gpu_adam = 0
        self._default_chunk_size = 0
        self.max_cpu_sys_used = 0

    def close_tracer(self):
        """
        Close memory tracer.
        """
        if self.use_async_mem_monitor:
            self.async_mem_monitor.finish()
        log_dist('**** Memory Tracer is closed! ****')

    def start_train(self, param_fp16_chunk_size, chunk_size):
        self._param_fp16_chunk_size = param_fp16_chunk_size
        self._default_chunk_size = chunk_size
        if self.use_async_mem_monitor:
            self.async_mem_monitor.start()
        log_dist('**** Memory Tracer is stared! ****')

    def update_margin_mem(self):
        """Update the number of GPU free chunks for optimizer."""
        if len(self.gpu_sys_used_list) == 0:
            logger.warning('No gpu info collected. Maybe there are no chunk based tensors.')
            max_gpu_sys_used = 0
        else:
            max_gpu_sys_used = max(self.gpu_sys_used_list)
        if len(self.cpu_sys_used_list) == 0:
            logger.warning('No gpu info collected. Maybe there are no chunk based tensors.')
            self.max_cpu_sys_used = 0
        else:
            self.max_cpu_sys_used = max(self.cpu_sys_used_list)
        margin_mem_size = self._overall_gpu_mem - max_gpu_sys_used - self._param_fp16_chunk_size
        self._margin_chunk_num_for_gpu_adam = margin_mem_size / (self._default_chunk_size * 12) * self._margin_use_ratio
        log_dist('--------------- GPU INFO AFTER BWD ----------------')
        log_dist(f'Max GPU System Mem (non-chunk) Used {max_gpu_sys_used / 1000000.0} MB')
        log_dist(f'Max CPU System Mem (non-chunk) Used {self.max_cpu_sys_used / 1000000.0} MB')
        log_dist(f'Param FP16 Chunk Size {self._param_fp16_chunk_size / 1000000.0} MB')
        log_dist(f'Margin Mem Size {margin_mem_size / 1000000.0} MB, available chunk num for Optimizer States {self._margin_chunk_num_for_gpu_adam}')
        log_dist('--------------- GPU INFO AFTER BWD ----------------')
        logger.debug(f'OVERALL GPU MEM {self._overall_gpu_mem / 1024 / 1024} MB')

    def reset_memory_stats(self):
        """
        Reset statistics collected from memory tracing.
        It is used in case of gradient overflow during warmup and
        the memory stats is incomplete.
        """
        if self.metronome.is_warmup():
            self.cpu_used_list = []
            self.cpu_chunk_used_list = []
            self.cpu_sys_used_list = []
            self.gpu_used_list = []
            self.gpu_chunk_used_list = []
            self.gpu_sys_used_list = []
        log_dist('Reset Memory Statistics')

    def get_margin_chunk_num_for_gpu_adam(self):
        return self._margin_chunk_num_for_gpu_adam

    def trace_memory(self):
        """Record the memory usage of the moment and increase moment counter."""
        if torch.distributed.is_initialized():
            rank = self.local_rank
        else:
            rank = 0
        gpu_device = torch.device(f'cuda:{rank}')
        cpu_device = torch.device('cpu:0')
        gpu_used = get_sys_memory_used(gpu_device)
        if profiler.started():
            timestamp = time.time()
            cur_mom = self.metronome.moment()
            profiler.gpu_memory_used.append((cur_mom, timestamp, gpu_used))
            profiler.gpu_chunk_memory_used.append((cur_mom, timestamp, self.gpu_chunk_used_mem))
            cpu_used = get_sys_memory_used(cpu_device)
            profiler.cpu_memory_used.append((cur_mom, timestamp, cpu_used))
            profiler.cpu_chunk_memory_used.append((cur_mom, timestamp, self.cpu_chunk_used_mem))
        if self.metronome.is_warmup():
            if self.use_async_mem_monitor:
                max_mem_period = self.async_mem_monitor.finish()
                gpu_used = max(max_mem_period, gpu_used)
                self.async_mem_monitor.start()
            self.gpu_used_list.append(gpu_used)
            self.gpu_chunk_used_list.append(self.gpu_chunk_used_mem)
            self.gpu_sys_used_list.append(gpu_used - self.gpu_chunk_used_mem)
            cpu_used = get_sys_memory_used(cpu_device)
            self.cpu_used_list.append(cpu_used)
            self.cpu_chunk_used_list.append(self.cpu_chunk_used_mem)
            self.cpu_sys_used_list.append(cpu_used - (self.cpu_chunk_used_mem - self.cpu_chunk_used_mem_pinned))
            cur_mom = self.metronome.moment()
            assert len(self.gpu_sys_used_list) - 1 == cur_mom, f'{len(self.gpu_sys_used_list) - 1} vs {cur_mom}'
        self.metronome.tiktac()

    def add(self, device_type: str, size_in_bytes: int, is_pinned: bool=False):
        if device_type == 'cpu':
            self.cpu_chunk_used_mem += size_in_bytes
            if is_pinned:
                self.cpu_chunk_used_mem_pinned += size_in_bytes
        elif device_type == 'cuda':
            self.gpu_chunk_used_mem += size_in_bytes
        else:
            raise f'device type {device_type} is not supported'

    def delete(self, device_type, size_in_bytes, is_pinned: bool=False):
        if device_type == 'cpu':
            self.cpu_chunk_used_mem -= size_in_bytes
            if is_pinned:
                self.cpu_chunk_used_mem_pinned -= size_in_bytes
        elif device_type == 'cuda':
            self.gpu_chunk_used_mem -= size_in_bytes
        else:
            raise f'device type {device_type} is not supported'

    def remaining_chunk_mem(self, device_type):
        """
        Return the remainig chunkable memory on device_type,
        which can be used to host chunks.
        """
        size = self.available_chunk_mem(device_type) - self.used_chunk_mem(device_type)
        logger.debug(f'remaining_chunk_mem on {device_type} {size / 1000000.0} MB on mement {self.metronome.moment()}')
        return size

    def used_chunk_mem(self, device_type):
        """
        Return the used chunkable memory on device_type.
        """
        if device_type == 'cpu':
            return self.cpu_chunk_used_mem
        elif device_type == 'cuda':
            return self.gpu_chunk_used_mem
        else:
            raise RuntimeError(f'used_chunk_mem {device_type}')

    def available_chunk_mem(self, device_type):
        """The amount of memory on device_type that can be used for chunks.
        A.k.a chunkale memory.
        This includes the used memory that has been allocated for chunks
        and the remaining memory.

            available_chunk_mem = remaining_chunk_mem + used_chunk_mem

        In warmup, the available chunk mem is part of GPU mem and all
        CPU mem.
        After warmup, it is the minimal value of available mem of the
        current moment and next moment.
        """
        is_training_start = self.metronome.training_stage() is not TrainingStage.UNSTART
        if not is_training_start:
            if device_type == 'cpu':
                return self._overall_cpu_mem
            elif device_type == 'cuda':
                return self._overall_gpu_mem
        is_warmup = self.metronome.is_warmup() or self.with_static_partition
        if is_warmup:
            if device_type == 'cpu':
                return self._overall_cpu_mem
            elif device_type == 'cuda':
                if self.metronome.training_stage() == TrainingStage.ADAM:
                    ava_mem = self._overall_gpu_mem - 4 * self._default_chunk_size * 4
                    logger.debug(f'GPU available_chunk_mem is {ava_mem / 1000000.0} MB')
                    return ava_mem
                else:
                    return self._overall_gpu_mem * self.warmup_gpu_chunk_mem_ratio
        if device_type == 'cpu':
            local_world_size = get_local_world_size()
            if self.metronome.training_stage() != TrainingStage.ADAM:
                return self._overall_cpu_mem - self.max_cpu_sys_used / local_world_size
            else:
                return self._overall_cpu_mem
        elif device_type == 'cuda':
            if self.with_mem_saving_comm:
                msc_factor = 1
            else:
                msc_factor = get_world_size()
            if self.metronome.training_stage() == TrainingStage.ADAM:
                return self._overall_gpu_mem - 4 * self._default_chunk_size * 4
            elif self.metronome.training_stage() == TrainingStage.FWD:
                next_mom = self.metronome.next_moment()
                cur_mom = self.metronome.moment()
                next_mom_ava_mem = self._overall_gpu_mem - self.gpu_sys_used_list[next_mom]
                cur_mom_ava_mem = self._overall_gpu_mem - self.gpu_sys_used_list[cur_mom]
                return min(next_mom_ava_mem, cur_mom_ava_mem) - msc_factor * 2 * self._default_chunk_size
            elif self.metronome.training_stage() == TrainingStage.BWD:
                next_mom = self.metronome.next_moment()
                cur_mom = self.metronome.moment()
                next_mom_ava_mem = self._overall_gpu_mem - self.gpu_sys_used_list[next_mom]
                cur_mom_ava_mem = self._overall_gpu_mem - self.gpu_sys_used_list[cur_mom]
                return min(next_mom_ava_mem, cur_mom_ava_mem) - msc_factor * 2 * self._default_chunk_size * msc_factor


def getsizeof(data_type: torch.dtype):
    if data_type == torch.float:
        return 4
    elif data_type == torch.half:
        return 2
    elif data_type == torch.int8:
        return 1
    elif data_type == torch.int16:
        return 2
    elif data_type == torch.int32:
        return 4
    elif data_type == torch.int64:
        return 8
    else:
        raise TypeError(f'getsizeof dose not support data type {data_type}')


class MemoryCache(object):

    def __init__(self, capacity, memtracer: RuntimeMemTracer):
        """ "
        A cache of chunk to avoid too much memory allocation and free.
        `capacity` chunks always stay in the GPU memory.
        If we have allocated a chunk on the target device, just reuse the cached one.
        Params:
            `capacity` : the capacity size of each type of tensor cache list.
        Returns:
            None or a `torch.Tensor`.
        """
        self._capacity = capacity
        self._cached_tensors = {}
        self._memtracer = memtracer

    def _new_mem(self, size, data_type, device_type, pin_memory):
        space_size = getsizeof(data_type) * size
        ret = torch.zeros(size, dtype=data_type, device=device_type, pin_memory=pin_memory)
        self._memtracer.add(device_type.type, space_size, pin_memory)
        return ret

    def pop_or_allocate(self, device_type: torch.device, size: int, data_type: torch.dtype, pin_memory: bool) ->torch.Tensor:
        """
        Return a tensor including `size` `device_type` elements on `device_type`.
        Delete the reference to the tenor in MemoryCache.
        Return:
            torch.Tensor
        """
        assert isinstance(device_type, torch.device), 'device_type must be type of torch.device'
        if (device_type, data_type) not in self._cached_tensors:
            return self._new_mem(size, data_type, device_type, pin_memory)
        tensors = self._cached_tensors[device_type, data_type]
        i = -1
        for i in range(len(tensors)):
            if tensors[i].numel() == size:
                break
        if i == -1:
            return self._new_mem(size, data_type, device_type, pin_memory)
        new_tensor_ref = tensors[i]
        tensors.pop(i)
        return new_tensor_ref

    def push(self, payload):
        """
        NOTE() must set payload to None outside of this function.
        Recycle a payload tensor.
        If the cache is fulled, delete the payload.
        Returns:
            success pushed or not.
        """
        device_type = payload.device
        data_type = payload.dtype
        if (device_type, data_type) not in self._cached_tensors and self._capacity > 0:
            self._cached_tensors[device_type, data_type] = [payload.zero_()]
        else:
            size = payload.numel()
            if len(self._cached_tensors[device_type, data_type]) == self._capacity:
                is_pinned_flag = payload.is_pinned()
                del payload
                space_size = getsizeof(data_type) * size
                self._memtracer.delete(device_type.type, space_size, is_pinned_flag)
            else:
                self._cached_tensors[device_type, data_type].append(payload.zero_())


class Chunk(object):

    def __init__(self, capacity: int, data_type: torch.dtype, chunk_id: int, memory_tracer: RuntimeMemTracer, memory_cache: Optional[MemoryCache], with_async_move: bool, local_rank: int=0, is_dummy: bool=False):
        """
        Chunk is the minimal unit of the data transfer.
        It is a contiguous memory for saving tensors.
        To remove a tensor, we only need to set the state of the tensor to `FREE`.

        Chunk does no know if we are doing distributed training or not.
        Every process will observe its own chunk instances.

        Args:
            capacity: int. The maximum number of elements in the chunk.
            data_type: :class:`torch.dtype`.
            chunk_id: int.
            local_rank: int.
            is_dummy: bool.
        """
        self.chunk_id = chunk_id
        self.capacity = capacity
        self.data_type = data_type
        self.local_rank = local_rank
        self._is_dummy = is_dummy
        self.memory_tracer = memory_tracer
        self._state_dict = {TensorState.COMPUTE: 0, TensorState.HOLD: 0, TensorState.HOLD_AFTER_FWD: 0, TensorState.HOLD_AFTER_BWD: 0, TensorState.FREE: 0}
        self.unused = 0
        self.payload = None
        self._time_profile = True
        self._pin_flag = False
        self.with_mem_cache = memory_cache is not None
        if self.with_mem_cache:
            self.memory_cache = memory_cache
        self.with_async_move = with_async_move
        if self.with_async_move:
            self.compute_finish_event = torch.cuda.Event()

    def is_dummy(self):
        return self._is_dummy

    def get_chunk_space(self):
        """Size of the chunk (Bytes)."""
        return getsizeof(self.data_type) * self.capacity

    def get_payload_space(self):
        """Size of the payload (Bytes)."""
        if self.payload is None:
            return 0
        else:
            return getsizeof(self.payload.dtype) * self.payload.numel()

    def pin(self):
        self._pin_flag = True

    def unpin(self):
        self._pin_flag = False

    def is_pin(self):
        return self._pin_flag

    def allocate_payload(self, device):
        """Allocate payload on device for the chunk.

        NOTE() This method does not check availability. Please check if
        there is enough room for the chunk.
        This function should be exception-safe.
        Args:
            device: :class:`torch.device`.
        """
        payload_numel = self.capacity
        if self._time_profile:
            global_timer.my_timer.start_profile(f'CHUNK_allocate_payload_{device.type}')
        if self.with_mem_cache:
            try:
                self.payload = self.memory_cache.pop_or_allocate(device, payload_numel, self.data_type, device.type == 'cpu')
            except RuntimeError:
                if self._time_profile:
                    global_timer.my_timer.finish_profile(f'CHUNK_allocate_payload_{device.type}')
                return False
        else:
            try:
                self.payload = torch.zeros(payload_numel, dtype=self.data_type, device=device, pin_memory=device.type == 'cpu')
                self.memory_tracer.add(device.type, self.get_payload_space(), self.payload.is_pinned())
            except RuntimeError:
                if self._time_profile:
                    global_timer.my_timer.finish_profile(f'CHUNK_allocate_payload_{device.type}')
                return False
        if profiler.started():
            profiler.chunk_life_cycle[self.chunk_id]['life_cycle'].append((time.time(), 'allocate', device))
        if self._time_profile:
            global_timer.my_timer.finish_profile(f'CHUNK_allocate_payload_{device.type}')
        return True

    def release_payload(self):
        """Release the payload."""
        if self.with_mem_cache:
            self.memory_cache.push(self.payload)
            self.payload = None
        else:
            self.memory_tracer.delete(self.get_device().type, self.get_payload_space(), self.payload.is_pinned())
            del self.payload
            self.payload = None
        if profiler.started():
            profiler.chunk_life_cycle[self.chunk_id]['life_cycle'].append((time.time(), 'release', None))

    def update_state(self, old_state, new_state):
        """Update the state counter of tensors of the chunk.

        Args:
            old_state: :class:`TensorState`.
            new_state: :class:`TensorState`.
        """
        self._state_dict[old_state] -= 1
        self._state_dict[new_state] += 1
        if self.with_async_move and old_state == TensorState.COMPUTE and self._state_dict[TensorState.COMPUTE] == 0:
            cuda_ctx = CUDAContext()
            self.compute_finish_event.record(cuda_ctx.compute_stream)

    def get_state(self):
        """
        When payload is None, the state is `RELEASED`,
        otherwise, state of the chunk is decided by its tensors.

        Returns:
            :class:`ChunkState`.
        """
        if self.payload is None:
            return ChunkState.RELEASED
        if self._state_dict[TensorState.COMPUTE] > 0:
            return ChunkState.COMPUTE
        elif self._state_dict[TensorState.HOLD] > 0:
            return ChunkState.HOLD
        elif self._state_dict[TensorState.HOLD_AFTER_FWD] > 0:
            return ChunkState.HOLD_AFTER_FWD
        elif self._state_dict[TensorState.HOLD_AFTER_BWD] > 0:
            return ChunkState.HOLD_AFTER_BWD
        else:
            return ChunkState.FREE

    def all_tensor_state(self, state):
        """If all tensors are in the state or `FREE`.

        Args:
            state: :class:`TensorState`.
        Return:
            bool.
        """
        for k, v in self._state_dict.items():
            if k != TensorState.FREE and k != state:
                if v != 0:
                    if k == TensorState.HOLD and v == self.unused:
                        continue
                    return False
        return True

    def set_unused(self):
        """
        After forward calculation, the tensors in `HOLD` state are the ones
        that are not used. Remember them for the release.
        NOTE() This function can only be called at the end of forward calculation.
        """
        self.unused = self._state_dict[TensorState.HOLD]

    def move(self, target_device: torch.device):
        """
        Move the chunk to `target_device`.
        """
        if self.with_async_move:
            cuda_ctx = CUDAContext()
            self.compute_finish_event.synchronize()
            with torch.cuda.stream(cuda_ctx.copy_stream):
                self.move_sync(target_device)
        else:
            self.move_sync(target_device)

    def move_sync(self, target_device: torch.device):
        """
        Move the chunk to `target_device` synchronizely.
        NOTE() Please check if the `target_device` has enough room before.

        Args:
            target_device: :class:`torch.device`.
        """
        if self.get_device() is None:
            logger.warning(f'chunk move payload None to {target_device}')
            return
        if self.get_device() == target_device:
            return
        if self._time_profile:
            if target_device.type == 'cuda':
                global_timer.my_timer.start_profile('chunk_cpu_gpu_move')
            else:
                global_timer.my_timer.start_profile('chunk_gpu_cpu_move')
        src_device = self.get_device()
        logger.debug(f'move chunk {self.chunk_id}, which has {self.payload.numel() / 1000000.0} M {self.payload.dtype} elements, from {src_device} to {target_device}, used mem {self.memory_tracer.used_chunk_mem(target_device.type) / 1000000.0} MB')
        if self.with_mem_cache:
            payload_numel = self.payload.numel()
            if target_device.type == 'cpu':
                pinned_payload_cpu = self.memory_cache.pop_or_allocate(target_device, payload_numel, self.payload.dtype, True)
                pinned_payload_cpu.reshape(self.payload.shape)
                pinned_payload_cpu.copy_(self.payload)
                self.memory_cache.push(self.payload)
                self.payload = pinned_payload_cpu
            elif target_device.type == 'cuda':
                self.payload = self.payload.pin_memory()
                cuda_tmp_payload = self.memory_cache.pop_or_allocate(target_device, payload_numel, self.payload.dtype, False)
                cuda_tmp_payload.reshape(self.payload.shape)
                cuda_tmp_payload.copy_(self.payload)
                self.memory_cache.push(self.payload)
                self.payload = cuda_tmp_payload
        else:
            if target_device.type == 'cpu':
                pinned_payload_cpu = torch.empty(self.payload.shape, dtype=self.payload.dtype, device='cpu:0', pin_memory=True)
                pinned_payload_cpu.copy_(self.payload)
                self.payload = pinned_payload_cpu
            elif target_device.type == 'cuda':
                self.payload = self.payload.pin_memory()
                self.payload = self.payload
            self.memory_tracer.delete(src_device.type, self.get_payload_space(), self.payload.is_pinned())
            self.memory_tracer.add(target_device.type, self.get_payload_space(), self.payload.is_pinned())
        if self._time_profile:
            if target_device.type == 'cuda':
                global_timer.my_timer.finish_profile('chunk_cpu_gpu_move')
                global_timer.data_move_cnter.update('chunk_cpu_gpu_move', self.get_payload_space())
            elif target_device.type == 'cpu':
                global_timer.my_timer.finish_profile('chunk_gpu_cpu_move')
                global_timer.data_move_cnter.update('chunk_gpu_cpu_move', self.get_payload_space())
        if profiler.started():
            if len(profiler.chunk_life_cycle[self.chunk_id]['life_cycle']) == 0:
                raise RuntimeError(f'Chunk {self.chunk_id} allocation time is not recorded. You may need to put profiler.start() before initialize_engine ')
            profiler.chunk_life_cycle[self.chunk_id]['life_cycle'].append((time.time(), 'move', target_device))

    def get_device(self):
        """Get device of the payload of chunk, return None if not allocated."""
        if self.payload is not None:
            return self.payload.device
        else:
            return None


class ChunkEvictionPolicyBase(ABC):

    def __init__(self, metronome: Metronome):
        self.chunk_access_dict = {}
        self.chunk_release_dict = {}
        self.metronome = metronome

    def trace_access(self, chunk_id, dev):
        """
        Trace access information of chunk_id.
        Only works for the warmup phase.
        args:
            chunk_id : the id of chunk
            dev : the device uses the chunk at the moment
        """
        if not self.metronome.is_warmup():
            return
        cur_mom = self.metronome.moment()
        if (chunk_id, dev) not in self.chunk_access_dict:
            self.chunk_access_dict[chunk_id, dev] = [cur_mom]
        else:
            self.chunk_access_dict[chunk_id, dev].append(cur_mom)
            self.chunk_access_dict[chunk_id, dev].sort()

    def trace_release(self, chunk_id, dev):
        """
        Trace release information of chunk_id.
        Only works for the warmup phase.
        args:
            chunk_id : the id of chunk
            dev : the device uses the chunk at the moment
        """
        if not self.metronome.is_warmup():
            return
        cur_mom = self.metronome.moment()
        if (chunk_id, dev) not in self.chunk_access_dict:
            self.chunk_release_dict[chunk_id, dev] = [cur_mom]
        else:
            self.chunk_release_dict[chunk_id, dev].append(cur_mom)
            self.chunk_release_dict[chunk_id, dev].sort()

    def _chunk_next_used_moment(self, chunk_id, dev):
        """
        The very next memonet chunk_id has to be placed on dev.
        """
        if self.metronome.is_warmup():
            return 0
        cur_mom = self.metronome.moment()
        total_mom = self.metronome._total_moment
        if (chunk_id, dev) not in self.chunk_access_dict:
            return 2 * total_mom
        access_mom_list = self.chunk_access_dict[chunk_id, dev]
        for mom in access_mom_list:
            if mom > cur_mom:
                return mom
        return total_mom + access_mom_list[0]

    @abstractmethod
    def derive_eviction_list(self, id_to_chunk_map, required_room, target_device):
        NotImplemented


class CommGroupInfo(object):

    def __init__(self, chunk_type, id):
        self.chunk_type = chunk_type
        self.id = id

    def __hash__(self):
        return hash((self.chunk_type, self.id))

    def __eq__(self, other):
        return (self.chunk_type, self.id) == (other.chunk_type, other.id)

    def __str__(self):
        return f'({self.chunk_type}, {self.id})'


class CommInfo(object):

    def __init__(self, chunk_type, group_id, offset):
        assert offset < get_world_size()
        self.group = CommGroupInfo(chunk_type=chunk_type, id=group_id)
        self.offset = offset

    @property
    def chunk_type(self):
        return self.group.chunk_type

    @property
    def group_id(self):
        return self.group.id

    def __str__(self):
        return f'({self.group}, {self.offset})'


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_param_registered(param) ->bool:
    assert isinstance(param, torch.nn.Parameter)
    return hasattr(param, 'ps_attr')


def empty_cpu_param():
    return torch.nn.Parameter(torch.tensor([], dtype=torch.float, device=torch.device('cpu:0')), requires_grad=False)


def get_real_data_tensor(param):
    if param.ps_attr.param_type == ParamType.TORCH_BASED:
        return param.data
    elif param.ps_attr.param_type == ParamType.CHUNK_BASED:
        return param.ps_attr.access_tensor(AccessType.DATA)
    else:
        raise RuntimeError


class PSTensor(object):
    global_id = 0

    def __init__(self):
        self.tensor = None
        self.id = PSTensor.global_id
        self.state = TensorState.FREE
        PSTensor.global_id += 1

    def __str__(self):
        return f'id: {self.id}, state: {self.state}, tensor: {self.tensor}'


def register_param(param, param_type, data_type, name=None):
    assert isinstance(param, torch.nn.Parameter)
    if not hasattr(param, 'ps_attr'):
        param.ps_attr = PSParameter(param, param_type, data_type, name)


def zero_param(p):
    return torch.nn.Parameter(torch.zeros_like(p, dtype=torch.float), requires_grad=False)


class LossScaler:
    """
    Class that manages a static loss scale.  This class is intended to interact with
    :class:`Fp16Optimizer`, and should not be directly manipulated by the user.
    Use of :class:`LossScaler` is enabled via the ``static_loss_scale`` argument to
    :class:`Fp16Optimizer`'s constructor.
    Args:
        scale (float, optional, default=1.0):  The loss scale.
    """

    def __init__(self, scale=1):
        self.cur_scale = scale

    def has_overflow(self, param):
        return False

    def _has_inf_or_nan(x):
        return False

    def update_scale(self, overflow):
        pass

    @property
    def loss_scale(self):
        return self.cur_scale

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)


class PatrickStarEngine(torch.nn.Module):
    """patrickStar engine for training."""

    def __init__(self, model, client, config):
        super(PatrickStarEngine, self).__init__()
        self.module = model
        self.module.train()
        self.client = client
        default_optim_config = {'type': 'Adam', 'params': {'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'use_hybrid_adam': True}}
        if config is not None:
            optim_config = config.get('optimizer', default_optim_config)
            optim_type = optim_config.get('type', default_optim_config['type'])
            if optim_type not in ['Adam', 'AdamW']:
                raise ValueError(f'Only support Adam and AdamW at the moment. Get optimizer type {optim_type}')
            optim_params = optim_config.get('params', default_optim_config['params'])
            for key, val in default_optim_config['params'].items():
                if key not in optim_params:
                    optim_params[key] = val
            if 'fp16' not in config:
                self.loss_scaler = None
            else:
                loss_scale_config = config['fp16']
                assert loss_scale_config['enabled'], 'Must enable fp16 training.'
                assert 'loss_scale' in loss_scale_config, 'Must have `loss_scale` field set.'
                loss_scale = loss_scale_config['loss_scale']
                if loss_scale == 0:
                    log_dist('Use DynamicLossScaler')
                    self.loss_scaler = DynamicLossScaler(init_scale=2 ** loss_scale_config.get('initial_scale_power', 16), scale_factor=loss_scale_config.get('hysteresis', 2), scale_window=loss_scale_config.get('loss_scale_window', 2000), min_scale=loss_scale_config.get('min_loss_scale', 1))
                else:
                    self.loss_scaler = LossScaler(loss_scale)
            if 'gradient_clipping' not in config:
                self.gradient_clipping = -1
            else:
                self.gradient_clipping = config['gradient_clipping']
        else:
            optim_type = default_optim_config['type']
            optim_params = default_optim_config['params']
            self.loss_scaler = None
            self.gradient_clipping = -1
        self._move_torch_parts_to_gpu(model)
        self.optimizer = FP16Adam(self.client, self.module.parameters(), loss_scaler=self.loss_scaler, gradient_clipping=self.gradient_clipping, lr=optim_params['lr'], betas=optim_params['betas'], eps=optim_params['eps'], weight_decay=optim_params['weight_decay'], use_adamw=optim_type == 'AdamW', use_hybrid_adam=optim_params['use_hybrid_adam'])
        self.client.init(self.module, self.optimizer)
        self.iteration_cnt_ = 0
        self.warmup_times = 1
        log_dist('PatrickStarEngine initialized.')

    def _move_torch_parts_to_gpu(self, model):
        for buffer in model.buffers():
            buffer.data = buffer.data

        def move_param_to_gpu(module):
            if module.__class__.__name__ == 'Embedding':
                return
            for param in module.parameters(recurse=False):
                if param.ps_attr.param_type == ParamType.TORCH_BASED:
                    param.data = param.data
            for submodule in module.children():
                move_param_to_gpu(submodule)
        move_param_to_gpu(model)

    def _reset_before_forward(self):
        self.client.mem_tracer.reset_memory_stats()
        self.client.mem_tracer.metronome.reset()
        for param_fp16 in self.client.chunk_based_param_fp16:
            param_fp16.ps_attr.fwd_used_cnt = 0
        for _, chunk in self.client.chunk_list.generate_chunk():
            chunk.unused = 0
        self.client.reset_visited_chunk()

    def _set_state_after_forward(self):
        """
        After forward calculation, we need to reset the state of
        tensors from HOLD_AFTER_FWD to HOLD. Otherwise, chunks may be
        released accidentally when using gradient checkpointing.
        """
        for chunk_id, chunk in self.client.chunk_list.generate_chunk():
            if chunk.get_state() == ChunkState.HOLD or chunk.get_state() == ChunkState.HOLD_AFTER_FWD:
                chunk.set_unused()
                self.client.set_all_tensors_state_in_chunk(chunk_id, TensorState.HOLD)

    def forward(self, *inputs, **kwargs):
        """Execute forward propagation
        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        if self.iteration_cnt_ == 0:
            self.client.set_warmup(True)
        if self.iteration_cnt_ == self.warmup_times:
            self.client.set_warmup(False)
            self.client.mem_tracer.close_tracer()
        global_timer.my_timer.start_profile('FWD')
        if profiler.started():
            profiler.stage_convert_time.append((time.time(), TrainingStage.FWD))
        self.client.set_training_phase(TrainingStage.FWD)
        self._reset_before_forward()
        loss = self.module(*inputs, **kwargs)
        self._set_state_after_forward()
        global_timer.my_timer.finish_profile('FWD')
        self.client.reset_visited_chunk()
        return loss

    def backward(self, loss):
        """Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
        """
        global_timer.my_timer.start_profile('BWD')
        if profiler.started():
            profiler.stage_convert_time.append((time.time(), TrainingStage.FWD))
        self.client.set_training_phase(TrainingStage.BWD)
        for param_fp16 in self.client.chunk_based_param_fp16:
            param_fp16.ps_attr.bwd_used_cnt = 0
        self.optimizer.zero_grad()
        if self.loss_scaler:
            self.loss_scaler.backward(loss)
        else:
            loss.backward()
        self.client.mem_tracer.update_margin_mem()
        self.iteration_cnt_ += 1
        global_timer.my_timer.finish_profile('BWD')

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return state_dict(self.module, self.client, destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=False):
        return load_state_dict(self.module, self.client, state_dict=state_dict, strict=strict)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BertAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, position_embedding_type=4, is_decoder=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, position_embedding_type=4, is_decoder=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TiledLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Tencent_PatrickStar(_paritybench_base):
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

