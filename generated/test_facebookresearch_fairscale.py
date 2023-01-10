import sys
_module = sys.modules[__name__]
del sys
benchmarks = _module
datasets = _module
mnist = _module
wikitext2_data = _module
benchmark_dataset = _module
benchmark_mevo = _module
experimental_async_approaches = _module
offload = _module
sync_batchnorm = _module
fsdp = _module
golden_configs = _module
lm_wikitext2 = _module
oss_mnist = _module
models = _module
transformer_lm = _module
moe = _module
oss = _module
pipe = _module
utils = _module
conf = _module
fairscale = _module
experimental = _module
nn = _module
ampnet_pipe = _module
ampnet = _module
pipe = _module
auto_shard = _module
data_parallel = _module
gossip = _module
distributed = _module
gossiper = _module
graph_manager = _module
mixing_manager = _module
cuda_metering = _module
helpers = _module
distributed_pipeline = _module
data = _module
graph = _module
loss = _module
partition_handler = _module
pipeline = _module
trace = _module
mevo = _module
offload = _module
sync_batchnorm = _module
optim = _module
dynamic_loss_scaler = _module
tooling = _module
layer_memory_tracker = _module
wgit = _module
cli = _module
pygit = _module
repo = _module
sha1_store = _module
signal_sparsity = _module
signal_sparsity_profiling = _module
version = _module
fair_dev = _module
common_paths = _module
testing = _module
golden_testing_data = _module
testing = _module
testing_memory = _module
internal = _module
containers = _module
object = _module
parallel = _module
params = _module
reduce_scatter_bucketer = _module
state_dict = _module
version = _module
nn = _module
checkpoint = _module
checkpoint_activations = _module
checkpoint_utils = _module
data_parallel = _module
fsdp_optim_utils = _module
fully_sharded_data_parallel = _module
sharded_ddp = _module
misc = _module
flatten_params_wrapper = _module
param_bucket = _module
model_parallel = _module
cross_entropy = _module
initialize = _module
layers = _module
mappings = _module
random = _module
utils = _module
moe_layer = _module
top2gate = _module
async_pipe = _module
async_pipeline = _module
async_schedule = _module
balance = _module
blockpartition = _module
profile = _module
batchnorm = _module
checkpoint = _module
copy = _module
dependency = _module
messages = _module
microbatch = _module
phony = _module
pipe = _module
pipeline = _module
rpc = _module
skip = _module
layout = _module
namespace = _module
portal = _module
skippable = _module
tracker = _module
stream = _module
types = _module
worker = _module
wrap = _module
auto_wrap = _module
optim = _module
adam = _module
adascale = _module
grad_scaler = _module
layerwise_gradient_scaler = _module
oss = _module
release_utils = _module
setup = _module
tests = _module
ampnet_pipe_process = _module
test_ampnet_pipe = _module
test_gossip = _module
test_auto_shard = _module
test_mevo = _module
test_multiprocess_pipe = _module
test_offload = _module
test_sync_batchnorm = _module
test_dynamic_loss_scaler = _module
test_layer_memory_tracker = _module
test_api = _module
test_cli = _module
test_pygit = _module
test_sha1_store = _module
test_signal_sparsity = _module
test_signal_sparsity_profiling = _module
test_checkpoint_activations = _module
test_checkpoint_activations_norm = _module
test_fsdp = _module
test_fsdp_apply = _module
test_fsdp_freezing_weights = _module
test_fsdp_fwd_fwd_bwd_bwd = _module
test_fsdp_grad_acc = _module
test_fsdp_hf_transformer_eval = _module
test_fsdp_input = _module
test_fsdp_memory = _module
test_fsdp_metadata = _module
test_fsdp_multiple_forward = _module
test_fsdp_multiple_forward_checkpoint = _module
test_fsdp_multiple_wrapping = _module
test_fsdp_optimizer_utils = _module
test_fsdp_overlap = _module
test_fsdp_pre_backward_hook = _module
test_fsdp_regnet = _module
test_fsdp_shared_weights = _module
test_fsdp_shared_weights_mevo = _module
test_fsdp_state_dict = _module
test_fsdp_summon_full_params = _module
test_fsdp_uneven = _module
test_fsdp_with_checkpoint_wrapper = _module
test_sharded_ddp_features = _module
test_sharded_ddp_pytorch_parity = _module
test_flatten_params_wrapper = _module
test_grad_bucket = _module
test_param_bucket = _module
test_cross_entropy = _module
test_initialize = _module
test_layers = _module
test_random = _module
test_moe_layer = _module
test_top2gating = _module
conftest = _module
test_api = _module
test_gpipe = _module
test_inspect_skip_layout = _module
test_leak = _module
test_portal = _module
test_stash_pop = _module
test_tracker = _module
test_verify_skippables = _module
test_balance = _module
test_bugs = _module
test_checkpoint = _module
test_checkpoint_ddp = _module
test_copy = _module
test_deferred_batch_norm = _module
test_dependency = _module
test_inplace = _module
test_microbatch = _module
test_parity = _module
test_phony = _module
test_pipe = _module
test_pipeline = _module
test_stream = _module
test_transparency = _module
test_worker = _module
pipe_process = _module
conftest = _module
test_bugs = _module
test_inplace = _module
test_pipe = _module
test_rpc = _module
test_transparency = _module
test_wrap = _module
test_adam = _module
test_ddp_adascale = _module
test_layerwise_gradient_scaler = _module
test_oss = _module
test_oss_adascale = _module
test_single_node_adascale = _module
test_containers = _module
test_parallel = _module
test_reduce_scatter_bucketer = _module
test_state_dict = _module
test_version = _module

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


from collections import namedtuple


import torch


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


import torchtext


from torchtext.data.utils import get_tokenizer


from torchtext.utils import download_from_url


from torchtext.utils import extract_archive


from torch.utils.data import Dataset


import time


from torch import nn


from torch.cuda import Event


import logging


import math


import warnings


from torch.distributed import rpc


import torch.multiprocessing as mp


import torch.nn as nn


from torch.optim.optimizer import Optimizer


from functools import reduce


import numpy as np


from torch.optim import Adam


from torch.utils.data.dataloader import DataLoader


from torchvision.datasets import FakeData


from torchvision.transforms import ToTensor


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from collections import defaultdict


from enum import Enum


from typing import Any


from typing import List


from typing import Optional


from typing import cast


import torch.autograd.profiler as profiler


from torch.cuda.amp import GradScaler as TorchGradScaler


from torch.utils.data import BatchSampler


from torch.utils.data import Sampler


from torchvision.datasets import MNIST


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from typing import Dict


from typing import Tuple


from typing import Union


from torch.autograd.profiler import record_function


from torch.distributed import ProcessGroup


from typing import Set


import torch.fx


from torch.fx.node import Node


import functools


from typing import Callable


from typing import Iterable


from torch.autograd import Variable


from torch.nn.modules import Module


from typing import Iterator


from abc import ABC


from abc import abstractmethod


from math import log as mlog


from collections import deque


from functools import partial


from typing import ClassVar


from typing import Deque


import collections


from typing import MutableMapping


from torch import Tensor


from torch.distributed.nn import RemoteModule


from types import TracebackType


from typing import Type


import inspect


import torch.nn.functional as F


from enum import auto


from functools import lru_cache


from typing import NamedTuple


from typing import Sequence


from torch.utils.hooks import RemovableHandle


import copy


import random


from typing import TYPE_CHECKING


from typing import Generator


import numpy


from collections import OrderedDict


from torch.nn.utils.rnn import PackedSequence


import collections.abc as abc


from math import inf


import re


import torch.utils.checkpoint as torch_checkpoint


from torch.nn.modules.batchnorm import _BatchNorm


from itertools import groupby


import typing


from typing import Mapping


from torch.nn.parameter import Parameter


from itertools import chain


import torch.nn.init as init


from torch.cuda import _lazy_call


from torch.utils.checkpoint import detach_variable


from torch.nn import Module


from torch.nn import ModuleList


import itertools


from typing import TypeVar


from torch import ByteTensor


import torch.autograd


from queue import Empty as QueueEmpty


from queue import Queue


import torch.cuda.comm


import torch.cuda


from torch.distributed.distributed_c10d import _get_global_rank


from typing import FrozenSet


from torch.optim import SGD


from torch.optim import Optimizer


from torch.cuda import FloatTensor


from torch.cuda.amp.common import amp_definitely_not_available


from torch.cuda.amp.grad_scaler import GradScaler as TorchGradScaler


from torch.optim.sgd import SGD


from torch.autograd import profiler


from torch.nn import Parameter


import torch.distributed


import torch.nn


import torch.distributed.autograd as dist_autograd


from torch.distributed.optim import DistributedOptimizer


import torch.distributed.rpc as rpc


import torch.optim as optim


from torch.nn.parallel import DistributedDataParallel


from torch.utils.checkpoint import checkpoint as torch_checkpoint_wrapper


from torch.nn import BatchNorm2d


from torch.nn import LayerNorm


from torch.nn import Linear


from torch.nn import Sequential


from itertools import product


from time import time


from torch.optim import Adadelta


from torch.cuda.amp import GradScaler


from torch.nn import AdaptiveAvgPool2d


from torch.nn import Conv2d


from torch.nn import CrossEntropyLoss


from torch.nn import ReLU


from torch.nn import Sigmoid


from torch.nn import SyncBatchNorm


from copy import deepcopy


from torch.utils.checkpoint import checkpoint as torch_checkpoint


from torch import optim


from sklearn.datasets import make_blobs


from torch.cuda.amp.autocast_mode import autocast


import torchvision


import torchvision.transforms as transforms


from torch.optim.lr_scheduler import LambdaLR


class EmbeddingLayer(nn.Embedding):
    """Wrapped nn.Embedding layer to allow for weight initialization."""

    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp_sqrt = math.sqrt(ninp)
        self.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        return super().forward(src) * self.ninp_sqrt


class PositionalEncodingLayer(nn.Module):
    """PositionalEncoding layer for a given Transformer model."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FeedForwardLayer(nn.Module):
    """FeedForward layer for a given Transformer model."""

    def __init__(self, d_model, dim_feedforward, activation, dropout) ->None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))


class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) ->Tensor:
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) ->Tuple[None, Tensor]:
        return None, _AllToAll.apply(ctx.group, *grad_output)


def gumbel_rsample(shape: Tuple, device: torch.device) ->Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(tensor: torch.Tensor, num_classes: int) ->Tensor:
    """Workaround for https://github.com/pytorch/pytorch/issues/55579"""
    assert num_classes > 0, 'num_classes must be a positive integer'
    ret = torch.zeros(tensor.shape + (num_classes,), device=tensor.device, dtype=tensor.dtype)
    ret.scatter_(-1, tensor.unsqueeze(-1), 1)
    return ret


def top2gating(logits: torch.Tensor) ->Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    gates = F.softmax(logits, dim=1, dtype=torch.float)
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = 2 * num_tokens // num_experts
    assert num_tokens % num_experts == 0
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts)
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float('-inf'))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = one_hot(indices2_s, num_classes=num_experts)
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce)
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)
    denom_s = gates1_s + gates2_s
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    gates1 = gates1_s.unsqueeze(-1) * mask1
    gates2 = gates2_s.unsqueeze(-1) * mask2
    locations1_sc = one_hot(locations1_s, num_classes=capacity)
    locations2_sc = one_hot(locations2_s, num_classes=capacity)
    combine1_sec = gates1.unsqueeze(2) * locations1_sc.unsqueeze(1)
    combine2_sec = gates2.unsqueeze(2) * locations2_sc.unsqueeze(1)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    return l_aux, combine_weights, dispatch_mask


class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """
    wg: torch.nn.Linear

    def __init__(self, model_dim: int, num_experts: int) ->None:
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, input: torch.Tensor) ->Tuple[Tensor, Tensor, Tensor]:
        logits = self.wg(input)
        return top2gating(logits)


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
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
        is_moe: if ``True``, the feedforward layer will have MOE enabled.
        num_local_experts: number of local experts for MOE.


    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU(), layer_norm_eps=1e-05, norm_first=False, is_moe=False, num_local_experts=1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.is_moe = is_moe
        if is_moe:
            world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
            num_global_experts = num_local_experts * world_size
            self.gate = Top2Gate(d_model, num_global_experts)
            experts = nn.ModuleList([FeedForwardLayer(d_model, dim_feedforward, activation, dropout) for _ in range(num_local_experts)])
            self.moe_layer = MOELayer(self.gate, experts)
        else:
            self.ff_block = FeedForwardLayer(d_model, dim_feedforward, activation, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def _ff_block(self, x):
        if self.is_moe:
            return self.moe_layer(x)
        else:
            return self.ff_block(x)


class TransformerDecoderLayer(TransformerEncoderLayer):
    """TransformerDecoder layer which inherits from TransformerEncoderLayer."""

    def __init__(self, ninp, nhead, nhid, dropout, is_moe=False, num_local_experts=1):
        super().__init__(ninp, nhead, nhid, dropout, is_moe=is_moe, num_local_experts=num_local_experts)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask
        return super().forward(src, self.src_mask)


class LinearLayer(nn.Linear):
    """Wrapped nn.Linear layer to allow for weight initialization."""

    def __init__(self, ninp, ntoken, initrange):
        super().__init__(ninp, ntoken)
        self.bias.data.zero_()
        self.weight.data.uniform_(-initrange, initrange)


class TransformerLMSequntial(nn.Sequential):
    """A small language model based on the design of GPT-2 using nn.Sequeitnal
    for compatability with Pipe"""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder):
        layers = [EmbeddingLayer(ntokens, ninp, initrange), PositionalEncodingLayer(ninp, dropout)]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout))
        layers.append(LinearLayer(ninp, ntokens, initrange))
        super(TransformerLMSequntial, self).__init__(*layers)


class TransformerLM(nn.Sequential):
    """A GPT-2 based nn.Sequential language model."""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder, is_moe=False, num_local_experts=1):
        layers = [EmbeddingLayer(ntokens, ninp, initrange), PositionalEncodingLayer(ninp, dropout)]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout, is_moe, num_local_experts))
        layers.append(LinearLayer(ninp, ntokens, initrange))
        super(TransformerLM, self).__init__(*layers)


BROADCAST_BUCKET_SIZE = 10 * 1024 * 1024


class dist_backend(str, Enum):
    UNDEFINED = 'undefined'
    TCP = 'tcp'
    MPI = 'mpi'
    GLOO = 'gloo'
    NCCL = 'nccl'


HEARTBEAT_TIMEOUT = 300


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    Creates an adapter to make logging for multiple processes cleaner
    """

    def process(self, msg: str, kwargs: Any) ->Tuple[str, MutableMapping[str, Any]]:
        process_num = kwargs.pop('process_num', self.extra['process_num'])
        return f'process: {process_num} {msg}', kwargs


class SlowMoBaseAlgorithm(str, Enum):
    LOCALSGD = 'localsgd'
    SGP = 'sgp'


def flatten_tensors(tensors: List[torch.Tensor]) ->torch.Tensor:
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually
    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten
    Returns:
        A 1D buffer containing input tensors
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def group_by_dtype(tensors: List[torch.Tensor]) ->Dict[torch.dtype, List[torch.Tensor]]:
    """
    Returns a dict mapping from the tensor dtype to a list containing all
    tensors of that dtype.
    Arg:
        tensors (Iterable[Tensor]): list of tensors
    """
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype


def unflatten_tensors(flat: torch.Tensor, tensors: List[torch.Tensor]) ->List[torch.Tensor]:
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Args:
        flat (Tensor): flattened dense tensors to unflatten
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs


def communicate(tensors: List[torch.Tensor], communication_op: Any, logger: logging.Logger=None) ->None:
    """
    Communicate a list of tensors
    Args:
        tensors (Iterable[Tensor]): list of tensors
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce
    """
    tensors_by_dtype = group_by_dtype(tensors)
    for tensors_with_same_dtype in tensors_by_dtype.values():
        flat_tensor = flatten_tensors(tensors_with_same_dtype)
        if logger is not None:
            logger.debug('Flatten completed')
        communication_op(tensor=flat_tensor)
        if logger is not None:
            logger.debug('Commmunication completed')
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(flat_tensor, tensors_with_same_dtype), tensors_with_same_dtype):
                t.copy_(f)
        if logger is not None:
            logger.debug('Unflatten completed')


def create_and_record_event() ->torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


MAX_LEN_DEQUEUE = 10 ** 4


deque_with_max_len_fixed = partial(deque, maxlen=MAX_LEN_DEQUEUE)


def create_process_group(ranks: List[int]) ->torch.distributed.ProcessGroup:
    """
    Creates and intializes a new process group. Assumes init_process_group
    has already been called
    Arguments:
        ranks (list<int>): ranks corresponding to the processes which should
            belong the created process group
    Returns:
        New process group
    """
    new_group = dist.new_group(ranks=ranks)
    init_tensor_fp32, init_tensor_fp16 = torch.zeros(1), torch.zeros(1).half()
    for init_tensor in [init_tensor_fp32, init_tensor_fp16]:
        if torch.cuda.is_available():
            init_tensor = init_tensor
        if dist.get_rank() in ranks:
            dist.all_reduce(init_tensor, group=new_group)
        torch.cuda.synchronize()
    return new_group


def make_logger(rank: int, verbose: bool=True) ->logging.Logger:
    """
    Return a logger for writing to stdout
    Args:
        rank (int): rank of node making logger
        verbose (bool): whether to set log-level to INFO; o.w. WARNING
    Returns:
        Python logger
    """
    logger = logging.getLogger(__name__)
    if logger not in HANDLER_AND_LEVEL_SET:
        console = logging.StreamHandler(stream=sys.stdout)
        format_str = '{}'.format(rank)
        format_str += ': %(levelname)s -- %(threadName)s -- %(message)s'
        console.setFormatter(logging.Formatter(format_str))
        logger.addHandler(console)
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        HANDLER_AND_LEVEL_SET.add(logger)
    return logger


class MultiInputSequential(nn.Module):
    """A variation of nn.Sequential, that allows the first module in the sequence accepts
    multiple inputs. To be used internally by _split_module
    """

    def __init__(self, *modules: nn.Module) ->None:
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, *inputs: Tuple[Tensor]) ->Tensor:
        input = self.modules_list[0](*inputs)
        for module in self.modules_list[1:]:
            input = module(input)
        return input


ConsumerType = TypeVar('ConsumerType')


def RemoteSequential(rref_list: List[rpc.RRef]) ->MultiInputSequential:
    return MultiInputSequential(*(r.local_value() for r in rref_list))


Device = Union[torch.device, int, str]


Tensors = Tuple[Tensor, ...]


TensorOrTensors = Union[Tensor, Tensors]


class Batch:
    """An abstraction of an atomic tensor or a tuple of tensors. This
    eliminates every boilerplate code to classify an atomic tensor or a tuple
    of tensors.
    ::

        x = generate_tensor_or_tensors()
        x = Batch(x)

        # in-place update
        x[0] = F.apply(x[0])
        x[:] = F.apply(*x)

        # f(x) if x is a tensor.
        # f(*x) if x is a tuple of tensors.
        # y is also a batch.
        y = x.call(f)

    """

    def __init__(self, value: TensorOrTensors, index: int) ->None:
        self.value = value
        self.atomic = torch.is_tensor(value)
        self.__index = index

    @property
    def index(self) ->int:
        return self.__index

    @property
    def tensor(self) ->Tensor:
        """Retrieves the underlying tensor."""
        if not self.atomic:
            raise AttributeError('not atomic batch')
        return cast(Tensor, self.value)

    @property
    def tensors(self) ->Tensors:
        """Retrieves the underlying tensors."""
        if self.atomic:
            raise AttributeError('batch is atomic')
        return cast(Tensors, self.value)

    @property
    def tensor_or_tensors(self) ->TensorOrTensors:
        """Retrieves the underlying tensor or tensors regardless of type."""
        return self.value

    def call(self, function: Function) ->'Batch':
        """Calls a function by the underlying tensor or tensors. It also wraps
        the output with :class:`Batch`.
        """
        return Batch(function(self.value), self.index)

    def __repr__(self) ->str:
        return f'Batch[atomic={self.atomic!r}]({self.value!r})'

    def __iter__(self) ->Iterator[Tensor]:
        if self.atomic:
            yield self.tensor
        else:
            yield from self.tensors

    def __len__(self) ->int:
        return 1 if self.atomic else len(self.tensors)

    def __getitem__(self, index: int) ->Tensor:
        if not self.atomic:
            return self.tensors[index]
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        return self.tensor

    @typing.overload
    def __setitem__(self, index: int, value: Tensor) ->None:
        ...

    @typing.overload
    def __setitem__(self, index: slice, value: Tensors) ->None:
        ...

    def __setitem__(self, index: Union[int, slice], value: TensorOrTensors) ->None:
        if isinstance(index, int):
            value = cast(Tensor, value)
            self._setitem_by_index(index, value)
        else:
            value = cast(Tensors, value)
            self._setitem_by_slice(index, value)

    def _setitem_by_index(self, index: int, value: Tensor) ->None:
        if not self.atomic:
            i = index
            self.value = self.value[:i] + (value,) + self.value[i + 1:]
            return
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        self.value = value

    def _setitem_by_slice(self, index: slice, value: Tensors) ->None:
        if not index.start is index.stop is index.step is None:
            raise NotImplementedError('only slice [:] supported')
        if not self.atomic:
            self.value = value
            return
        if len(value) != 1:
            raise IndexError('atomic batch cannot be replaced with multiple tensors')
        self.value = value[0]


class CPUStreamType:
    pass


AbstractStream = Union[torch.Stream, CPUStreamType]


CPUStream = CPUStreamType()


def default_stream(device: torch.device) ->AbstractStream:
    """:func:`torch.cuda.default_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.default_stream(device)


def as_cuda(stream: AbstractStream) ->torch.Stream:
    """Casts the given stream as :class:`torch.cuda.Stream`."""
    return cast(torch.Stream, stream)


def is_cuda(stream: Optional[AbstractStream]) ->bool:
    """Returns ``True`` if the given stream is a valid CUDA stream."""
    return stream is not CPUStream


def get_phony(device: torch.device, *, requires_grad: bool) ->Tensor:
    """Gets a phony. Phony is tensor without space. It is useful to make
    arbitrary dependency in a autograd graph because it doesn't require any
    gradient accumulation.

    .. note::

        Phonies for each device are cached. If an autograd function gets a phony
        internally, the phony must be detached to be returned. Otherwise, the
        autograd engine will mutate the cached phony in-place::

            class Phonify(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    phony = get_phony(input.device, requires_grad=False)
                    return phony.detach()  # detach() is necessary.

    """
    key = device, requires_grad
    try:
        phony = _phonies[key]
    except KeyError:
        with use_stream(default_stream(device)):
            phony = torch.empty(1, device=device, requires_grad=requires_grad)
        _phonies[key] = phony
    return phony


class Fork(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Fork', input: Tensor) ->Tuple[Tensor, Tensor]:
        phony = get_phony(input.device, requires_grad=False)
        return input.detach(), phony.detach()

    @staticmethod
    def backward(ctx: 'Fork', grad_input: Tensor, grad_grad: Tensor) ->Tensor:
        return grad_input


def fork(input: Tensor) ->Tuple[Tensor, Tensor]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
    else:
        phony = get_phony(input.device, requires_grad=False)
    return input, phony


class Join(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Join', input: Tensor, phony: Tensor) ->Tensor:
        return input.detach()

    @staticmethod
    def backward(ctx: 'Join', grad_input: Tensor) ->Tuple[Tensor, None]:
        return grad_input, None


def join(input: Tensor, phony: Tensor) ->Tensor:
    """Merges two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)
    return input


MOVING_DENIED = TypeError('denied to move parameters and buffers, because Pipe should manage device placement')


def current_stream(device: torch.device) ->AbstractStream:
    """:func:`torch.cuda.current_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.current_stream(device)


def get_device(stream: AbstractStream) ->torch.device:
    """Gets the device from CPU or CUDA stream."""
    if is_cuda(stream):
        return as_cuda(stream).device
    return torch.device('cpu')


def record_stream(tensor: torch.Tensor, stream: AbstractStream) ->None:
    """:meth:`torch.Tensor.record_stream` for either CPU or CUDA stream."""
    if is_cuda(stream):
        tensor = tensor.new_empty([0]).set_(tensor.storage())
        tensor.record_stream(as_cuda(stream))


class Portal:
    """A portal for a tensor."""

    def __init__(self, tensor: Optional[Tensor], tensor_life: int, index: int) ->None:
        self.put_tensor(tensor, tensor_life)
        self.grad: Optional[Tensor] = None
        self.__index = index
        self.ns_name: Optional[Tuple[Namespace, str]]
        self.pipeline: Any

    @property
    def index(self) ->int:
        return self.__index

    def blue(self) ->Tensor:
        """Creates a :class:`PortalBlue` which hides the underlying tensor from
        the autograd engine.

        Join the returning phony to the main lane of the autograd graph to
        assure the correct backpropagation::

            PortalBlue --+
                         |
            ---------- Join --

        """
        tensor = self.use_tensor()
        if tensor is None:
            return get_phony(torch.device('cpu'), requires_grad=False)
        return PortalBlue.apply(self, tensor)

    def orange(self, phony: Tensor) ->Optional[Tensor]:
        """Creates a :class:`PortalOrange` which retrieves the hidden tensor
        without losing ability of backpropagation.

        Give a phony forked from the main lane of an autograd graph::

                +-- PortalOrange --+
                |                  |
            -- Fork --------- f(a, b) --

        """
        self.check_tensor_life()
        if self.tensor is None:
            return self.use_tensor()
        return PortalOrange.apply(self, phony)

    def copy(self, prev_stream: AbstractStream, next_stream: AbstractStream, phony: Tensor) ->Tensor:
        """Copies the hidden tensor by a :class:`PortalCopy`.

        Give a phony and use the returning phony to keep backpropagation::

                +-- PortalCopy --+
                |                |
            -- Fork ---------- Join --

        """
        if self.tensor is None:
            return get_phony(torch.device('cpu'), requires_grad=False)
        return PortalCopy.apply(self, prev_stream, next_stream, phony)

    def check_tensor_life(self) ->None:
        if self.tensor_life <= 0:
            raise RuntimeError('tensor in portal has been removed')

    def put_tensor(self, tensor: Optional[Tensor], tensor_life: int) ->None:
        """Stores a tensor into this portal."""
        self.tensor_life = tensor_life
        if tensor_life > 0:
            self.tensor = tensor
        else:
            self.tensor = None

    def use_tensor(self) ->Optional[Tensor]:
        """Retrieves the underlying tensor and decreases the tensor  life. When
        the life becomes 0, it the tensor will be removed.
        """
        self.check_tensor_life()
        tensor = self.tensor
        self.tensor_life -= 1
        if self.tensor_life <= 0:
            self.tensor = None
        return tensor

    def put_grad(self, grad: Tensor) ->None:
        """Stores a gradient into this portal."""
        if hasattr(self, 'pipeline'):
            self.pipeline.send_portal_grad(self.ns_name, self.index, grad)
        self.grad = grad

    def use_grad(self) ->Tensor:
        """Retrieves and removes the underlying gradient. The gradient is
        always ephemeral.
        """
        if self.grad is None and hasattr(self, 'pipeline'):
            self.grad = self.pipeline.recv_portal_grad(self.ns_name, self.index)
        if self.grad is None:
            raise RuntimeError('grad in portal has been removed or never set')
        grad = self.grad
        self.grad = None
        return grad


RNGStates = Tuple[ByteTensor, Optional[ByteTensor]]


Recomputed = Tuple[TensorOrTensors, Tensors]


def save_rng_states(device: torch.device, rng_states: Deque[RNGStates]) ->None:
    """:meth:`Checkpoint.forward` captures the current PyTorch's random number
    generator states at CPU and GPU to reuse in :meth:`Recompute.backward`.

    .. seealso:: :ref:`Referential Transparency`

    """
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state: Optional[ByteTensor]
    if device.type == 'cuda':
        gpu_rng_state = torch.get_rng_state(device)
    else:
        gpu_rng_state = None
    rng_states.clear()
    rng_states.append((cpu_rng_state, gpu_rng_state))


class Checkpointing:
    """Generates a pair of :class:`Checkpoint` and :class:`Recompute`."""

    def __init__(self, function: Function, batch: Batch) ->None:
        self.function = function
        self.batch = batch
        self.recomputed: Deque[Recomputed] = deque(maxlen=1)
        self.rng_states: Deque[RNGStates] = deque(maxlen=1)

    def checkpoint(self) ->Batch:
        """Returns a batch applied by :class:`Checkpoint`."""
        input_atomic = self.batch.atomic
        input = tuple(self.batch)
        phony = get_phony(self.batch[0].device, requires_grad=True)
        output = Checkpoint.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *input)
        if isinstance(output, tuple):
            output = tuple([(x if x.is_floating_point() else x.detach()) for x in output])
        return Batch(output, self.batch.index)

    def recompute(self, batch: Batch) ->None:
        """Applies :class:`Recompute` to the batch in place."""
        input_atomic = self.batch.atomic
        input = tuple(self.batch)
        batch[0], phony = fork(batch[0])
        phony = Recompute.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *input)
        batch[0] = join(batch[0], phony)


ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]


class Task:
    """A task represents how to compute a micro-batch on a partition.

    It consists of two parts: :meth:`compute` and :meth:`finalize`.
    :meth:`compute` should be executed in worker threads concurrently.
    :meth:`finalize` should be executed after when worker threads complete to
    execute :meth:`compute`.

    :meth:`compute` might be boosted by worker threads. Because it produces
    several CUDA API calls by user code. In PyTorch, parallel CUDA API calls
    are not serialized through GIL. So more than one CUDA API call can be
    produced at the same time.

    """

    def __init__(self, stream: Optional[AbstractStream], *, compute: Callable[[], Batch], finalize: Optional[Callable[[Batch], None]]) ->None:
        self.stream = stream
        self._compute = compute
        self._finalize = finalize
        self._grad_enabled = torch.is_grad_enabled()

    def compute(self) ->Batch:
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):
            return self._compute()

    def finalize(self, batch: Batch) ->None:
        if self._finalize is None:
            return
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):
            self._finalize(batch)


def torch_version(version: str=torch.__version__) ->Tuple[int, ...]:
    numbering = re.search('^(\\d+).(\\d+).(\\d+)([^\\+]*)(\\+\\S*)?$', version)
    if not numbering:
        return tuple()
    if numbering.group(4):
        logging.warning(f'Pytorch pre-release version {version} - assuming intent to test it')
    return tuple(int(numbering.group(n)) for n in range(1, 4))


def check_pytorch_version() ->None:
    if torch_version() < (1, 8, 0):
        raise Exception('DistributedPipeline requires PyTorch version 1.8 or higher')


def _reshape_inputs(input: torch.Tensor, target: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    """Convert 3D inputs to 2D for this kernel"""
    if len(input.shape) == 3:
        input = input.reshape(-1, input.shape[2])
    if len(target.shape) == 2:
        target = target.reshape(-1)
    return input, target


class BaselineSoftmax(nn.Module):
    """Baseline softmax that does an output linear projection and a softmax.


        We also support LMCL (Large Margin Cosine Loss) from the CosFace paper. See
        more detailed comment in the MEVO class below.

        This is intended to be used with an embedding layer with shared weights.

    Args:
        proj_weight (nn.Parameter):
            The shared weight.
        tile_factor (int):
            Unused. It is here to make kernel init easier with MEVO.
        log_softmax (bool):
            If True, use log_softmax instead of softmax.
        margin (float):
            Used in LMCL (when scale != None). See MEVO comments for
            more details.
        scale (Optional[float]):
            Used in LMCL. If scale is None, LMCL is turned off. See
            MEVO comments for more details.

    """

    def __init__(self, proj_weight: nn.Parameter, tile_factor: int=0, log_softmax: bool=True, margin: float=0.35, scale: Optional[float]=None):
        super().__init__()
        out_dim, in_dim = proj_weight.shape
        assert 'cuda' in str(proj_weight.device), 'weight should be on GPU'
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        assert proj_weight.dtype in [torch.float16, torch.float32]
        if proj_weight.dtype == torch.float16:
            self.fc = self.fc.half()
        self.fc.weight = proj_weight
        assert self.fc.weight.dtype in [torch.float16, torch.float32], self.fc.weight.dtype
        self.fp16 = self.fc.weight.dtype == torch.float16
        self.log_softmax = log_softmax
        self.margin = margin
        self.scale = scale

    def lmcl_pre_softmax(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        x = F.normalize(input, dim=1)
        w = F.normalize(self.fc.weight, dim=1)
        logits = torch.einsum('nc,kc->nk', x, w)
        row_ind = torch.arange(x.shape[0], dtype=torch.long)
        col_ind = target
        logits[row_ind, col_ind] -= self.margin
        logits *= self.scale
        return logits

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        """Forward function that computes softmax output with the input and target."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        if self.fp16:
            assert input.dtype == torch.float16
        if self.scale is not None:
            x = self.lmcl_pre_softmax(input, target)
        else:
            x = self.fc(input)
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1, dtype=torch.float32)
        else:
            x = F.softmax(x, dim=-1, dtype=torch.float32)
        assert x.dtype == torch.float32
        return x


class BaselineSoftmaxNllLoss(BaselineSoftmax):
    """Baseline that does an output projection, a softmax & a NLL loss (cross-entropy).

    See BaselineSoftmax above. Constructor is the same. Only difference is in the
    forward function.

    This class is used for testing and benchmarking.
    """

    def __init__(self, proj_weight: nn.Parameter, tile_factor: int=0, log_softmax: bool=True, margin: float=0.35, scale: Optional[float]=None):
        super().__init__(proj_weight, tile_factor, log_softmax, margin, scale)

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        """Forward that directly compute the loss."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        x = super().forward(input, target)
        return F.nll_loss(x, target, reduction='sum')


DEBUG = False


class BackwardTriggerFn(torch.autograd.Function):
    """A backward trigger function."""

    @staticmethod
    def forward(ctx: Any, w: torch.Tensor, trigger_tensor: torch.Tensor) ->torch.Tensor:
        """We take a weight tensor and the trigger as inputs and output the weight directly."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(w, trigger_tensor)
        return w

    @staticmethod
    def backward(ctx: Any, *args: Any) ->Any:
        """We return zero grad for the trigger only."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        w, trigger = ctx.saved_tensors
        assert w.requires_grad
        assert trigger.requires_grad
        return None, torch.zeros_like(trigger)


class BackwardTrigger(nn.Module):
    """A backward trigger module.

    This module takes a parameter as an input and create a linked parameter
    from a newly created trigger parameter.

    The way to use it in a module's ``__init__'' and ``forward'' functions:

    ```
    def __init__():
      ...
      self.trigger = BackwardTrigger(some_layer.weight)
      ...

    def forward():
      w = self.trigger()
      ... continue to use w ...
    ```

    As a resule, the trigger's backward hook will be called at the end of
    the backward for the module that uses this trigger.
    """

    def __init__(self, linked_param: torch.Tensor):
        super().__init__()
        assert isinstance(linked_param, nn.Parameter)
        self.trigger = nn.Parameter(torch.rand(1, dtype=linked_param.dtype, device=linked_param.device))
        self.trigger._linked_param = linked_param

    def forward(self) ->torch.Tensor:
        return BackwardTriggerFn.apply(self.trigger._linked_param, self.trigger)


def lmcl_matmul(i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, w_idx: int, margin: float, scale: Optional[float]) ->torch.Tensor:
    """LMCL variation of matmul with normalization, margin and scale."""
    logits = torch.matmul(F.normalize(i, dim=1), F.normalize(w, dim=1).T)
    mask = torch.arange(w_idx * w.shape[0], (w_idx + 1) * w.shape[0], dtype=torch.long, device=i.device).expand(i.shape[0], -1)
    logits[mask == tgt.reshape(-1, 1)] -= margin
    logits *= scale
    return logits


class GetMaxFunction(torch.autograd.Function):
    """Custom checkpointed function to get max-per-token from an input and a weight"""

    @staticmethod
    def get_max(i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, w_idx: int, full_precision: bool, margin: float, scale: Optional[float]) ->torch.Tensor:
        """
        Throughout this code:

          i: input data with shape = (split-of-tokens, d_model)
          w: weight data with shape = (split-of-vocabs, d_model)
          tgt: target prediction data with shape = (split-of-tokens,)
        """
        if scale is not None:
            _m = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _m = torch.matmul(i, w.T)
        if full_precision:
            _m = _m.float()
        _m = _m.max(dim=1)[0]
        return _m

    @staticmethod
    def forward(ctx: Any, i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, kernel_obj: 'MemoryEfficientVocabOutput', w_idx: int, w_split_size: int, split_dim: int) ->torch.Tensor:
        """Forward function that computes the max, without saving activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(i, w, tgt)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        ctx.args = {}
        assert split_dim == 0
        with torch.no_grad():
            return GetMaxFunction.get_max(i, w, tgt, w_idx, kernel_obj.fp_max, kernel_obj.margin, kernel_obj.scale)

    @staticmethod
    def backward(ctx: Any, *args: Any) ->Any:
        """Recompute the forward max and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        assert ctx.kernel_obj.proj_weight.grad is not None
        i, w, tgt = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            maxs = GetMaxFunction.get_max(i, w, tgt, ctx.w_idx, ctx.kernel_obj.fp_max, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(maxs, *args)
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, None, None, None, None, None


class GetSumFunction(torch.autograd.Function):
    """Custom checkpointed function to get sum-per-token from an input and a weight."""

    @staticmethod
    def get_sum(i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, maxs: torch.Tensor, w_idx: int, full_precision: bool, margin: float, scale: Optional[float]) ->torch.Tensor:
        if scale is not None:
            _s = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _s = torch.matmul(i, w.T)
        if full_precision:
            _s = _s.float()
        _s = (_s - maxs.reshape(-1, 1)).exp().sum(dim=1)
        return _s

    @staticmethod
    def forward(ctx: Any, i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, maxs: torch.Tensor, kernel_obj: 'MemoryEfficientVocabOutput', w_idx: int, w_split_size: int, split_dim: int) ->torch.Tensor:
        """Forward function that computes the sum, without saving activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(i, w, tgt, maxs)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        assert split_dim == 0
        with torch.no_grad():
            return GetSumFunction.get_sum(i, w, tgt, maxs, w_idx, kernel_obj.fp_sum, kernel_obj.margin, kernel_obj.scale)

    @staticmethod
    def backward(ctx: Any, *args: Any) ->Any:
        """Recompute the forward sum and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        assert ctx.kernel_obj.proj_weight.grad is not None
        i, w, tgt, maxs = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert maxs.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        maxs = maxs.detach().requires_grad_(True)
        with torch.enable_grad():
            sums = GetSumFunction.get_sum(i, w, tgt, maxs, ctx.w_idx, ctx.kernel_obj.fp_sum, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(sums, *args)
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, None, maxs.grad, None, None, None, None


class TargetScoreFunction(torch.autograd.Function):
    """Custom checkpointed function to compute the target score."""

    @staticmethod
    def get_target_score(i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, full_precision: bool, margin: float, scale: Optional[float]) ->torch.Tensor:
        tokens, d_model = i.shape
        assert d_model == w.shape[1]
        tw = w.gather(dim=0, index=target.reshape(target.shape[0], 1).expand(target.shape[0], d_model))
        assert tw.shape == (tokens, d_model)
        if scale is not None:
            target_score = F.normalize(i, dim=1) * F.normalize(tw, dim=1)
        else:
            target_score = i * tw
        if full_precision:
            target_score = target_score.float()
        target_score = target_score.sum(dim=1)
        if scale is not None:
            target_score -= margin
            target_score *= scale
        return target_score

    @staticmethod
    def forward(ctx: Any, i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, kernel_obj: 'MemoryEfficientVocabOutput') ->torch.Tensor:
        """Forward, without activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(i, w, target)
        ctx.kernel_obj = kernel_obj
        with torch.no_grad():
            x = TargetScoreFunction.get_target_score(i, w, target, kernel_obj.fp_target, kernel_obj.margin, kernel_obj.scale)
        return x

    @staticmethod
    def backward(ctx: Any, *args: Any) ->Any:
        """Forward and backward again, assign or accumulate the gradients."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        i, w, target = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert not target.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            scores = TargetScoreFunction.get_target_score(i, w, target, ctx.kernel_obj.fp_target, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(scores, *args)
        if ctx.kernel_obj.proj_weight.grad is not None:
            ctx.kernel_obj.proj_weight.grad.add_(w.grad)
        else:
            ctx.kernel_obj.proj_weight.grad = w.grad
        return i.grad, None, None, None


def _next_power_of_2_or_max(n: int, max_n: int) ->int:
    """Return the smallest power of 2 greater than or equal to n, with a limit.

    Useful when used in splitting a tensor into chunks with power-of-2 sizes.
    """
    if n == 0:
        return 1
    orig_n = n
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    assert n >= orig_n, f'{n} vs. {orig_n}'
    assert bin(n).count('1') == 1, bin(n)
    if n > max_n:
        return max_n
    return n


class MemoryEfficientVocabOutput(nn.Module):
    """Fused fc + softmax + nll_loss in a tiled fashion.

        MEVO uses much less memory but is quite a bit slower.

        MEVO also implements the LMCL (Large Margin Cosine Loss) function introduced by
        highly cited
        `CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`_.

        .. _`CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`: https://arxiv.org/abs/1801.09414

        LMCL can be turned on using the ``margin`` and ``scale`` parameters below. These
        hyperparameters most likely require tuning, depending on the number of classes etc.

        MEVO LMCL can be suitable for face recognition and image retrieval tasks, esp. when
        the number prediction target classes is large. MEVO is slower but can use much
        less GPU memory in that case, which enables training with larger batches. We
        hope this is helpful but we strongly recommend users (AI researchers
        and engineers) to carefully consider their applications of this technology. This
        types of technology should not be used by small group of people exclusively to
        potentially harm the general public.

    Args:
        proj_weight (nn.Parameter):
            Sharing this weight with an embedding layer.
        tile_factor (int):
            Number of splits to use on the input sequence and vocab dimensions.
            Default: 16
        reduction (str):
            Reduction OP (sum or mean).
            Default: sum
        margin (float):
            Hyperparameter of the separation margin between classes. See the
            appendix of the CosFace paper for a formula on how to compute its
            value properly. The default value is unlikely to be suitable in all
            cases.
            Default: 0.35
        scale (Optional[float]):
            Hyperparameter of the feature-vector-scaling for LMCL. When not
            supplied, LMCL is turned off. See the appendix of the CosFace paper for
            a formula on how to compute its value properly.
            Default: None
    """

    def __init__(self, proj_weight: nn.Parameter, tile_factor: int=16, reduction: str='sum', margin: float=0.35, scale: Optional[float]=None):
        super().__init__()
        self.proj_weight = proj_weight
        self.tf_in, self.tf_w = tile_factor, tile_factor
        self.fp_max = True
        self.fp_sum = True
        self.fp_target = True
        self.log_softmax = True
        self.reduction = reduction
        assert self.reduction in ['sum', 'mean']
        self.margin = margin
        self.scale = scale
        self.trigger = BackwardTrigger(self.proj_weight)
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None

    def get_target_nlprob(self, i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, debase_max: torch.Tensor, exp_sums: torch.Tensor) ->torch.Tensor:
        """Get target's negative log probability."""
        target_score = TargetScoreFunction.apply(i, w, target, self)
        prob = (target_score - debase_max).exp() / exp_sums
        if self.log_softmax:
            prob = prob.log()
        return -prob.sum()

    def eval_forward(self, input: torch.Tensor) ->torch.Tensor:
        """Eval time forward that doesn't fuse the softmax and NLL Loss kernels."""
        return torch.matmul(input, self.proj_weight.T)

    def forward(self, input: torch.Tensor, target: Optional[torch.Tensor]) ->torch.Tensor:
        if not self.training and target is None:
            return self.eval_forward(input)
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            cur_mem = round(torch.cuda.memory_allocated() / 1024 / 1024)
            mem = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
            None
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if torch.is_grad_enabled():
            assert input.requires_grad
        input, target = _reshape_inputs(input, target)
        tokens, d_model = input.shape
        t2, = target.shape
        vocab, d2 = self.proj_weight.shape
        assert d_model == d2, f'incorrect shape {d_model} vs {d2}'
        assert tokens == t2, f'incorrect shape {tokens} vs {t2}'
        split_dim = 0
        input_split_size = _next_power_of_2_or_max(tokens // self.tf_in, tokens)
        weight_split_size = _next_power_of_2_or_max(vocab // self.tf_w, vocab)
        inputs = torch.split(input, input_split_size, split_dim)
        weight = self.trigger()
        weights = torch.split(weight, weight_split_size, split_dim)
        targets = tuple([torch.Tensor()] * len(inputs))
        if self.scale is not None:
            targets = torch.split(target, input_split_size, split_dim)
        maxs = []
        for i, tgt in zip(inputs, targets):
            m = None
            for w_idx, w in enumerate(weights):
                _m = GetMaxFunction.apply(i, w, tgt, self, w_idx, weight_split_size, split_dim)
                if m is None:
                    m = _m
                else:
                    m = torch.max(m, _m)
            assert m is not None
            maxs.append(m)
        maxs_tensor = torch.cat(maxs)
        assert maxs_tensor.shape == (tokens,)
        sums = []
        for i, tgt, debase_max in zip(inputs, targets, maxs):
            s = None
            for w_idx, w in enumerate(weights):
                _s = GetSumFunction.apply(i, w, tgt, debase_max, self, w_idx, weight_split_size, split_dim)
                if s is None:
                    s = _s
                else:
                    s += _s
            assert s is not None
            sums.append(s)
        sums_tensor = torch.cat(sums)
        assert sums_tensor.shape == (tokens,)
        result = self.get_target_nlprob(input, self.proj_weight, target, maxs_tensor, sums_tensor)
        if self.reduction == 'mean':
            result /= tokens
        return result


class ModelShard(nn.Module):
    """
    Wrap one shard of the model, make it possible to load parameters on the
    fly for the FW and BW pass on the given device.
    """

    def __init__(self, cpu_model_shard: nn.Module, device: torch.device, offload_device: torch.device, index: int):
        super().__init__()
        self.model_shard = cpu_model_shard
        self.index = index
        self.device = device
        torch.device(self.device)
        self.offload_device = offload_device
        self.model_shard
        self._cpu_to_gpu_stream = torch.Stream(device=self.device)
        self._gpu_to_cpu_stream = torch.Stream(device=self.device)

    def forward(self, *inputs):
        return self.model_shard(*inputs) if isinstance(inputs, tuple) else self.model_shard(inputs)

    def to(self, device: torch.device) ->'ModelShard':
        self.model_shard
        return self

    def train(self, mode: bool=True) ->'ModelShard':
        self.model_shard.train(mode)
        return self

    def to_device(self) ->None:
        self.model_shard

    def forward_load(self, non_blocking: bool=True) ->None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard

    def backward_load(self, non_blocking: bool=True) ->None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard

    def forward_drop(self, non_blocking: bool=True) ->None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard

    def backward_drop(self, non_blocking: bool=True) ->None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard


def _conditional_amp_bwd_decorator(orig_func):
    if hasattr(torch.amp, 'custom_bwd'):
        return torch.amp.custom_bwd(orig_func)

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) ->Any:
        return orig_func(*args, **kwargs)
    return inner_decorator


def _conditional_amp_fwd_decorator(orig_func):
    if hasattr(torch.amp, 'custom_fwd'):
        return torch.amp.custom_fwd(orig_func)

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) ->Any:
        return orig_func(*args, **kwargs)
    return inner_decorator


class OffloadFunction(torch.autograd.Function):
    """
    This Function enables checkpointing of intermediate activations at
    shard boundaries by overriding the forward and backward pass of the nn.Module.

    - In the FW pass, it drops parameters in the previous shard and
    loads parameters for the next shard. No graph is constructed in the FW pass.
    This enables us to offload intermediate activations present at the shard
    boundaries.

    - In the BW pass, it does the reverse. We run the forward pass using the
    saved intermediate activations and calculate gradients as needed.
    The trade-off is latency vs memory when using activation checkpointing.

    - Follows heavily from https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html#checkpoint.

    NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    @_conditional_amp_fwd_decorator
    def forward(ctx: Any, inputs: Any, dummy_input: Any, model_instance: Any) ->Any:
        inputs = inputs if isinstance(inputs, tuple) else (inputs,)
        ctx.inputs = inputs
        ctx.model_instance = model_instance
        ctx.grad_requirements = tuple(x.requires_grad for x in inputs)
        ctx.fwd_rng_state = torch.get_rng_state()
        model_instance._activations = [inputs]
        for index, layer_shard in enumerate(model_instance.model_slices):
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:forward_load'):
                model_instance._activations[index] = tuple([a for a in list(model_instance._activations[index])])
                layer_shard.forward_load()
            inputs = model_instance._activations[index]
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:no_grad_forward_pass'):
                with torch.no_grad():
                    output_list: List[Any] = []
                    for given_input in inputs:
                        given_input_list = torch.chunk(given_input, model_instance._num_microbatches)
                        given_output_list = []
                        for inputs in given_input_list:
                            output = layer_shard(inputs)
                            given_output_list.append(output)
                        given_output = torch.cat(given_output_list).squeeze(-1)
                        output_list.append(given_output)
                    output = tuple(output_list)
            output = output if isinstance(output, tuple) else (output,)
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:forward_drop'):
                model_instance._activations[index] = tuple([a.cpu() for a in list(model_instance._activations[index])])
                model_instance._activations.append(output)
                layer_shard.forward_drop()
        result = model_instance._activations[-1]
        result = [r for r in result]
        for r in result:
            r.requires_grad = True
        return result[0] if len(result) == 1 else result

    @staticmethod
    @_conditional_amp_bwd_decorator
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('Checkpointing is not compatible with .grad(), please use .backward() if possible')
        inputs = ctx.inputs
        model_instance = ctx.model_instance
        for i, need_grad in enumerate(ctx.grad_requirements):
            inputs[i].requires_grad = need_grad
        all_grads = [grad_outputs]
        for model_shard, activation in zip(reversed(model_instance.model_slices), reversed(model_instance._activations[:-1])):
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:backward_load'):
                activation = tuple([a for a in list(activation)])
                model_shard.backward_load()
            bwd_rng_state = torch.get_rng_state()
            activation = torch.utils.checkpoint.detach_variable(activation)
            final_grads = all_grads[-1]
            if isinstance(activation, torch.Tensor):
                activation = activation,
            if isinstance(final_grads, torch.Tensor):
                final_grads = final_grads,
            chunked_grad_list: List[Any] = []
            for chunked_activation, chunked_grad in zip(torch.chunk(*activation, model_instance._num_microbatches), torch.chunk(*final_grads, model_instance._num_microbatches)):
                torch.set_rng_state(ctx.fwd_rng_state)
                if isinstance(chunked_activation, torch.Tensor):
                    chunked_activation = chunked_activation,
                if isinstance(chunked_grad, torch.Tensor):
                    chunked_grad = chunked_grad,
                for a in chunked_activation:
                    if a.dtype == torch.long:
                        continue
                    a.requires_grad = True
                    a.retain_grad()
                with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:forward_pass_with_enable_grad'):
                    with torch.enable_grad():
                        outputs = model_shard(*chunked_activation)
                torch.set_rng_state(bwd_rng_state)
                with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:backward_pass'):
                    torch.autograd.backward(outputs, chunked_grad)
                intermediate_grads = []
                for a in chunked_activation:
                    if a.grad is not None:
                        intermediate_grads.append(a.grad)
                if None not in intermediate_grads:
                    chunked_grad_list += intermediate_grads
            if chunked_grad_list:
                all_grads.append(torch.cat(chunked_grad_list).squeeze(-1))
            with torch.autograd.profiler.record_function('fairscale.experimental.nn.offload:backward_drop'):
                model_shard.backward_drop()
        detached_inputs = model_instance._activations[0]
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads


class ShardSyncLayer(torch.autograd.Function):
    """
    The shard sync layer is a synchronization point between model shards.
    - In the forward pass, it drops parameters in the previous shard and
    loads parameters for the next shard.
    - In the backward pass, it does the reverse.
    It does not change or create any outputs at all, instead it just
    forwards the input as the output.
    NOTE: see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    @_conditional_amp_fwd_decorator
    def forward(ctx: Any, inputs: Any, index: int, model_slices: Any, model_instance: Any) ->Any:
        drop_index = index
        load_index = index + 1
        max_slices = len(model_slices)
        if drop_index >= 0:
            model_slices[drop_index].forward_drop()
        if load_index < max_slices:
            model_slices[load_index].forward_load()
        ctx.index = index
        ctx.model_slices = model_slices
        ctx.model_instance = model_instance
        return inputs if isinstance(inputs, tuple) else (inputs,)

    @staticmethod
    @_conditional_amp_bwd_decorator
    def backward(ctx, *grad_outputs):
        load_index = ctx.index
        drop_index = load_index + 1
        model_slices = ctx.model_slices
        model_instance = ctx.model_instance
        if drop_index == len(model_slices):
            model_instance._activations[-1] = tuple([a for a in list(model_instance._activations[-1])])
        if drop_index < len(model_slices):
            model_slices[drop_index].backward_drop()
            model_instance._activations[drop_index] = tuple([a.cpu() for a in list(model_instance._activations[drop_index])])
        if load_index >= 0:
            model_slices[load_index].backward_load()
            model_instance._activations[load_index] = tuple([a for a in list(model_instance._activations[load_index])])
        if isinstance(grad_outputs, tuple):
            return grad_outputs[0], None, None, None
        return grad_outputs, None, None, None


_MODEL_PARALLEL_GROUP = None


def get_model_parallel_group() ->torch.distributed.ProcessGroup:
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def ensure_divisibility(numerator: int, denominator: int) ->None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) ->int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool=False) ->Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide_and_check_no_remainder(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _split(input_: torch.Tensor) ->torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()
    return output


class OffloadModel(nn.Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train by offloading majority of the model parameters to the CPU.
    `OffloadModel` is heavily inspired by the _L2L algorithm and _Zero-Offload.
    ::

        model = get_model()
        offload_model = OffloadModel(model, device,
                                    offload_device=torch.device(“cpu”),
                                    num_slices=3,
                                    checkpoint_activation=True,
                                    num_microbatches=5)

    .. _L2L: https://arxiv.org/abs/2002.05645
    .. _Zero-Offload: https://arxiv.org/abs/2101.06840

    At each step, a layer(or series of layers) are loaded
    onto the GPU for the forward and backward pass with intermediate
    activations being copied onto the GPU as required. Once the forward
    or backward pass is completed for a given shard, it is moved back to
    the CPU again.

    `OffloadModel` supports activation checkpointing which reduces
    the memory footprint. You can also increase the number of
    microbatches which translates to more computation cycles for
    every shard load. This helps offset the cost of moving the shard
    from the CPU to GPU and vice versa.

    Note: OffloadModel currently only supports nn.Sequential models.

    Args:
        module (~torch.nn.Sequential): Module to be offloaded.

        device (torch.device):
            Device where the active model should reside.

        offload_device (torch.device):
            Device where the inactive model should reside.

        num_slices (int):
            Number of slices into which the model should be chunked.

        checkpoint_activation (bool):
            Boolean to indicate if we want to checkpoint intermediate
            activation states on the CPU. Default value is False.

        num_microbatches (int):
            Number of microbatches which should be run per model
            shard on device.
    """

    def __init__(self, model: Any, device: torch.device, offload_device: torch.device=torch.device('cpu'), num_slices: int=3, checkpoint_activation: bool=False, num_microbatches: int=1):
        super().__init__()
        if not model:
            raise TypeError('`model` argument to `OffloadModel` cannot be None.')
        if not device:
            raise TypeError('`device` argument to `OffloadModel` cannot be None.')
        if not (isinstance(model, nn.Sequential) or type(model) == list):
            raise TypeError('`model` argument to `OffloadModel` must be of type `nn.Sequential`.')
        if not torch.cuda.is_available():
            raise TypeError('CUDA must be available as one of the compute devices for `OffloadModel`.')
        self.device = device
        self.offload_device = offload_device
        self.model_slices: List[nn.Module] = []
        if type(model) == list:
            for i, m in enumerate(model):
                self.model_slices.append(ModelShard(cpu_model_shard=m, device=device, offload_device=offload_device, index=i))
        else:
            splits = _split(model, num_slices)
            for i, split in enumerate(splits):
                self.model_slices.append(ModelShard(cpu_model_shard=nn.Sequential(*split), device=device, offload_device=offload_device, index=i))
        self._model = torch.nn.Sequential(*self.model_slices)
        self._activations: List[Tuple] = []
        if not checkpoint_activation and num_microbatches > 1:
            raise RuntimeError('We currently only support microbatches with activation checkpointing.')
        self._checkpoint_activation = checkpoint_activation
        self._num_microbatches = num_microbatches

    def forward(self, *inputs: Any, **_: Any) ->Any:
        if self._checkpoint_activation:
            return OffloadFunction.apply(*inputs, torch.tensor([], requires_grad=True), self)
        self._activations = []
        for index in range(-1, len(self.model_slices)):
            if index >= 0:
                self._activations[index] = tuple([a for a in list(self._activations[index])])
                inputs = self._activations[index]
                inputs = self.model_slices[index](*inputs)
            inputs = ShardSyncLayer.apply(inputs, index, self.model_slices, self)
            self._activations.append(inputs)
            if index >= 0:
                self._activations[index] = tuple([a.cpu() for a in list(self._activations[index])])
        result = self._activations[-1]
        result = tuple([r for r in result])
        return result[0] if len(result) == 1 else result


def _forward(input: Tensor, affine: bool, mean: Tensor, invstd: Tensor, weight: Tensor, bias: Tensor) ->Tensor:
    if affine:
        return (input - mean) * (invstd * weight.reshape_as(mean)) + bias.reshape_as(mean)
    else:
        return (input - mean) * invstd


class _SyncBatchNormFunction(torch.autograd.Function):
    """
    An autograd function used to avoid storing activations for intermediate results.

    NOTE: Even though the mean and var are passed into this function, we do the entire
    backward, including mean and var, here. We have to calculate statistics outside
    this function in order to avoid multiple all_reduces when using checkpointing.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, affine, mean, invstd, total_count, process_group):
        ctx.save_for_backward(input, weight, bias, mean, invstd, total_count)
        ctx.process_group = process_group
        return _forward(input, affine, mean, invstd, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        needs_input_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1]
        grad_input = None
        grad_weight = None
        grad_bias = None
        input, weight, bias, mean, invstd, total_count = ctx.saved_tensors
        process_group = ctx.process_group
        dim = [d for d in range(input.ndim) if d != 1]
        if needs_input_grad or needs_weight_grad:
            grad_common = torch.sum((input - mean) * grad_output, dim=dim, keepdim=True)
        if needs_input_grad:
            if weight is None:
                grad_input = invstd * grad_output
                grad_mean = -torch.sum(grad_input, dim=dim, keepdim=True)
                grad_invstd = grad_common
            else:
                grad_input = invstd * weight.reshape_as(mean) * grad_output
                grad_mean = -torch.sum(grad_input, dim=dim, keepdim=True)
                grad_invstd = grad_common * weight.reshape_as(mean)
            grad_var = -0.5 * invstd.pow(3) * grad_invstd
            grad_mean += -2 * mean * grad_var
            grad_meansqr = grad_var
            vec = torch.cat([grad_mean, grad_meansqr])
            all_reduce_handle = dist.all_reduce(vec, group=process_group, async_op=True)
        if needs_weight_grad:
            grad_weight = (grad_common * invstd).resize_as(weight)
            grad_bias = torch.sum(grad_output, dim=dim)
        if needs_input_grad:
            all_reduce_handle.wait()
            vec = vec / total_count
            grad_mean, grad_meansqr = vec.chunk(2)
            grad_input += grad_mean
            grad_input += input * (2 * grad_meansqr)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


def _calculate_stats(input: Tensor, eps: float, process_group: ProcessGroup) ->Tuple[Tensor, Tensor, Tensor, Tensor]:
    dim = [d for d in range(input.ndim) if d != 1]
    count = torch.full((1,), input.numel() // input.size(1), device=input.device, dtype=input.dtype)
    total_count = count.clone()
    all_reduce_handle = dist.all_reduce(total_count, group=process_group, async_op=True)
    mean = torch.mean(input, dim=dim, keepdim=True)
    meansqr = torch.mean(input * input, dim=dim, keepdim=True)
    vec = torch.cat([mean, meansqr])
    all_reduce_handle.wait()
    vec = vec * (count / total_count)
    dist.all_reduce(vec, group=process_group)
    mean, meansqr = vec.chunk(2)
    var = meansqr - mean * mean
    invstd = torch.rsqrt(var + eps)
    return mean, var, invstd, total_count


def _track_running_stats(running_mean: Tensor, running_var: Tensor, momentum: float, mean: Tensor, var: Tensor, total_count: Tensor) ->None:
    unbiased_var = var * (total_count / (total_count - 1))
    running_mean += momentum * (mean.reshape(-1) - running_mean)
    running_var += momentum * (unbiased_var.reshape(-1) - running_var)


def is_checkpointing() ->bool:
    """Whether the current forward propagation is under checkpointing.

    Returns:
        bool: :data:`True` if it's under checkpointing.

    """
    return thread_local.is_checkpointing


def is_recomputing() ->bool:
    """Whether the current forward propagation is under checkpoint
    recomputation. Use this to prevent duplicated side-effects at forward
    propagation::

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input

    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.

    .. seealso:: :ref:`Detecting Recomputation`

    """
    return thread_local.is_recomputing


class SyncBatchNorm(torch.nn.BatchNorm2d):
    """
    Fast re-implementation of ``torch.nn.SyncBatchNorm`` that can achieve a speedup
    of 5x or more over the default implementation depending on size of the input
    and number of distributed workers.
    """

    def __init__(self, *args: Tuple[Any, ...], process_group: Optional[ProcessGroup]=None, **kwargs: Dict[str, Any]) ->None:
        super().__init__(*args, **kwargs)
        self._process_group = process_group if process_group is not None else dist.group.WORLD
        self.saved_for_2nd_fwd: List[Tuple] = []
        self.disable_patch_batchnorm = True

    def forward(self, input: Tensor) ->Tensor:
        if not dist.is_initialized() or not self.training:
            return super().forward(input)
        wrapped = is_checkpointing() or is_recomputing()
        if not wrapped or is_checkpointing():
            with torch.no_grad():
                mean, var, invstd, total_count = _calculate_stats(input, self.eps, self._process_group)
                if self.track_running_stats:
                    _track_running_stats(self.running_mean, self.running_var, self.momentum, mean, var, total_count)
        if is_checkpointing():
            self.saved_for_2nd_fwd.append((mean, invstd, total_count))
            return _forward(input, self.affine, mean, invstd, self.weight, self.bias)
        if is_recomputing():
            mean, invstd, total_count = self.saved_for_2nd_fwd.pop(0)
        return _SyncBatchNormFunction.apply(input, self.weight, self.bias, self.affine, mean, invstd, total_count, self._process_group)

    @classmethod
    def convert_sync_batchnorm(cls, module: torch.nn.Module, process_group: Optional[ProcessGroup]=None) ->torch.nn.Module:
        """Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`fairscale.experimental.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = fairscale.experimental.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats, process_group=process_group)
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, 'qconfig'):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output


class FlatParameter(nn.Parameter):
    """A parameter that is initialized from a list of parameters and can be
    turned into a list of views as needed.
    """

    def __new__(cls, params: Sequence[nn.Parameter], requires_grad: bool=True) ->'FlatParameter':
        """Make an object using the parent's __new__ function."""
        if not isinstance(params, (list, tuple)) or len(params) == 0:
            raise ValueError('An non-empty list or tuple argument is needed')
        if not all(isinstance(p, (nn.Parameter, Tensor)) for p in params):
            raise ValueError('List items need to be Parameter types')
        if any(isinstance(p, FlatParameter) for p in params):
            raise ValueError('Nesting FlatParameter is not supported')
        data = torch.cat([(p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1)) for p in params], 0)
        return super(FlatParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, params: Sequence[nn.Parameter], requires_grad: bool=True):
        """Initialize the _param_numels and _param_shapes lists."""
        self._param_numels = [p.numel() for p in params]
        assert self.numel() <= sum(self._param_numels), f'Something wrong with __new__ method, {self.numel()} vs. {sum(self._param_numels)}'
        self._param_shapes = [p.size() for p in params]
        self._param_infos: List[Tuple[str, nn.Module, str]] = []
        self._shared_param_infos: List[Tuple[str, str, nn.Module, str, nn.Module, str]] = []

    def get_param_views(self, external_data: Optional[Tensor]=None) ->Iterator[Tensor]:
        """Return a generator of views that map to the original parameters."""
        assert self.data.numel() <= sum(self._param_numels), f'Incorrect internal state {self.data.numel()} vs. {sum(self._param_numels)}'
        data = external_data if external_data is not None else self
        if data.numel() != sum(self._param_numels):
            raise ValueError(f'Incorrect numel of supplied data: got {data.numel()} but expected {sum(self._param_numels)}')
        return (t.view(s) for t, s in zip(data.split(self._param_numels), self._param_shapes))

    def metadata(self) ->Tuple[List[str], List[torch.Size], List[int]]:
        """Return tuple of (names, shapes, numels) metadata for this flat parameter."""
        names = [('.'.join([m, n]) if m else n) for m, _, n in self._param_infos]
        return names, self._param_shapes, self._param_numels

    def __setstate__(self, state: Tuple[Any, Any, Any, Any]) ->None:
        """Use by pickle to set the internal states."""
        self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos = state
        assert self.numel() <= sum(self._param_numels), f'Incorrect pickling {self.numel()} vs. {sum(self._param_numels)}'

    def __reduce_ex__(self, proto: int) ->Tuple[Any, Any, Any]:
        """Support pickling between ranks."""
        return FlatParameter, ([self.data], self.requires_grad), (self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos)


ParamGroups = Optional[Union[List[List[nn.Parameter]], List[nn.Parameter]]]


def replace_by_prefix_(state_dict: Union[Dict[str, Tensor], 'OrderedDict[str, Tensor]'], old_prefix: str, new_prefix: str) ->None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError('old_prefix and new_prefix must be distinct')
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix):]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


def _post_state_dict_hook(module: nn.Module, state_dict: 'OrderedDict[str, Tensor]', prefix: str, *args: Any) ->'OrderedDict[str, Tensor]':
    replace_by_prefix_(state_dict, prefix + '_fpw_module.', prefix)
    return state_dict


_enable_pre_load_state_dict_hook = True


def _pre_load_state_dict_hook(state_dict: Union[Dict[str, Tensor], 'OrderedDict[str, Tensor]'], prefix: str, *args: Any) ->None:
    if not _enable_pre_load_state_dict_hook:
        return
    replace_by_prefix_(state_dict, prefix, prefix + '_fpw_module.')
    flat_param_key = prefix + '_fpw_module.flat_param'
    for k in list(state_dict.keys()):
        if k.startswith(flat_param_key):
            last_part = k.split('.')[-1]
            assert last_part.startswith('flat_param_'), last_part
            replace_by_prefix_(state_dict, k, prefix + last_part)


class ProcessGroupName(str, Enum):
    default = 'default'
    reduce_scatter = 'reduce_scatter'


class Bucket:
    """
    Helper class to simplify the handling of buckets, which unify the underlying storage of multiple tensors
    """

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device) ->None:
        self._params: List[torch.Tensor] = []
        self._param_ids: List[int] = []
        self._fill = 0
        self.buffer: torch.Tensor = torch.zeros(size, dtype=dtype, device=device)

    def to(self, device: Optional[Union[int, torch.device]], dtype: Optional[torch.dtype]=None, non_blocking: bool=False, keep_param_alignment: bool=True) ->'ParamBucket':
        """
        Move the underlying buffer
        """
        assert self.buffer is not None, 'Cannot move a collapsed bucket, please rebuild it'
        self.buffer = self.buffer


class ReduceScatterBucketer:
    """
    Helper for bucketing multiple reduce-scatter operations on small tensors
    into larger reduce-scatter ops to improve communication efficiency.

    Usage::

        bucketer = ReduceScatterBucketer()
        bucketer.reduce_scatter_async(
            small_tensors, callback_fn=lambda result: print("small")
        )
        bucketer.reduce_scatter_async(
            big_tensors, callback_fn=lambda result: print("big")
        )
        bucketer.reduce_scatter_async(
            more_small_tensors, callback_fn=lambda result: print("small2")
        )
        bucketer.flush()  # callbacks only guaranteed to be called after flush()
        # Example output (note that it is out of order, due to bucketing):
        # big
        # small
        # small2

    Args:
        bucket_cap_mb (int, Optional): bucket size for communicating. Buckets
            are sub-divided based on world_size. Values <= 0 disable bucketing.
    """

    def __init__(self, bucket_cap_mb: int=25):
        self.bucket_cap_mb = bucket_cap_mb
        self.buckets: Dict[Tuple[torch.dtype, torch.device, 'ProcessGroup'], Bucket] = {}

    @torch.no_grad()
    def reduce_scatter_async(self, input_list: List[Tensor], group: 'ProcessGroup', callback_fn: Optional[Callable]=None) ->None:
        """
        Reduce-scatter a list of tensors asynchronously, so smaller reductions
        can be bucketed together. The given callback (``callback_fn``) will be
        called with the reduced result at some later time. Call ``flush()`` to
        force all queued ops and callbacks to be executed.

        Note that large inputs will be reduced immediately, and this function
        may also flush the relevant bucket to make room for ``input_list``.

        Args:
            input_list (List[Tensor]): list of tensors to reduce-scatter. List
                should contain ``group.size()`` tensors and each tensor should
                have identical shape, dtype and device.
            group (ProcessGroup): process group for reduction
            callback_fn (Callable, Optional): callback function to call after
                the reduction executes. Function will be called with a single
                argument corresponding to the reduced result.
        """
        world_size = group.size()
        assert len(input_list) == world_size, f'reduce_scatter received {len(input_list)} inputs, expected group.size() ({world_size})'
        first_input = input_list[0]
        first_input_size = first_input.numel()
        bucket_shard_size = self._get_shard_size(first_input.element_size(), world_size)
        if first_input_size > bucket_shard_size:
            output = torch.zeros_like(input_list[0])
            if hasattr(dist, '_reduce_scatter_base') and enable_nccl_base_collectives:
                input_flattened = torch.cat(input_list)
                dist._reduce_scatter_base(output, input_flattened, group=group)
            else:
                dist.reduce_scatter(output, input_list, group=group)
            if callback_fn is not None:
                callback_fn(output)
            return
        bucket = self._get_bucket(first_input, group)
        if first_input_size > bucket.data.size(1) - bucket.offset:
            bucket.flush()
        stacked_input = torch.stack(input_list).view(world_size, first_input_size)
        offset = bucket.offset
        bucket.data[:, offset:offset + first_input_size].copy_(stacked_input)
        bucket.offset += first_input_size
        if callback_fn is not None:
            result_view = bucket.output_shard[offset:offset + first_input_size].view_as(first_input)
            bucket.callbacks.append(functools.partial(callback_fn, result_view))

    @torch.no_grad()
    def flush(self) ->None:
        """Reduce-scatter any partial buckets."""
        for bucket in self.buckets.values():
            bucket.flush()

    @torch.no_grad()
    def teardown(self) ->None:
        """Free buffers from all buckets."""
        for bucket in self.buckets.values():
            bucket.teardown()

    @functools.lru_cache()
    def _get_shard_size(self, element_size: int, num_shards: int) ->int:
        if self.bucket_cap_mb <= 0:
            return 0
        MB = 1024 * 1024
        bucket_size = self.bucket_cap_mb * MB / element_size
        return int(bucket_size // num_shards)

    def _get_bucket(self, tensor: Tensor, group: 'ProcessGroup') ->Bucket:
        key = tensor.dtype, tensor.device, group
        if key not in self.buckets:
            world_size = group.size()
            shard_size = self._get_shard_size(tensor.element_size(), world_size)
            data = tensor.new_zeros((world_size, shard_size))
            self.buckets[key] = Bucket(data, group)
        self.buckets[key].setup()
        return self.buckets[key]


class TrainingState(Enum):
    """
    Simple enum to indicate what state FSDP is in. Used for asserting
    to make sure APIs are called in the correct state.

    ..note::

        BACKWARD_PRE and BACKWARD_POST states are used to ensure we
        receives backward hooks in the correct order. It is used to catch
        unexpected order of hooks being called (likely due to our
        hook registration logic or autograd engine logic changes).

    TODO (Min): It would be nice to capture the stepping state as well.
        Maybe we can use the model.zero_grad() call, but not sure if it
        is called if optim.zero_grad() is used instead.
        It would be nice to have clear state transition be explicit like:

        zero_grad -> fwd -> bwd -> optionally accum grad by repeating
        fwd/bwd -> stepping -> loop back to zero_grad
    """
    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


def _clean_path(path: str) ->str:
    """Remove FSDP related wrapper modules from a given state dict key str path."""
    return '.'.join([split for split in path.split('.') if split not in {'_fsdp_wrapped_module', '_fpw_module'}])


def _get_default_cuda_device(module: nn.Module) ->torch.device:
    """Try to infer CUDA device from module parameters."""
    try:
        compute_device = next(module.parameters()).device
        if compute_device.type == 'cuda':
            return compute_device
    except StopIteration:
        pass
    return torch.device('cuda')


def _unpad(shard: torch.Tensor, pad: int) ->torch.Tensor:
    if pad > 0:
        shard = shard[:-pad]
    return shard


@torch.no_grad()
def alloc_storage_(data: torch.Tensor, size: torch.Size) ->None:
    """Allocate storage for a tensor."""
    if data.storage().size() == size.numel():
        return
    assert data.storage().size() == 0
    data.storage().resize_(size.numel())


def apply_to_type(type_fn: Callable, fn: Callable, container: Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set, NamedTuple]) ->Any:
    """Recursively apply to all objects in different kinds of container types that matches a type function."""

    def _apply(x: Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set]) ->Any:
        if type_fn(x):
            return fn(x)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = _apply(value)
            return od
        elif isinstance(x, PackedSequence):
            _apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            f = getattr(x, '_fields', None)
            if f is None:
                return tuple(_apply(x) for x in x)
            else:
                assert isinstance(f, tuple), 'This needs to be a namedtuple'
                x = cast(NamedTuple, x)
                _dict: Dict[str, Any] = x._asdict()
                _dict = {key: _apply(value) for key, value in _dict.items()}
                return type(x)(**_dict)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    return _apply(container)


def apply_to_tensors(fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set]) ->Any:
    """Recursively apply to all tensor in different kinds of container types."""
    return apply_to_type(torch.is_tensor, fn, container)


def calc_grad_norm(parameters: List[torch.nn.Parameter], p: float) ->torch.Tensor:
    """Calculate gradient norm of an iterable of parameters.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda par: par.grad is not None, parameters))
    if len(parameters) == 0:
        return torch.tensor(0.0)
    p = float(p)
    if p == inf:
        local_norm = max(par.grad.detach().abs().max() for par in parameters)
    else:
        local_norm = torch.norm(torch.stack([torch.norm(par.grad.detach(), p, dtype=torch.float32) for par in parameters]), p)
    return local_norm


def cast_floats_to_right_precision(to_fp16: bool, no_grad: bool, *args: Any, **kwargs: Any) ->Tuple[Any, Any]:
    """
    Cast floating point Tensors in *args or **kwargs to FP16 or FP32 if they are not.
    We also retain the requires_grad flag so that casting doesn't affect the autograd graph.
    """

    def fn_fp16(x: torch.Tensor) ->torch.Tensor:
        if x.dtype is torch.float32:
            y = x.half()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x

    def fn_fp32(x: torch.Tensor) ->torch.Tensor:
        if x.dtype is torch.float16:
            y = x.float()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x
    fn = fn_fp16 if to_fp16 else fn_fp32
    context = torch.no_grad() if no_grad else contextlib.suppress()
    with context:
        return apply_to_tensors(fn, args), apply_to_tensors(fn, kwargs)


def chunk_and_pad(tensor: torch.Tensor, num_chunks: int) ->List[torch.Tensor]:
    """Chunk a given Tensor into num_chunks parts and add any necessary padding."""
    chunks = list(torch.flatten(tensor).chunk(num_chunks))
    num_pad_for_partial_chunk = chunks[0].numel() - chunks[-1].numel()
    if num_pad_for_partial_chunk > 0:
        chunks[-1] = F.pad(chunks[-1], [0, num_pad_for_partial_chunk])
    if len(chunks) < num_chunks:
        chunks.extend([torch.zeros_like(chunks[0]) for _ in range(num_chunks - len(chunks))])
    return chunks


def enable_pytorch_sync_bn(module: torch.nn.Module) ->None:
    """Call _specify_ddp_gpu_num for all pytorch SyncBN layers so that it
    is happily running even without DDP. E.g. this is used by FSDP.
    """
    for layer in module.modules():
        if isinstance(layer, torch.nn.modules.SyncBatchNorm) and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)


def free_storage_(data: torch.Tensor) ->None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        assert data.storage_offset() == 0
        data.storage().resize_(0)


def get_process_group_cached(name: ProcessGroupName=ProcessGroupName.default, ranks: Optional[Sequence[int]]=None) ->'ProcessGroup':
    """
    Singleton PyTorch distributed group cache. Inspired by the code from fairseq.

    Just like torch.distributed.new_group, this method needs to be called on all ranks
    at the same time when a new group is created. This is true for all ranks irrespective
    of their group membership status.

    For FSDP, it is important to use the same group between outer and inner FSDP instances,
    otherwise, inner FSDP instances will not share the gradient reduction bucket buffer with
    the root instance. This will result in increased GPU memory utilization.

    Each separate process group also uses separate NCCL library instances, which will have
    a significant effect on GPU memory use if too many process groups are created and used.
    Setting NCCL_BUFFSIZE=102400 env variable is a useful technique to check if the NCCL
    memory is causing GPU OOM. Note, the NCCL buffers are not allocated
    through the PyTorch caching allocator, therefore, you may see GPU OOM even when
    torch.cuda.reserved_memory() is still way below the total amount of GPU memory.

    Extra process groups can also reduce training speed (observed on VISSL models).

    Args:
        name ProcessGroupName:
            There are two process groups when reduce_scatter overlap is enabled. The "default" process group is the
            default process group. The other group is "reduce_scatter" group.
            Default: ProcessGroupName.default
        ranks (Optional[List[int]]):
            Ranks requested in the target group. None for all ranks.
            Default: None

    Returns:
        (ProcessGroup):
            Return the requested process group. Throws RuntimeError if torch.distributed module is not yet initialized.
    """
    if not dist.is_initialized():
        if name == ProcessGroupName.reduce_scatter and 'pytest' in sys.modules:
            return None
        else:
            raise RuntimeError('torch.distributed is not yet initialized but process group is requested.')
    if not hasattr(get_process_group_cached, '_global_group_cache'):
        get_process_group_cached._global_group_cache = {}
        cache = get_process_group_cached._global_group_cache
        default_pg = dist.new_group(ranks=ranks)
        cache[None] = default_pg
        cache[ProcessGroupName.default, None] = default_pg
        cache[ProcessGroupName.default, frozenset(list(range(dist.get_world_size())))] = default_pg
    cache = get_process_group_cached._global_group_cache
    if ranks is not None:
        ranks = tuple(sorted(list(set(ranks))))
    if (name, ranks) not in cache:
        cache[name, ranks] = dist.new_group(ranks=ranks)
    return cache[name, ranks]


def p_assert(cond: Any, s: Any) ->None:
    """Used in backward context to make sure error is printed."""
    if not cond:
        None
        raise AssertionError


def recursive_copy_to_device(value: Any, non_blocking: bool, device: torch.device) ->Any:
    """
    Recursively searches lists, tuples, dicts and copies tensors to device if
    possible. Non-tensor values are passed as-is in the result.

    NOTE:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        values = []
        for val in value:
            values.append(recursive_copy_to_device(val, non_blocking=non_blocking, device=device))
        return values if isinstance(value, list) else tuple(values)
    if isinstance(value, abc.Mapping):
        device_val: Dict[str, Any] = {}
        for key, val in value.items():
            device_val[key] = recursive_copy_to_device(val, non_blocking=non_blocking, device=device)
        return device_val
    return value


def validate_process_group(device: torch.device, process_group: 'ProcessGroup') ->None:
    """Do a quick test in case user called FSDP without calling torch.cuda.set_device()
    correctly. This can easily happen in cpu_offload case where the model resides on
    the CPU.
    """
    if not hasattr(process_group, 'allgather'):
        return
    world_size = process_group.size()
    if 'cuda' in str(device):
        input_tensor = torch.ones(1)
        output = list(torch.zeros(world_size).chunk(world_size))
        dist.all_gather(output, input_tensor, group=process_group)
        assert torch.cat(output).sum() == float(world_size), f'found {torch.cat(output).sum()} devices in process group but world_size={world_size}. Check torch.cuda.set_device is called properly'


class GradBucket(Bucket):
    """
    Helper class to simplify the handling of gradient buckets
    """

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device, destination: int) ->None:
        super().__init__(size, dtype, device)
        self._max_size = size
        self._is_collapsed = False
        self.params_checked_in = 0
        self.destination = destination
        self.sent = True
        self.callback: Optional[Callable[[Any], None]] = None

    def reset_checked_in(self) ->None:
        """Reset the counter of the parameter grads which have been checked in"""
        self.params_checked_in = 0
        self.sent = False

    @property
    def all_checked_in(self) ->bool:
        """Have all the expected gradient check-in happened ?"""
        return len(self._params) == self.params_checked_in

    def can_add_grad_view(self, param: torch.Tensor) ->bool:
        """Is there enough room in the bucket to add this parameter gradient, and is this param not already checked in ?"""
        return self._fill + param.numel() < self._max_size and id(param) not in self._param_ids

    def to(self, device: Optional[Union[int, torch.device]], dtype: Optional[torch.dtype]=None, non_blocking: bool=False, keep_param_alignment: bool=True) ->'GradBucket':
        """
        Move the underlying buffer
        """
        if self._is_collapsed:
            self.rebuild()
        super()
        if keep_param_alignment:
            self._reattach_grads()

    def zero(self) ->None:
        """
        Set all the grads to zero
        """
        self.buffer.fill_(0.0)

    @torch.no_grad()
    def add_grad(self, param: torch.Tensor) ->None:
        """
        Add a new parameter gradient to the bucket. Param.grad becomes a view of this bucket buffer
        """
        assert id(param) not in self._param_ids, 'The same gradients cannot be checked in twice'
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        self._add_grad_as_view(param)
        self._params.append(param)
        self._param_ids.append(id(param))

    @torch.no_grad()
    def collapse(self) ->None:
        """
        Release the buffer from memory. The bucket will need to be rebuilt before use
        """
        if not self._is_collapsed:
            for p in self._params:
                assert p.grad is not None
                p.grad.detach_()
                p.grad = None
            self.buffer = torch.zeros(0, dtype=self.buffer.dtype, device=self.buffer.device)
            self._fill = 0
            self.params_checked_in = 0
            self._is_collapsed = True

    @torch.no_grad()
    def rebuild(self) ->None:
        """
        Given the parameter gradients which have been registered previously, rebuild the whole bucket
        """
        assert len(self._params) > 0
        if self._is_collapsed:
            self.buffer = torch.zeros(self._max_size, dtype=self._params[0].dtype, device=self._params[0].device)
            for p in self._params:
                self._add_grad_as_view(p)
            self._is_collapsed = False

    @torch.no_grad()
    def shrink(self) ->None:
        """
        Shrink the buffer to the size of the parameter gradients currently checked in, release the extra memory
        """
        assert self.buffer.numel() > 0, 'Cannot shrink a collapsed bucket, please rebuild'
        self.buffer = self.buffer.resize_(self._fill).clone()
        self._fill = 0
        for p in self._params:
            self._add_grad_as_view(p)
        self._max_size = self._fill

    @torch.no_grad()
    def _reattach_grads(self) ->None:
        """
        Given the parameters gradients which have been registered previously, rebuild the whole bucket
        """
        assert len(self._params) > 0
        self._fill = 0
        for p in self._params:
            self._add_grad_as_view(p, keep_existing_value=False)

    @torch.no_grad()
    def _add_grad_as_view(self, param: torch.Tensor, keep_existing_value: bool=True) ->None:
        assert self.buffer.numel() > 0, 'Cannot add a gradient to a collapsed bucket, please rebuild'
        assert param.dtype == self.buffer.dtype
        assert param.device == self.buffer.device
        fill_next = self._fill + param.numel()
        assert fill_next <= self.buffer.numel()
        if param.grad is not None:
            if keep_existing_value:
                self.buffer[self._fill:fill_next].copy_(param.grad.data.flatten())
            param.grad.data = self.buffer[self._fill:fill_next].view_as(param.data)
        else:
            param.grad = self.buffer[self._fill:fill_next].view_as(param.data)
        self._fill = fill_next


class ParamBucket(Bucket):
    """
    Helper class to simplify the handling of parameter buckets
    """

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device) ->None:
        super().__init__(size, dtype, device)

    def to(self, device: Optional[Union[int, torch.device]], dtype: Optional[torch.dtype]=None, non_blocking: bool=False, keep_param_alignment: bool=True) ->'ParamBucket':
        """
        Move the underlying buffer
        """
        super()
        if keep_param_alignment:
            self._reattach_params()

    @torch.no_grad()
    def add_param(self, param: torch.Tensor) ->None:
        """
        Add a new parameter gradient to the bucket. Param.grad becomes a view of this bucket buffer
        """
        assert id(param) not in self._param_ids, 'The same param cannot be checked in twice'
        self._add_param_as_view(param)
        self._params.append(param)
        self._param_ids.append(id(param))

    @torch.no_grad()
    def _add_param_as_view(self, param: torch.Tensor, keep_existing_value: bool=True) ->None:
        assert self.buffer is not None
        assert param.dtype == self.buffer.dtype, f'Different types for the bucket and the param, cannot proceed: {param.dtype} - {self.buffer.dtype}'
        assert param.device == self.buffer.device, f'Different devices for the bucket and the param, cannot proceed: {param.device} - {self.buffer.device}'
        fill_next = self._fill + param.numel()
        assert fill_next <= self.buffer.numel()
        if keep_existing_value:
            self.buffer[self._fill:fill_next].copy_(param.data.flatten())
        param.data = self.buffer[self._fill:fill_next].view_as(param.data)
        self._fill = fill_next

    @torch.no_grad()
    def _reattach_params(self) ->None:
        """
        Given the parameters which have been registered previously, rebuild the whole bucket
        """
        assert len(self._params) > 0
        self._fill = 0
        for p in self._params:
            if p.dtype != self.buffer.dtype:
                p.data = p.data
            self._add_param_as_view(p, keep_existing_value=False)


def _broadcast_object(obj: Any, src_rank: int, group: object=dist.group.WORLD, dist_device: torch.device=torch.device('cpu')) ->Any:
    """
    Either broadcast from master to the fleet (default),
    or use the src setting as the original rank.

    This is only needed for some older GPUs where dist.broadcast_object_list seems to hang. Also
    the hang behavior persist across processes once it happens. I.e. once we call dist.broadcast_object_list,
    subsequent calls with _broadcast_object also hang.
    """
    if dist.get_rank() == src_rank:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)])
        data_send_tensor = torch.ByteTensor(data)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        length_tensor = torch.LongTensor([0])
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty([int(length_tensor.item())], dtype=torch.uint8, device=dist_device)
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=dist_device)
    return obj


def _gpu_capabilities_older_than_50() ->bool:
    """Return True if the GPU's compute capability is older than SM50."""
    global _gpu_is_old
    if _gpu_is_old is None:
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(f'cuda:{i}')
            if major <= 5:
                _gpu_is_old = True
        if _gpu_is_old is None:
            _gpu_is_old = False
    return _gpu_is_old


def get_global_rank(group: Any, rank: int) ->int:
    if group is dist.group.WORLD:
        return rank
    return dist.distributed_c10d._get_global_rank(group, rank)


def _trainable(param: torch.Tensor) ->bool:
    return param.requires_grad


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [first, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int, rank: int, world_size: int) ->Tuple[int, int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) ->Tuple[int, int]:
        per_partition_vocab_size = divide_and_check_no_remainder(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)


def get_model_parallel_rank() ->int:
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_world_size() ->int:
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def _initialize_affine_weight(weight: torch.Tensor, out_features: int, in_features: int, per_partition_size: int, partition_dim: int, init_method: Callable[[torch.Tensor], torch.Tensor], stride: int=1, return_master_weight: bool=False) ->Optional[torch.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None
    master_weight = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)
    per_partition_per_stride_size = divide_and_check_no_remainder(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


def _reduce(ctx: Any, input_: torch.Tensor) ->torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()
    if ctx:
        ctx.mark_dirty(input_)
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    torch.distributed.all_reduce(input_, group=group)
    return input_


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return _reduce(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_model_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None, max_norm: Optional[float]=None, norm_type: float=2.0, scale_grad_by_freq: bool=False, sparse: bool=False, init_method: Callable[[torch.Tensor], torch.Tensor]=init.xavier_normal_) ->None:
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(self.num_embeddings, get_model_parallel_rank(), get_model_parallel_world_size())
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.weight = Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))
        _initialize_affine_weight(self.weight, self.num_embeddings, self.embedding_dim, self.num_embeddings_per_partition, 0, init_method)

    def forward(self, input_: torch.Tensor) ->torch.Tensor:
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        output_parallel[input_mask, :] = 0.0
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(None, grad_output)


def copy_to_model_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def _gather(input_: torch.Tensor) ->torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    group = get_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


def gather_from_model_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None, max_norm: Optional[float]=None, norm_type: float=2.0, scale_grad_by_freq: bool=False, sparse: bool=False, init_method: Callable[[torch.Tensor], torch.Tensor]=init.xavier_normal_, keep_master_weight_for_test: bool=False) ->None:
        super(ParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = scale_grad_by_freq
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        world_size = get_model_parallel_world_size()
        self.embedding_dim_per_partition = divide_and_check_no_remainder(self.embedding_dim, world_size)
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim_per_partition))
        _initialize_affine_weight(self.weight, self.num_embeddings, self.embedding_dim, self.embedding_dim_per_partition, 1, init_method, stride=1, return_master_weight=False)

    def forward(self, input_: torch.Tensor) ->torch.Tensor:
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.embedding(input_parallel, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        output = gather_from_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool=True, gather_output: bool=True, init_method: Callable[[torch.Tensor], torch.Tensor]=init.xavier_normal_, stride: int=1, keep_master_weight_for_test: bool=False) ->None:
        super(ColumnParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(self.weight, self.out_features, self.in_features, self.output_size_per_partition, 0, init_method, stride=stride, return_master_weight=keep_master_weight_for_test)

    def get_master_weight(self) ->torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data.transpose(0, 1)).transpose_(0, 1)

    def forward(self, input_: torch.Tensor) ->torch.Tensor:
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


def scatter_to_model_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_)


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool=True, input_is_parallel: bool=False, init_method: Callable[[torch.Tensor], torch.Tensor]=init.xavier_normal_, stride: int=1, keep_master_weight_for_test: bool=False):
        super(RowParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)
        self.weight = Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(self.weight, self.out_features, self.in_features, self.input_size_per_partition, 1, init_method, stride=stride, return_master_weight=keep_master_weight_for_test)

    def get_master_weight(self) ->torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data)

    def forward(self, input_: torch.Tensor) ->torch.Tensor:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight)
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output


Activations = Dict[int, Dict[int, Dict[int, Batch]]]


class AsyncMessageType(Enum):
    Activations = auto()
    Gradients = auto()


InputDevice = Union[None, int, str, torch.device]


_PIPELINE_PARALLEL_RANKS = None


def get_pipeline_parallel_ranks() ->List[int]:
    """Get the pipeline parallel group the caller rank belongs to."""
    assert _PIPELINE_PARALLEL_RANKS is not None, 'pipeline parallel group is not initialized'
    return _PIPELINE_PARALLEL_RANKS


class AutogradWithoutActivations(torch.autograd.Function):
    """A helper class to add another edge in the autograd graph which allows us
    to delete the potentially large activations and still perform a backward
    pass. Returns return a phony tensor which is connected to the graph."""

    @staticmethod
    def forward(ctx, *x):
        return torch.tensor(1.0)

    @staticmethod
    def backward(ctx, grad):
        assert ctx.grad_from_pipeline is not None
        return ctx.grad_from_pipeline


EVENT_LOOP_QUEUE = 3


def to_input_device(tensors: Tensors, input_device: InputDevice) ->Tensors:
    if input_device is None:
        return tensors
    else:
        return tuple(t for t in tensors)


MESSAGE_TENSOR_SIZE = 1024


_PIPELINE_PARALLEL_GROUP = None


def get_pipeline_parallel_group() ->torch.distributed.ProcessGroup:
    """Get the pipeline parallel group the caller rank belongs to."""
    assert _PIPELINE_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return _PIPELINE_PARALLEL_GROUP


def pyobject_to_tensor(obj: Any, fixed_buffer_size: int=0) ->torch.Tensor:
    pickled = pickle.dumps(obj)
    result: torch.Tensor = torch.ByteTensor(bytearray(pickled))
    if fixed_buffer_size:
        delta = fixed_buffer_size - len(result)
        if delta < 0:
            raise ValueError(f'message too big to send, increase `fixed_buffer_size`? - {len(result)} > {fixed_buffer_size}')
        elif delta > 0:
            result = torch.cat((result, torch.zeros(delta, dtype=torch.uint8)))
    return result


def tensor_to_pyobject(tensor: torch.Tensor) ->Any:
    nparray = tensor.cpu().numpy()
    return pickle.loads(nparray.tobytes())


TModule = TypeVar('TModule', bound=nn.Module)


class DeferredBatchNorm(_BatchNorm):
    """A BatchNorm layer tracks multiple micro-batches to update running
    statistics per mini-batch.
    """
    sum: Tensor
    sum_squares: Tensor

    def __init__(self, num_features: int, eps: float=1e-05, momentum: Optional[float]=0.1, affine: bool=True, chunks: int=1) ->None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats=True)
        self.register_buffer('sum', torch.zeros_like(self.running_mean))
        self.register_buffer('sum_squares', torch.zeros_like(self.running_var))
        self.counter = 0
        self.tracked = 0
        self.chunks = chunks

    def _check_input_dim(self, input: Tensor) ->None:
        if input.dim() <= 2:
            raise ValueError('expected at least 3D input (got %dD input)' % input.dim())

    def _track(self, input: Tensor) ->bool:
        """Tracks statistics of a micro-batch."""
        dim = [0]
        dim.extend(range(2, input.dim()))
        with torch.no_grad():
            self.sum += input.sum(dim)
            self.sum_squares += (input ** 2).sum(dim)
        size = input.size().numel() // input.size(1)
        self.counter += size
        self.tracked += 1
        return self.tracked == self.chunks

    def _commit(self) ->None:
        """Updates the running statistics of a mini-batch."""
        exponential_average_factor = 0.0
        self.num_batches_tracked += 1
        if self.momentum is None:
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:
            exponential_average_factor = self.momentum
        mean = self.sum / self.counter
        var = self.sum_squares / self.counter - mean ** 2
        m = exponential_average_factor
        self.running_mean *= 1 - m
        self.running_mean += mean * m
        self.running_var *= 1 - m
        self.running_var += var * m
        self.sum.zero_()
        self.sum_squares.zero_()
        self.counter = 0
        self.tracked = 0

    def forward(self, input: Tensor) ->Tensor:
        if not self.training:
            return F.batch_norm(input, running_mean=self.running_mean, running_var=self.running_var, weight=self.weight, bias=self.bias, training=False, momentum=0.0, eps=self.eps)
        if not is_recomputing():
            tracked_enough = self._track(input)
            if tracked_enough:
                self._commit()
        return F.batch_norm(input, running_mean=None, running_var=None, weight=self.weight, bias=self.bias, training=True, momentum=0.0, eps=self.eps)

    @classmethod
    def convert_deferred_batch_norm(cls, module: TModule, chunks: int=1) ->TModule:
        """Converts a :class:`nn.BatchNorm` or underlying
        :class:`nn.BatchNorm`s into :class:`DeferredBatchNorm`::

            from torchvision.models.resnet import resnet101
            from torchpipe.batchnorm import DeferredBatchNorm
            model = resnet101()
            model = DeferredBatchNorm.convert_deferred_batch_norm(model)

        """
        if isinstance(module, DeferredBatchNorm) and module.chunks is chunks:
            return cast(TModule, module)
        module_output: nn.Module = module
        if isinstance(module, _BatchNorm) and module.track_running_stats:
            module_output = DeferredBatchNorm(module.num_features, module.eps, module.momentum, module.affine, chunks)
            if module.affine:
                module_output.register_parameter('weight', module.weight)
                module_output.register_parameter('bias', module.bias)
            module_output.register_buffer('running_mean', module.running_mean)
            module_output.register_buffer('running_var', module.running_var)
            module_output.register_buffer('num_batches_tracked', module.num_batches_tracked)
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_deferred_batch_norm(child, chunks))
        return cast(TModule, module_output)


class LazyModule:

    def __init__(self, function: Callable[[], nn.Module]):
        self.function = function

    def __call__(self) ->nn.Module:
        return self.function()


T = TypeVar('T', bound='Skippable')


class pop:
    """The command to pop a skip tensor.

    ::

        def forward(self, input):
            skip = yield pop('name')
            return f(input) + skip

    Args:
        name (str): name of skip tensor

    Returns:
        the skip tensor previously stashed by another layer under the same name

    """
    __slots__ = 'name',

    def __init__(self, name: str) ->None:
        self.name = name


class stash:
    """The command to stash a skip tensor.

    ::

        def forward(self, input):
            yield stash('name', input)
            return f(input)

    Args:
        name (str): name of skip tensor
        input (torch.Tensor or None): tensor to pass to the skip connection

    """
    __slots__ = 'name', 'tensor'

    def __init__(self, name: str, tensor: Optional[Tensor]) ->None:
        self.name = name
        self.tensor = tensor


def check_balance(module: Union[nn.Sequential, List[LazyModule]], balance: List[int]) ->None:
    if len(module) != sum(balance):
        raise ValueError(f'module and sum of balance have different length (module: {len(module)}, sum of balance: {sum(balance)})')
    if any(x <= 0 for x in balance):
        raise ValueError(f'all balance numbers must be positive integer (balance: {balance})')


def verify_module(module: nn.Sequential) ->None:
    if not isinstance(module, nn.Sequential):
        raise TypeError('module must be nn.Sequential to be partitioned')
    named_children = list(module.named_children())
    if len(named_children) != len(module):
        raise ValueError('module with duplicate children is not supported')


class BalanceError(ValueError):
    pass


Devices = Union[Iterable[Device], List[Device]]


def clock_cycles(m: int, n: int) ->Iterable[List[Tuple[int, int]]]:
    """Generates schedules for each clock cycle."""
    for k in range(m + n - 1):
        yield [(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))]


def depend(fork_from: Batch, join_to: Batch) ->None:
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)


def wait_stream(source: AbstractStream, target: AbstractStream) ->None:
    """:meth:`torch.cuda.Stream.wait_stream` for either CPU or CUDA stream. It
    makes the source stream wait until the target stream completes work queued.
    """
    if is_cuda(target):
        if is_cuda(source):
            as_cuda(source).wait_stream(as_cuda(target))
        else:
            as_cuda(target).synchronize()


def wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) ->None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)
    batch[:] = tuple([(x if x.is_floating_point() else x.detach()) for x in batch])


def new_stream(device: torch.device) ->AbstractStream:
    """Creates a new stream for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.Stream(device)


def destroy_model_parallel() ->None:
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _PIPELINE_PARALLEL_GROUP
    _PIPELINE_PARALLEL_GROUP = None
    global _PIPELINE_PARALLEL_RANKS
    _PIPELINE_PARALLEL_RANKS = None


def get_world_sizes() ->List[int]:
    limit = torch.cuda.device_count()
    return [x for x in [1, 2, 4, 8] if x <= limit]


def initialize_model_parallel(model_parallel_size_: int, pipeline_length: int=1, *, model_parallel_backend: Optional[str]=None, pipeline_backend: Optional[str]=None, ddp_backend: Optional[str]=None) ->None:
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = int(min(model_parallel_size_, world_size))
    ensure_divisibility(world_size, model_parallel_size)
    ensure_divisibility(world_size, model_parallel_size * pipeline_length)
    rank = torch.distributed.get_rank()
    data_parallel_size = int(world_size / (model_parallel_size * pipeline_length))
    if torch.distributed.get_rank() == 0:
        None
        None
        None
    groups = torch.LongTensor(range(world_size)).reshape(data_parallel_size, pipeline_length, model_parallel_size)
    found = torch.where(groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    for j in range(pipeline_length):
        for k in range(model_parallel_size):
            group = torch.distributed.new_group(groups[:, j, k].tolist(), backend=ddp_backend)
            if j == found[1] and k == found[2]:
                _DATA_PARALLEL_GROUP = group
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for i in range(data_parallel_size):
        for j in range(pipeline_length):
            group = torch.distributed.new_group(groups[i, j, :].tolist(), backend=model_parallel_backend)
            if i == found[0] and j == found[1]:
                _MODEL_PARALLEL_GROUP = group
    global _PIPELINE_PARALLEL_GROUP
    assert _PIPELINE_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    global _PIPELINE_PARALLEL_RANKS
    assert _PIPELINE_PARALLEL_RANKS is None, 'model parallel group is already initialized'
    for i in range(data_parallel_size):
        for k in range(model_parallel_size):
            ranks = groups[i, :, k].tolist()
            group = torch.distributed.new_group(ranks, backend=pipeline_backend)
            if i == found[0] and k == found[2]:
                _PIPELINE_PARALLEL_GROUP = group
                _PIPELINE_PARALLEL_RANKS = ranks


def rmf(filename: str) ->None:
    """Remove a file like rm -f."""
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def make_cudnn_deterministic() ->None:
    """Make cudnn (matmul) deterministic"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def test_runner(rank: int, test_func: Callable, deterministic: bool=False, *args: List[Any], **kwargs: Dict[str, Any]) ->None:
    if deterministic:
        make_cudnn_deterministic()
        torch.manual_seed(1357)
    test_func(rank, *args, **kwargs)


def spawn_for_all_world_sizes(test_func: Callable, world_sizes: List[int]=get_world_sizes(), args: Any=[], deterministic: bool=False) ->None:
    for world_size in world_sizes:
        _, filename = tempfile.mkstemp()
        _, filename_rpc = tempfile.mkstemp()
        try:
            mp.spawn(test_runner, args=(test_func, deterministic, world_size, filename, filename_rpc, *args), nprocs=world_size, join=True)
        finally:
            rmf(filename)
            rmf(filename_rpc)


def teardown() ->None:
    destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    try:
        torch.distributed.rpc.shutdown(graceful=False)
    except Exception:
        pass


BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO


def dist_init(rank, world_size, tempfile_name, backend=BACKEND):
    url = 'file://' + tempfile_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)


def worker_process(rank: int, world_size: int, filename: str, filename_rpc: str, func: Callable, args: Any, error_queue: Any) ->None:
    """Main function for unit tests launched with torch_spawn"""
    if not dist_init(rank, world_size, filename, filename_rpc):
        logging.warning('failed initializing torch distributed')
        teardown()
        return
    kwargs = {}
    if 'OMPI_COMM_WORLD_RANK' not in os.environ:
        kwargs['pipeline_backend'] = 'gloo'
    initialize_model_parallel(1, world_size, **kwargs)
    context = torch.backends.cudnn.flags(benchmark=False, deterministic=True) if torch.cuda.is_available() and hasattr(torch.backends.cudnn, 'flags') else contextlib.suppress()
    if torch.cuda.is_available() and not hasattr(torch.backends.cudnn, 'flags'):
        make_cudnn_deterministic()
    try:
        with context:
            func(*args)
        teardown()
    except BaseException as e:
        logging.warning(f' Rank {rank}: {e}')
        teardown()
        if e.__class__.__name__ == 'Skipped':
            error_queue.put(str(e))
            return
        raise e


def torch_spawn(world_sizes: Optional[List[int]]=None) ->Callable:
    if world_sizes is None:
        world_sizes = get_world_sizes()

    def prepare_test(func: Callable) ->Callable:
        """Function called with the test function as the argument. Generates a
        replacement which serves as the actual test function."""
        name = func.__name__
        parameters = inspect.signature(func).parameters
        if name.startswith('test'):
            raise ValueError(f"Tests marked with @torch_spawn (i.e. '{name}') should not have names beginning in 'test' as they will be picked up by pytest without running the spawn wrapper")

        @functools.wraps(func)
        def replacement(*args: Any, **kwargs: Any) ->None:
            assert args == tuple()
            assert world_sizes is not None
            args = tuple(kwargs[p] for p in parameters if p != 'rank')
            error_queue = multiprocessing.get_context('spawn').SimpleQueue()
            if 'OMPI_COMM_WORLD_RANK' in os.environ:
                global filename_mpi
                if filename_mpi is None:
                    filename_mpi = tempfile.mkstemp()[1]
                os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
                os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
                torch.distributed.init_process_group('mpi', init_method=f'file://{filename_mpi}')
                world_size = torch.distributed.get_world_size()
                destroy_model_parallel()
                initialize_model_parallel(1, world_size)
                torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())
                if world_size in world_sizes:
                    try:
                        func(*args)
                        teardown()
                    except BaseException as e:
                        teardown()
                        None
                        raise e
                else:
                    pytest.skip("Requested world size doesn't match current world size")
            else:
                spawn_for_all_world_sizes(worker_process, world_sizes, (func, args, error_queue))
            if not error_queue.empty():
                msg = error_queue.get()
                pytest.skip(msg)
        current_frame = inspect.currentframe()
        assert current_frame is not None
        caller_module = inspect.getmodule(current_frame.f_back)
        setattr(caller_module, f'test_{name}', replacement)
        return func
    return prepare_test


def split_module(module: nn.Sequential, balance: Iterable[int], devices: List[torch.device]) ->Tuple[List[nn.Sequential], List[int], List[torch.device]]:
    """Splits a module into multiple partitions.

    Returns:
        A tuple of (partitions, balance, devices).

        Partitions are represented as a :class:`~torch.nn.ModuleList` whose
        item is a partition. All layers in a partition are placed in the
        same device.

    Raises:
        BalanceError:
            wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """
    balance = list(balance)
    if len(module) != sum(balance):
        raise BalanceError(f'module and sum of balance have different length (module: {len(module)}, sum of balance: {sum(balance)})')
    if any(x <= 0 for x in balance):
        raise BalanceError(f'all balance numbers must be positive integer (balance: {balance})')
    if len(balance) > len(devices):
        raise IndexError(f'too few devices to hold given partitions (devices: {len(devices)}, partitions: {len(balance)})')
    j = 0
    partitions = []
    layers: NamedModules = OrderedDict()
    for name, layer in module.named_children():
        layers[name] = layer
        if len(layers) == balance[j]:
            partition = nn.Sequential(layers)
            device = devices[j]
            partition
            partitions.append(partition)
            layers.clear()
            j += 1
    partitions = cast(List[nn.Sequential], nn.ModuleList(partitions))
    del devices[j:]
    return partitions, balance, devices


def verify_skippables(module: nn.Sequential) ->None:
    """Verifies if the underlying skippable modules satisfy integrity.

    Every skip tensor must have only one pair of `stash` and `pop`. If there
    are one or more unmatched pairs, it will raise :exc:`TypeError` with the
    detailed messages.

    Here are a few failure cases. :func:`verify_skippables` will report failure
    for these cases::

        # Layer1 stashes "1to3".
        # Layer3 pops "1to3".

        nn.Sequential(Layer1(), Layer2())
        #               └──── ?

        nn.Sequential(Layer2(), Layer3())
        #                   ? ────┘

        nn.Sequential(Layer1(), Layer2(), Layer3(), Layer3())
        #               └───────────────────┘       ^^^^^^

        nn.Sequential(Layer1(), Layer1(), Layer2(), Layer3())
        #             ^^^^^^      └───────────────────┘

    To use the same name for multiple skip tensors, they must be isolated by
    different namespaces. See :meth:`isolate()
    <torchpipe.skip.skippable.Skippable.isolate>`.

    Raises:
        TypeError:
            one or more pairs of `stash` and `pop` are not matched.

    """
    stashed: Set[Tuple[Namespace, str]] = set()
    popped: Set[Tuple[Namespace, str]] = set()
    msgs: List[str] = []
    for layer_name, layer in module.named_children():
        if not isinstance(layer, Skippable):
            continue
        for name in (layer.stashable_names & layer.poppable_names):
            msg = f"'{layer_name}' declared '{name}' both as stashable and as poppable"
            msgs.append(msg)
        for ns, name in layer.stashable():
            if name in layer.poppable_names:
                continue
            if (ns, name) in stashed:
                msg = f"'{layer_name}' redeclared '{name}' as stashable but not isolated by namespace"
                msgs.append(msg)
                continue
            stashed.add((ns, name))
        for ns, name in layer.poppable():
            if name in layer.stashable_names:
                continue
            if (ns, name) in popped:
                msg = f"'{layer_name}' redeclared '{name}' as poppable but not isolated by namespace"
                msgs.append(msg)
                continue
            if (ns, name) not in stashed:
                msg = f"'{layer_name}' declared '{name}' as poppable but it was not stashed"
                msgs.append(msg)
                continue
            popped.add((ns, name))
    for _, name in (stashed - popped):
        msg = f"no module declared '{name}' as poppable but stashed"
        msgs.append(msg)
    if msgs:
        raise TypeError('one or more pairs of stash and pop do not match:\n\n%s' % '\n'.join('* %s' % x for x in msgs))


def verify_splitting(module: nn.Sequential, partitions: List[nn.Sequential], balance: Iterable[int], devices: List[torch.device]) ->None:
    num_parameters = len(list(module.parameters()))
    num_child_parameters = sum(len(list(child.parameters())) for child in module.children())
    if num_parameters == num_child_parameters:
        return
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            parti = partitions[i]
            partj = partitions[j]
            if devices[i] == devices[j]:
                continue
            for p in parti.parameters():
                for q in partj.parameters():
                    if p is q:
                        raise ValueError('module with duplicate parameters on distinct devices is not supported')


class Pipe(Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train on Pipe_. If the module requires lots of memory, Pipe will be
    very efficient.
    ::

        model = nn.Sequential(a, b, c, d)
        model = Pipe(model, balance=[1, 1, 1, 1], chunks=8)
        output = model(input)

    .. _Pipe: https://arxiv.org/abs/1811.06965

    Pipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should determine the balance when defining a :class:`Pipe` module, as
    balancing will not be done automatically. The module will be partitioned
    into multiple devices according to the given balance. You may rely on
    heuristics to find your own optimal configuration.

    Args:
        module (torch.nn.Sequential):
            sequential module to be parallelized
        balance (ints):
            list of number of layers in each partition

    Keyword Args:
        devices (iterable of devices):
            devices to use (default: all CUDA devices)
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``)
        deferred_batch_norm (bool):
            whether to use deferred BatchNorm moving statistics (default:
            :data:`False`, see :class:`Deferred Batch Normalization <DeferredBatchNorm>` for more
            details)

    Raises:
        TypeError:
            the module is not a :class:`nn.Sequential <torch.nn.Sequential>`.
        ValueError:
            invalid arguments, or wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """
    balance: List[int] = []
    devices: List[torch.device] = []
    chunks: int = 1
    checkpoint: str = 'except_last'

    def __init__(self, module: nn.Sequential, balance: Optional[Iterable[int]]=None, *, devices: Optional[Devices]=None, chunks: int=chunks, checkpoint: str=checkpoint, deferred_batch_norm: bool=False) ->None:
        super().__init__()
        if torch_version()[:2] >= (1, 8):
            warnings.warn('fairscale.nn.Pipe has been upstreamed to PyTorch as torch.distributed.pipeline.sync.Pipe. It is now deprecated and will be removed in a future version of fairscale. The PyTorch API has minor changes. Please see https://pytorch.org/docs/stable/pipeline.html for details.', DeprecationWarning)
        chunks = int(chunks)
        checkpoint = str(checkpoint)
        if balance is None:
            raise ValueError(recommend_auto_balance('balance is required'))
        if chunks <= 0:
            raise ValueError('number of chunks must be positive integer')
        if checkpoint not in ['always', 'except_last', 'never']:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")
        verify_module(module)
        verify_skippables(module)
        self.chunks = chunks
        self.checkpoint = checkpoint
        if deferred_batch_norm:
            module = DeferredBatchNorm.convert_deferred_batch_norm(module, chunks)
        if devices is None:
            devices = range(torch.cuda.device_count())
        devices = [torch.device(d) for d in devices]
        devices = cast(List[torch.device], devices)
        try:
            self.partitions, self.balance, self.devices = split_module(module, balance, devices)
        except BalanceError as exc:
            raise ValueError(recommend_auto_balance(str(exc)))
        verify_splitting(module, self.partitions, self.balance, self.devices)
        self._copy_streams: List[List[AbstractStream]] = []
        self._skip_layout = inspect_skip_layout(self.partitions)
        copy_streams = self._ensure_copy_streams()
        checkpoint_stop = {'always': self.chunks, 'except_last': self.chunks - 1, 'never': 0}[self.checkpoint]
        self.pipeline = Pipeline(self.partitions, self.devices, copy_streams, self._skip_layout, checkpoint_stop)

    def __len__(self) ->int:
        """Counts the length of the underlying sequential module."""
        return sum(len(p) for p in self.partitions)

    def __getitem__(self, index: int) ->nn.Module:
        """Gets a layer in the underlying sequential module."""
        partitions = self.partitions
        if index < 0:
            partitions = partitions[::-1]
        for partition in partitions:
            try:
                return partition[index]
            except IndexError:
                pass
            shift = len(partition)
            if index < 0:
                index += shift
            else:
                index -= shift
        raise IndexError

    def __iter__(self) ->Iterable[nn.Module]:
        """Iterates over children of the underlying sequential module."""
        for partition in self.partitions:
            yield from partition

    def cuda(self, device: Optional[Device]=None) ->'Pipe':
        raise MOVING_DENIED

    def cpu(self) ->'Pipe':
        raise MOVING_DENIED

    def to(self, *args: Any, **kwargs: Any) ->'Pipe':
        """Deny these usages:
        - to(device[, dtype, non_blocking])
        - to(tensor[, non_blocking])

        But allow this:
        - to(dtype[, non_blocking])"""
        if 'device' in kwargs or 'tensor' in kwargs:
            raise MOVING_DENIED
        if args:
            if isinstance(args[0], (torch.device, int, str)):
                raise MOVING_DENIED
            if torch.is_tensor(args[0]):
                raise MOVING_DENIED
        return super()

    def _ensure_copy_streams(self) ->List[List[AbstractStream]]:
        """Ensures that :class:`Pipe` caches CUDA streams for copy.

        It's worth to cache CUDA streams although PyTorch already manages a
        pool of pre-allocated CUDA streams, because it may reduce GPU memory
        fragementation when the number of micro-batches is small.

        """
        if not self._copy_streams:
            for device in self.devices:
                self._copy_streams.append([new_stream(device) for _ in range(self.chunks)])
        return self._copy_streams

    def forward(self, input: TensorOrTensors) ->TensorOrTensors:
        """:class:`Pipe` is a fairly transparent module wrapper. It doesn't
        modify the input and output signature of the underlying module. But
        there's type restriction. Input and output have to be a
        :class:`~torch.Tensor` or a tuple of tensors. This restriction is
        applied at partition boundaries too.

        Args:
            input (torch.Tensor or tensors): input mini-batch

        Returns:
            tensor or tensors: output mini-batch

        Raises:
            TypeError: input is not a tensor or tensors.

        """
        microbatch.check(input)
        if not self.devices:
            return input
        batches = microbatch.scatter(input, self.chunks)
        self.pipeline.run(batches)
        output = microbatch.gather(batches)
        return output


DtypeOrDtypes = Union[torch.dtype, List[torch.dtype]]


class PipeBackRedirect(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, dest, event, message, transport, futures):
        ctx.dest = dest
        ctx.event = event
        ctx.message = message
        ctx.transport = transport
        ctx.futures = futures
        return inputs

    @staticmethod
    def backward(ctx, *grad):
        ctx.message.tensors = tuple(grad)
        ctx.transport.send_message(ctx.message, sync=False, skip_header=True)
        ctx.event.set()
        return None, None, None, None, None, None


SizeOrSizes = Union[torch.Size, List[torch.Size]]


def set_device_based_on_group(group: ProcessGroup) ->None:
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())


def get_dtype(tensor: TensorOrTensors) ->DtypeOrDtypes:
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype
    else:
        return [t.dtype for t in tensor]


def get_shapes(tensor: TensorOrTensors) ->SizeOrSizes:
    if isinstance(tensor, torch.Tensor):
        return tensor.shape
    else:
        return [t.shape for t in tensor]


class Net(nn.Module):

    def __init__(self) ->None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: Any) ->torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class LargeNet(Net):

    def __init__(self) ->None:
        super(LargeNet, self).__init__()
        self.fc2 = nn.Linear(10, 5000000, bias=False)
        self.fc3 = nn.Linear(5000000, 4, bias=False)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, *args):
        src = args[0]
        src_mask = args[1]
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class Branch(torch.nn.Module):

    def __init__(self, features: int):
        super().__init__()
        self.left = nn.Linear(in_features=features, out_features=features)
        self.right = nn.Linear(in_features=features, out_features=features)

    def forward(self, x):
        if x.sum() > 1000:
            return self.left(x)
        else:
            return self.right(x)


class BranchedNetwork(torch.nn.Module):

    def __init__(self, features: int):
        super().__init__()
        self.net = torch.nn.ModuleList([Branch(features) for _ in range(10)])

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


class ConcatenateTensors(nn.Module):

    def forward(self, *inputs):
        return torch.cat(inputs, dim=1)


class SplitTensors(nn.Module):

    def forward(self, input):
        return torch.split(input, (input.shape[1] + 1) // 2, dim=1)


class ShardedLinearLayer(nn.Module):

    def __init__(self, input_device, shard_devices, output_device):
        super().__init__()
        self.split = RemoteModule(input_device, SplitTensors, (), {})
        self.linear_layers_2 = nn.ModuleList([RemoteModule(shard_devices[0], nn.Linear, (2, 2), {}), RemoteModule(shard_devices[1], nn.Linear, (2, 2), {})])
        self.concatenate = RemoteModule(output_device, ConcatenateTensors, ())

    def forward(self, input):
        shards = self.split(input)
        shards = [self.linear_layers_2[i](shards[i]) for i in range(2)]
        return self.concatenate(*shards)


class ManualLinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def _set_cuda_rng_state(new_state: torch.ByteTensor, device: Union[int, str, torch.device]=-1) ->None:
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if device == -1:
        device = torch.device('cuda')
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('cuda', device)

    def cb() ->None:
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        default_generator = torch.cuda.default_generators[idx]
        default_generator.set_state(new_state)
    _lazy_call(cb)


_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('Checkpointing is not compatible with .grad(), please use .backward() if possible')
        inputs = ctx.saved_tensors
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)
        if isinstance(outputs, torch.Tensor):
            outputs = outputs,
        torch.autograd.backward(outputs, args)
        return (None,) + tuple(inp.grad for inp in detached_inputs)


def pack_kwargs(*args: Any, **kwargs: Any) ->Tuple[Tuple[str, ...], Tuple[Any, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite)

    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys: List[str] = []
    flat_args: List[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return tuple(kwarg_keys), tuple(flat_args)


def unpack_non_tensors(tensors: Tuple[torch.Tensor, ...], packed_non_tensors: Optional[Dict[str, List[Any]]]) ->Tuple[Any, ...]:
    """See split_non_tensors."""
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict), type(packed_non_tensors)
    mixed: List[Any] = []
    is_tensor_list = packed_non_tensors['is_tensor']
    objects = packed_non_tensors['objects']
    assert len(tensors) + len(objects) == len(is_tensor_list), f'len(tensors) {len(tensors)} len(objects) {len(objects)} len(is_tensor_list) {len(is_tensor_list)}'
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)


def _checkpointed_forward(original_forward: Any, weak_self: Any, offload_to_cpu: bool, *args: Any, **kwargs: Any) ->Any:
    module = weak_self()
    if not torch.is_grad_enabled() or thread_local.is_checkpointing_disabled:
        return original_forward(module, *args, **kwargs)
    args = (module,) + args
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict: Dict[str, Any] = {'offload': offload_to_cpu}
    output = CheckpointFunction.apply(torch.tensor([], requires_grad=True), original_forward, parent_ctx_dict, kwarg_keys, *flat_args)
    output_requires_grad = parent_ctx_dict['output_requires_grad']
    if not isinstance(output, torch.Tensor):
        output = [(x.detach() if not output_requires_grad else x) for x in output]
        packed_non_tensor_outputs = parent_ctx_dict['packed_non_tensor_outputs']
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)
    elif not output_requires_grad:
        output = output.detach()
    return output


def patch_batchnorm(module: nn.Module) ->List:
    """Patch all batchnorm instances (1d, 2d, 3d, sync_bn, etc.) of a module
       so that they don't track running stats when torch.no_grad() is enabled.

       This is important in activation checkpointing to ensure stats are tracked
       correctly as if there were no activation checkpointing. The reason is
       that activation checkpointing runs the forward function twice, first
       with torch.no_grad(), then with torch.grad().

    Args:
        module (nn.Module):
            The module to be patched in-place.

    Returns:
        (list):
            A list of hook handles, late can be freed.
    """

    def pre_forward(module: _BatchNorm, input: Tensor) ->None:
        if torch.is_grad_enabled():
            return
        module._track_running_stats_backup = module.track_running_stats
        module.track_running_stats = False

    def post_forward(module: _BatchNorm, input: Tensor, result: Tensor) ->None:
        if torch.is_grad_enabled():
            return
        module.track_running_stats = module._track_running_stats_backup
    hooks = []
    for name, child in module.named_modules():
        if isinstance(child, _BatchNorm) and not hasattr(child, 'disable_patch_batchnorm'):
            pre_handle = child.register_forward_pre_hook(pre_forward)
            post_handle = child.register_forward_hook(post_forward)
            hooks += [pre_handle, post_handle]
    return hooks


def checkpoint_wrapper(module: nn.Module, offload_to_cpu: bool=False) ->nn.Module:
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:

        - wraps an nn.Module, so that all subsequent calls will use checkpointing
        - handles keyword arguments in the forward
        - handles non-Tensor outputs from the forward
        - supports offloading activations to CPU

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))

    To understand the benefits of checkpointing and the `offload_to_cpu` flag,
    let's divide activations into 2 types: inner activations and outer
    activations w.r.t. the checkpointed modules. The inner ones are saved
    by activation checkpointing, the outer ones are saved by offload_to_cpu.

    In terms of GPU memory savings:

        - When inner ones are large in size and outer ones are small,
          checkpointing helps a lot, offload_to_cpu may help a little.
        - When inner ones are small and outer ones are large,
          checkpointing helps little, offload_to_cpu helps a lot.
        - When both inner and outer are large, both help and the
          benefit is additive.

    ..Note::

        The first and last layers are not likely to benefit from the `offload_to_cpu` flag
        because (1) there are typically other references to the first layer's input, so
        the GPU memory won't be freed; (2) the input to the last layer is immediately
        used by the backward pass and won't result in memory savings.

    Args:
        module (nn.Module):
            The module to be wrapped
        offload_to_cpu (bool):
            Whether to offload activations to CPU.

    Returns:
        (nn.Module):
            Wrapped module
    """
    patch_batchnorm(module)
    module.forward = functools.partial(_checkpointed_forward, type(module).forward, weakref.ref(module), offload_to_cpu)
    return module


class BasicModel(nn.Module):
    """Basic model with a single FFN being checkpointed.

    Used for extensive checkings: equivalency with non-checkpoint, torch-checkpoint, etc.
    """

    def __init__(self, use_pytorch_checkpoint=False, use_fairscale_checkpoint=False, **kwargs):
        super().__init__()
        torch.manual_seed(0)
        assert not (use_pytorch_checkpoint and use_fairscale_checkpoint), 'Cannot use both pytorch and fairscale checkpointing mechanisms.'
        self.use_pytorch_checkpoint = use_pytorch_checkpoint
        self.ffn = nn.Sequential(nn.Linear(32, 128), nn.Dropout(p=0.5), nn.Linear(128, 32))
        if use_fairscale_checkpoint:
            self.ffn = checkpoint_wrapper(self.ffn, **kwargs)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        if self.use_pytorch_checkpoint:
            x = torch_checkpoint_wrapper(self.ffn, x)
        else:
            x = self.ffn(x)
        return self.out(x)


class CpuOffloadModel(nn.Module):
    """Model used to check cpu offload memory saving"""

    def __init__(self, enable_checkpoint=False, cpu_offload=False):
        super().__init__()
        torch.manual_seed(0)
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 8)), nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 4), nn.Linear(4, 4)), nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 8), nn.Linear(8, 2)))
        if enable_checkpoint:
            for i, layer in enumerate(self.layers):
                self.layers[i] = checkpoint_wrapper(layer, cpu_offload if i == 1 else False)

    def forward(self, x):
        return self.layers(x)


class MultiinMultioutModel(nn.Module):
    """Model used to check different inputs and outputs"""

    def __init__(self, multiout=False, checkpoint_config=0):
        super().__init__()
        torch.manual_seed(0)
        self.multiout = multiout
        self.conv1 = nn.Sequential(nn.Conv2d(1, 5, 3), nn.ReLU(), nn.Conv2d(5, 5, 3))
        self.conv2 = nn.Sequential(nn.Conv2d(3, 5, 3), nn.ReLU(), nn.Conv2d(5, 5, 3))
        assert 0 <= checkpoint_config <= 3
        if checkpoint_config & 1:
            self.conv1 = checkpoint_wrapper(self.conv1)
        if checkpoint_config & 1 << 1:
            self.conv2 = checkpoint_wrapper(self.conv2)

    def forward(self, x1, x2=None):
        out1 = self.conv1(x1)
        out2 = self.conv2(x2)
        if self.multiout:
            return out1, out2
        return out1 + out2


class TransformerWithSharedParams(nn.Module):

    def __init__(self, group, *unused_args, d_vocab=23, d_model=16, add_bn=True, **unused_kwargs):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        torch.manual_seed(0)
        assert d_vocab >= 12
        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=8, dropout=0.1)
        self.output_proj = nn.Linear(d_model, d_vocab)
        self.output_proj.weight = self.embed_tokens.weight
        self.register_buffer('vocab_bias', self.embed_tokens.weight.new_ones((d_model,)))
        self.register_buffer('long_buffer', torch.zeros_like(self.vocab_bias, dtype=torch.long))
        self.bs = 2
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)
        src = torch.arange(12, device=device).view(6, self.bs)
        tgt = torch.arange(self.bs * 4, device=device).view(4, self.bs)
        return src, tgt

    def forward(self, src_ids, tgt_ids):
        src = self.embed_tokens(src_ids)
        src = src + self.vocab_bias + self.long_buffer.type_as(src)
        tgt = self.embed_tokens(tgt_ids)
        tgt = self.bn(tgt)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    def get_loss(self, input, output):
        _, tgt = input
        return nn.functional.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1), reduction='sum')

    def run_backward(self, loss):
        loss.backward()


class NestedWrappedModule(nn.Module):

    def __init__(self, group, wrapper_config, wrap_everything=False, checkpoint=False):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        self.wrapper_config = wrapper_config

        def _maybe_wrap(layer):
            if wrapper_config is not None:
                return FullyShardedDataParallel(layer, group, **wrapper_config)
            return layer
        torch.manual_seed(0)
        self.module = nn.Sequential(nn.Linear(8, 4), _maybe_wrap(nn.Sequential(_maybe_wrap(nn.Linear(4, 16)), nn.Linear(16, 16))), _maybe_wrap(nn.Linear(16, 4)), nn.Linear(4, 8))
        if wrap_everything:
            if checkpoint:
                self.module = nn.Sequential(_maybe_wrap(checkpoint_wrapper(nn.Linear(8, 4))), _maybe_wrap(checkpoint_wrapper(nn.Linear(4, 16))), _maybe_wrap(checkpoint_wrapper(nn.Linear(16, 4))), _maybe_wrap(checkpoint_wrapper(nn.Linear(4, 8))))
            else:
                self.module = nn.Sequential(_maybe_wrap(nn.Linear(8, 4)), _maybe_wrap(nn.Linear(4, 16)), _maybe_wrap(nn.Linear(16, 4)), _maybe_wrap(nn.Linear(4, 8)))

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)
        return torch.rand(4, 8, device=device),

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = output.sum()
        return loss

    def run_backward(self, loss):
        loss.backward()


class DummyDDP(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@functools.lru_cache()
def get_cycles_per_ms() ->float:
    """Measure and return approximate number of cycles per millisecond for torch.cuda._sleep

    Copied from: github.com/pytorch/pytorch/blob/master/test/test_cuda.py
    """

    def measure() ->float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms
    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    return mean(vals[2:num - 2])


class MixtureOfExperts(NestedWrappedModule):

    def __init__(self, group, wrapper_config, checkpoint_act=False, delay_before_free_ms=0, expert_group=None):
        super().__init__(group, wrapper_config)
        self.group = group
        self.delay_before_free_ms = delay_before_free_ms
        torch.manual_seed(42 + group.rank())
        d_expert = 23
        d_shared = 12
        d_input = 8
        expert = nn.Linear(d_expert, d_shared)
        self.num_expert_params = sum([p.numel() for p in expert.parameters()])
        for p in expert.parameters():
            p.expert = True
        torch.manual_seed(0)
        shared = nn.Linear(d_shared, d_expert)
        if checkpoint_act:
            expert = checkpoint_wrapper(expert)
            shared = checkpoint_wrapper(shared)
        if wrapper_config is not None:
            expert_group = expert_group or torch.distributed.new_group([group.rank()])
            expert = FullyShardedDataParallel(expert, process_group=expert_group, process_group_reduce_scatter=expert_group, **wrapper_config)
            shared = FullyShardedDataParallel(shared, group, **wrapper_config)
        self.module = nn.Sequential(nn.Linear(d_input, d_shared), shared, expert, nn.Linear(d_shared, d_input))

    def forward(self, x):
        if self.delay_before_free_ms > 0:
            expert = self.module[2]
            if isinstance(expert, FullyShardedDataParallel):
                orig_free_full_params = self.module[2]._free_full_params

                def _free_full_params_with_delay(*args):
                    torch.cuda._sleep(int(self.delay_before_free_ms * get_cycles_per_ms()))
                    return orig_free_full_params(*args)
                assert hasattr(expert, '_free_full_params')
                with mock.patch.object(expert, '_free_full_params', _free_full_params_with_delay):
                    return self.module(x)
        return self.module(x)

    def run_backward(self, loss):
        loss.backward()
        if self.wrapper_config is None:
            with torch.no_grad():
                for p in self.parameters():
                    if hasattr(p, 'expert'):
                        continue
                    p.grad.data.div_(self.world_size)
                    torch.distributed.all_reduce(p.grad.data, group=self.group)


class ModuleWithDelay(nn.Module):

    def __init__(self, module, delay_after_loss_ms=0, delay_before_reduction_ms=0):
        super().__init__()
        self.delay_after_loss_ms = delay_after_loss_ms
        self.delay_before_reduction_ms = delay_before_reduction_ms
        self.module = module

    def get_input(self, device):
        return self.module.get_input(device)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = self.module.get_loss(input, output)
        if self.delay_after_loss_ms > 0:
            torch.cuda._sleep(int(self.delay_after_loss_ms * get_cycles_per_ms()))
        return loss

    def run_backward(self, loss):
        orig_reduce_scatter = torch.distributed.reduce_scatter

        def _delayed_reduce_scatter(*args, **kwargs):
            if self.delay_before_reduction_ms > 0:
                torch.cuda._sleep(int(self.delay_before_reduction_ms * get_cycles_per_ms()))
            return orig_reduce_scatter(*args, **kwargs)
        with mock.patch('torch.distributed.reduce_scatter', _delayed_reduce_scatter):
            self.module.run_backward(loss)


class NestedWrappedModuleWithDelay(ModuleWithDelay):

    def __init__(self, group, wrapper_config, **kwargs):
        super().__init__(NestedWrappedModule(group, wrapper_config), **kwargs)


class FSDP:

    def get_model_config():
        return {'vocab_size': 10000, 'ninp': 2048, 'nhid': 2048, 'nhead': 32, 'dropout': 0, 'initrange': 0.1, 'scaler': GradScaler(), 'clip_value': 0.05, 'num_decoder_layers': 10, 'seq_len': 32}

    def get_benchmark_config():
        return {'epochs': 1, 'lr': 0.001, 'batch_size': 8, 'criterion': nn.CrossEntropyLoss()}

    def get_golden_real_stats():
        raise NotImplementedError('Synthetic data benchmarks are not supported.')

    def get_golden_synthetic_stats():
        return {'avg_wps': 486.303, 'std_dev_wps': 71.307, 'peak_mem_usage': [5.5055 * 2 ** 30, 5.5055 * 2 ** 30, 5.5055 * 2 ** 30, 5.5055 * 2 ** 30]}


class FreezeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        self.head = nn.Linear(64, 10)
        self.trunk = FSDP(self.trunk)

    def forward(self, x):
        return self.head(self.trunk(x))


D_MODEL = 2


TILE = 2


VOCAB = 4


class Model(nn.Module):

    def __init__(self, with_fsdp=False, wrap_middle='none'):
        super().__init__()
        self.l0 = nn.Embedding(VOCAB, D_MODEL).half()
        nn.init.uniform_(self.l0.weight, -0.1, 0.1)
        self.l1 = MEVO(self.l0.weight, tile_factor=TILE, reduction='sum')
        self.middle = nn.Linear(D_MODEL, D_MODEL).half()
        self.ln1 = nn.LayerNorm(D_MODEL).half()
        self.ln2 = nn.LayerNorm(D_MODEL).half()
        if with_fsdp:
            self.l0 = FSDP(self.l0, flatten_parameters=False, mixed_precision=False, compute_dtype=torch.float16)
            self.l1 = FSDP(self.l1, flatten_parameters=False, mixed_precision=False, compute_dtype=torch.float16)
            self.l1.append_shared_param(self.l0.module.weight)
            assert wrap_middle in ['none', 'flat', 'nonflat']
            if wrap_middle != 'none':
                self.middle = FSDP(self.middle, flatten_parameters=wrap_middle == 'flat', mixed_precision=False, compute_dtype=torch.float16)

    def forward(self, x):
        target = x + 1
        x = self.l0(x)
        x = self.ln1(x)
        x = self.middle(x)
        x = self.ln2(x)
        x = self.l1(x, target)
        None
        assert x.item() not in [float('-inf'), float('inf')]
        return x


class NestedTrunkModel(nn.Module):

    def __init__(self, with_fsdp, freeze_after_wrap_fsdp):
        super().__init__()
        self.trunk = nn.Sequential(self._create_block(3, 64, with_fsdp, freeze_after_wrap_fsdp), self._create_block(64, 64, with_fsdp, freeze_after_wrap_fsdp))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), nn.Linear(64, 10))
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap()

    def fsdp_wrap(self):
        for name, child in self.trunk.named_children():
            wrapped_child = FSDP(child)
            setattr(self.trunk, name, wrapped_child)
        self.trunk = FSDP(self.trunk)
        self.head = FSDP(self.head)

    def forward(self, x):
        return self.head(self.trunk(x))

    def _create_block(self, in_channels, out_channels, with_fsdp, freeze_after_wrap_fsdp):
        block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3), nn.ReLU(inplace=True))
        return block


class FFN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ModelOutput(OrderedDict):

    def to_tuple(self):
        return tuple(self[k] for k in self.keys())

    def __post_init__(self):
        class_fields = getattr(self, '__dataclass_fields__')
        for field in class_fields:
            v = getattr(self, field)
            if v is not None:
                self[field] = v

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for k, v in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]


class TransformerWithCustomOutput(nn.Transformer):

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return TransformerOutput(output=output)


class TransformerWithLMHead(nn.Module):

    def __init__(self, d_vocab=100, d_model=16):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = TransformerWithCustomOutput(d_model, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64)
        self.output_proj = nn.Linear(d_model, d_vocab)

    def generate_random_sequences(self, seq_len=20, batch_size=2):
        source_seq = torch.randint(high=self.d_vocab, size=(seq_len, batch_size))
        target_seq = torch.randint(high=self.d_vocab, size=(seq_len, batch_size))
        return source_seq, target_seq

    def forward(self, source_seq, target_seq):
        source_embeddings = self.embed_tokens(source_seq)
        target_embeddings = self.embed_tokens(target_seq)
        output = self.transformer(source_embeddings, target_embeddings)
        return self.output_proj(output[0])


class ConvolutionalModel(nn.Module):

    def __init__(self, embedding_size: int, with_fsdp: bool, process_group):
        super().__init__()
        self.conv1 = self._conv_block(3, embedding_size)
        self.conv2: nn.Module = self._conv_block(embedding_size, embedding_size // 2)
        self.conv3: nn.Module = self._conv_block(embedding_size // 2, embedding_size)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()
        self.fc1: nn.Module = nn.Linear(embedding_size, 2 * embedding_size)
        self.fc2: nn.Module = nn.Linear(2 * embedding_size, 2 * embedding_size)
        self.fc3: nn.Module = nn.Linear(2 * embedding_size, embedding_size + 1)
        self.fc4: nn.Module = nn.Linear(embedding_size + 1, embedding_size)
        if with_fsdp:
            self.conv2 = FullyShardedDataParallel(self.conv2, process_group=process_group)
            self.conv3 = FullyShardedDataParallel(self.conv3, process_group=process_group, flatten_parameters=False)
            self.fc1 = FullyShardedDataParallel(self.fc1, process_group=process_group)
            self.fc3 = FullyShardedDataParallel(self.fc3, process_group=process_group, flatten_parameters=False)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class Model2(nn.Module):
    """Model to test FSDP(checkpoint(), checkpoint())."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3), nn.BatchNorm2d(4), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3), nn.BatchNorm2d(4), nn.ReLU(inplace=False))
        self.block3 = nn.Sequential(nn.Conv2d(4, 8, kernel_size=3), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), nn.Linear(8, 10))

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.head(self.block3(self.block2(self.block1(x))))
        elif isinstance(x, list):
            ys = [self.head(self.block3(self.block2(self.block1(e)))) for e in x]
            return torch.cat(ys, dim=0)


class ModelWithUnusedParams(nn.Module):

    def __init__(self, wrap_l2):
        super().__init__()
        self.l = nn.Linear(4, 4)
        self.not_trained = nn.Linear(4, 4).requires_grad_(False)
        self.not_trained = FullyShardedDataParallel(self.not_trained)
        self.l2 = nn.Linear(4, 4)
        if wrap_l2:
            self.l2 = FullyShardedDataParallel(self.l2)

    def forward(self, x):
        with torch.no_grad():
            y = self.not_trained(x)
        return self.l2(self.l(x)) - y


class Layer(nn.Module):

    def __init__(self, compute_cycles, has_params: bool):
        super().__init__()
        self.sleep_cycles = compute_cycles
        self.optional_param = None
        if has_params:
            self.optional_param = nn.Parameter(torch.rand(1))

    def forward(self, x):
        self.e1 = Event(enable_timing=True)
        self.e2 = Event(enable_timing=True)
        self.e1.record()
        if self.sleep_cycles > 0:
            torch.cuda._sleep(self.sleep_cycles)
        if self.optional_param is not None:
            x = x + self.optional_param
        self.e2.record()
        return x

    def get_time(self):
        return self.e1.elapsed_time(self.e2)


_relu_inplace = True


class ResBlock(Module):
    """Conv block in regnet with residual connection."""

    def __init__(self, width_in, width_out):
        super().__init__()
        self.proj = Conv2d(width_in, width_out, (1, 1), (2, 2), bias=False)
        self.bn = BatchNorm2d(width_out)
        self.f = Sequential(Sequential(Conv2d(width_in, width_out, (1, 1), (1, 1), bias=False), BatchNorm2d(width_out), ReLU(_relu_inplace)), Sequential(Conv2d(width_out, width_out, (3, 3), (2, 2), (1, 1), groups=2, bias=False), BatchNorm2d(width_out), ReLU(_relu_inplace)), Sequential(AdaptiveAvgPool2d((1, 1)), Sequential(Conv2d(width_out, 2, (1, 1), (1, 1), bias=False), ReLU(_relu_inplace), Conv2d(2, width_out, (1, 1), (1, 1), bias=False), Sigmoid())), Conv2d(width_out, width_out, (1, 1), (1, 1), bias=False), BatchNorm2d(width_out))
        self.relu = ReLU()
        self.need_fsdp_wrap = True

    def forward(self, x):
        x = self.bn(self.proj(x)) + self.f(x)
        return self.relu(x)


class SimpleModuleWithCheckpointing(nn.Module):

    def __init__(self, flatten, mixed_precision, fsdp_wrap_ckpt):
        super().__init__()
        if fsdp_wrap_ckpt:
            middle_module = FSDP(checkpoint_wrapper(nn.Linear(3, 3)), flatten_parameters=flatten, mixed_precision=mixed_precision)
        else:
            middle_module = checkpoint_wrapper(FSDP(nn.Linear(3, 3), flatten_parameters=flatten, mixed_precision=mixed_precision))
        self.ffn = nn.Sequential(nn.Linear(3, 3), middle_module, nn.Linear(3, 3))

    def forward(self, x):
        return self.ffn(x)


def _get_mlp(tripwire: bool=False):
    if not tripwire:
        return Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))


    class Tripwire(torch.nn.Module):
        """A model made to expose possible corner cases"""

        def __init__(self) ->None:
            super().__init__()
            self.model = Linear(2, 3, bias=False)
            self.register_parameter('tripwire', torch.nn.Parameter(torch.LongTensor((3, 3)), requires_grad=False))

        def forward(self, x):
            return self.model(x)
    return Tripwire()


class _DoubleInput(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = _get_mlp()

    def forward(self, x, y):
        x1 = self.mlp(x)
        x2 = self.mlp(y)
        return torch.cat((x1, x2), dim=1)


class IdentityLayer2D(torch.nn.Module):

    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


class RoundRobinGate(torch.nn.Module):

    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts

    def forward(self, input):
        s = input.shape[0]
        assert s % self.num_experts == 0, f'{s} % {self.num_experts} != 0'
        capacity = 2 * s // self.num_experts
        output = torch.zeros(s, self.num_experts, capacity, dtype=input.dtype, device=input.device)
        for i in range(s):
            output[i, i % self.num_experts, i // self.num_experts] = 1.0
        return 0.0, output, output.bool()


class Pass(nn.Module):

    def forward(self, input):
        return input


class FeedForward(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        torch.manual_seed(7)
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        out = self.identity(out)
        return out


class SimpleConvNet(nn.Module):

    def __init__(self):
        torch.manual_seed(24)
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.identity = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        out = self.relu4(out)
        out = self.fc3(out)
        out = self.identity(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Branch,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BranchedNetwork,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CpuOffloadModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeferredBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DummyDDP,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (FeedForward,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeedForwardLayer,
     lambda: ([], {'d_model': 4, 'dim_feedforward': 4, 'activation': _mock_layer(), 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IdentityLayer2D,
     lambda: ([], {'m': 4, 'n': 4}),
     lambda: ([], {}),
     True),
    (LinearLayer,
     lambda: ([], {'ninp': 4, 'ntoken': 4, 'initrange': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Model2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ModuleWithDelay,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Pass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncodingLayer,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'width_in': 4, 'width_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RoundRobinGate,
     lambda: ([], {'model_dim': 4, 'num_experts': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SplitTensors,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SyncBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerDecoderLayer,
     lambda: ([], {'ninp': 4, 'nhead': 4, 'nhid': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_facebookresearch_fairscale(_paritybench_base):
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

