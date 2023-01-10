import sys
_module = sys.modules[__name__]
del sys
arguments = _module
data_utils = _module
configure_data = _module
datasets = _module
samplers = _module
sp_tokenizer = _module
templates = _module
unified_tokenizer = _module
vqvae_tokenizer = _module
setup_connection = _module
dataset = _module
fid_score = _module
inception = _module
inception_score = _module
finetune = _module
fp16 = _module
fp16 = _module
fp16util = _module
loss_scaler = _module
generate_samples = _module
generation = _module
magnify = _module
sampling = _module
learning_rates = _module
model = _module
distributed = _module
gpt2_modeling = _module
mpu = _module
cross_entropy = _module
data = _module
grads = _module
initialize = _module
layers = _module
mappings = _module
random = _module
sparse_transformer = _module
utils = _module
preprocess = _module
preprocess_text_image_data = _module
preprocess_text_jsonformat_data = _module
pretokenized_data = _module
raw_datasets = _module
utils = _module
preprocess_entry = _module
pretrain_gpt2 = _module
test_lmdb = _module
utils = _module
vqvae = _module
api = _module
distributed = _module
launch = _module
vqvae_zc = _module

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


import math


import random


import copy


import numpy as np


import torch.nn.functional as F


from torch.utils import data


import logging


from torchvision import datasets


from torchvision import transforms


from collections import namedtuple


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


from torchvision.models.inception import inception_v3


from scipy import linalg


from torch.autograd import Variable


from torch.nn.functional import adaptive_avg_pool2d


import torchvision.transforms as transforms


import torch.utils.data


import torch.nn as nn


from torchvision import models


from torch import nn


from torch.nn import functional as F


from scipy.stats import entropy


from torch.nn.parameter import Parameter


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


import time


from copy import deepcopy


from torchvision.utils import save_image


import torch.distributed as dist


from torch.optim.lr_scheduler import _LRScheduler


from torch.nn.modules import Module


from torch.nn.parallel.distributed import DistributedDataParallel as DDP


from torch._six import inf


import torch.nn.init as init


from torch import _C


from torch.cuda import _lazy_call


from torch.cuda import device as device_ctx_manager


from torch.utils.data import DataLoader


from torchvision.transforms.functional import resize


import itertools


from collections import Iterable


from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP


from torch import distributed as dist


from torch import multiprocessing as mp


import warnings


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in 
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output 
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear', align_corners=True)
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


HALF_TYPES = torch.HalfTensor, torch.HalfTensor


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, HALF_TYPES):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)


FLOAT_TYPES = torch.FloatTensor, torch.FloatTensor


def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, FLOAT_TYPES):
            val = val.half()
        return val
    return conversion_helper(val, half_conversion)


class FP16_Module(nn.Module):

    def __init__(self, module):
        super(FP16_Module, self).__init__()
        self.add_module('module', module.half())

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*fp32_to_fp16(inputs), **kwargs))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


class tofp16(nn.Module):
    """
    Utility module that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def convert_module(module, dtype):
    """
    Converts a module's immediate parameters and buffers to dtype.
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data
    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data


def convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype)
    return network


class FP16Model(nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network):
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


class PyTorchDistributedDataParallel(DDP):

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.module.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


class DistributedDataParallel(Module):

    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False
        self.module = module
        self.data_parallel_group = mpu.get_data_parallel_group()
        src_rank = mpu.get_model_parallel_rank()
        for p in self.module.parameters():
            if torch.is_tensor(p):
                dist.broadcast(p, src_rank, group=self.data_parallel_group)

        def allreduce_params(reduce_after=True, no_scale=False, fp32_allreduce=False):
            if self.needs_reduction:
                self.needs_reduction = False
                buckets = {}
                for name, param in self.module.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = param.data.type()
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                if self.warn_on_half:
                    if torch.HalfTensor in buckets:
                        None
                        self.warn_on_half = False
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    if fp32_allreduce:
                        coalesced = coalesced.float()
                    if not no_scale and not reduce_after:
                        coalesced /= dist.get_world_size(group=self.data_parallel_group)
                    dist.all_reduce(coalesced, group=self.data_parallel_group)
                    torch.cuda.synchronize()
                    if not no_scale and reduce_after:
                        coalesced /= dist.get_world_size(group=self.data_parallel_group)
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)
        self.hook_handles = []
        self.hooks = []
        for param in list(self.module.parameters()):

            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)
        self.allreduce_params = allreduce_params

    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.module.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
    """
    def _sync_buffers(self):
        buffers = list(self.module._all_buffers())
        if len(buffers) > 0:
            # cross-node buffer sync
            flat_buffers = _flatten_dense_tensors(buffers)
            dist.broadcast(flat_buffers, 0)
            for buf, synced in zip(buffers, _unflatten_dense_tensors(flat_buffers, buffers)):
                buf.copy_(synced)
    def train(self, mode=True):
        # Clear NCCL communicator and CUDA event cache of the default group ID,
        # These cache will be recreated at the later call. This is currently a
        # work-around for a potential NCCL deadlock.
        if dist._backend == dist.dist_backend.NCCL:
            dist._clear_group_cache()
        super(DistributedDataParallel, self).train(mode)
        self.module.train(mode)
    """


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


class GPT2Model(torch.nn.Module):
    """GPT-2 Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self, num_layers, vocab_size, hidden_size, num_attention_heads, embedding_dropout_prob, attention_dropout_prob, output_dropout_prob, max_sequence_length, max_memory_length, checkpoint_activations, checkpoint_num_layers=1, parallel_output=True, query_window=128, key_window_times=6, num_pivot=768):
        super(GPT2Model, self).__init__()
        self.parallel_output = parallel_output
        init_method = init_method_normal(std=0.02)
        self.word_embeddings = mpu.VocabParallelEmbedding(vocab_size, hidden_size, init_method=init_method)
        self.transformer = mpu.GPT2ParallelTransformer(num_layers, hidden_size, num_attention_heads, max_sequence_length, max_memory_length, embedding_dropout_prob, attention_dropout_prob, output_dropout_prob, checkpoint_activations, checkpoint_num_layers, query_window=query_window, key_window_times=key_window_times, num_pivot=num_pivot)

    def forward(self, input_ids, position_ids, attention_mask, txt_indices_bool, img_indices_bool, is_sparse, *mems):
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        transformer_output = self.transformer(embeddings, position_ids, attention_mask, txt_indices_bool, img_indices_bool, is_sparse, *mems)
        logits, *hidden_layers = transformer_output
        logits_parallel = mpu.copy_to_model_parallel_region(logits)
        logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)
        if self.parallel_output:
            return logits_parallel, *hidden_layers
        return mpu.gather_from_model_parallel_region(logits_parallel), *hidden_layers


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
        first and last index of the vocabulary belonging to the `rank`
        partition: Note that indecies in [fist, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)


_MODEL_PARALLEL_GROUP = None


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def _initialize_affine_weight(weight, output_size, input_size, per_partition_size, partition_dim, init_method, stride=1, return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None
    master_weight = torch.empty(output_size, input_size, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    torch.distributed.all_reduce(input_, group=group)
    return input_


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_model_parallel_region(input_):
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

    def __init__(self, num_embeddings, embedding_dim, init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(self.num_embeddings, get_model_parallel_rank(), get_model_parallel_world_size())
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.weight = Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))
        self.weight.model_parallel = True
        _initialize_affine_weight(self.weight, self.num_embeddings, self.embedding_dim, self.num_embeddings_per_partition, 0, init_method)

    def forward(self, input_):
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
        return _reduce(grad_output)


def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def _gather(input_):
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


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _split(input_):
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


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


def gather_from_model_parallel_region(input_):
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

    def __init__(self, num_embeddings, embedding_dim, init_method=init.xavier_normal_, keep_master_weight_for_test=False):
        super(ParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        world_size = get_model_parallel_world_size()
        self.embedding_dim_per_partition = divide(self.embedding_dim, world_size)
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim_per_partition))
        self.weight.model_parallel = True
        _initialize_affine_weight(self.weight, self.num_embeddings, self.embedding_dim, self.embedding_dim_per_partition, 1, init_method, stride=1, return_master_weight=False)

    def forward(self, input_):
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.embedding(input_parallel, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        output = gather_from_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
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

    def __init__(self, input_size, output_size, bias=True, gather_output=True, init_method=init.xavier_normal_, stride=1, keep_master_weight_for_test=False):
        super(ColumnParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.input_size))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            self.bias.model_parallel = True
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(self.weight, self.output_size, self.input_size, self.output_size_per_partition, 0, init_method, stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
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


def scatter_to_model_parallel_region(input_):
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
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
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

    def __init__(self, input_size, output_size, bias=True, input_is_parallel=False, init_method=init.xavier_normal_, stride=1, keep_master_weight_for_test=False):
        super(RowParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.weight = Parameter(torch.Tensor(self.output_size, self.input_size_per_partition))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(self.weight, self.output_size, self.input_size, self.input_size_per_partition, 1, init_method, stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
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


def _chunk(x, w, times):
    """convert into overlapping chunkings. Chunk size = times * w, overlap size = w
    Args:
        x: [b, np, s, hn]
        ...
    """
    s = x.size(2)
    assert s % w == 0
    npad = (times - 1) * w
    x = torch.nn.functional.pad(x, (0, 0, npad, 0), value=0)
    x = x.view(x.size(0), x.size(1), x.size(2) // w, w, x.size(3))
    chunk_size = list(x.size())
    chunk_stride = list(x.stride())
    chunk_size[2] = chunk_size[2] - times + 1
    chunk_size[3] = w * times
    return x.as_strided(size=chunk_size, stride=chunk_stride)


_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):

        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)
    _lazy_call(cb)


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def sparse_attention(q, k, v, pivot_idx, pivot_attention_mask, query_window=128, key_window_times=6, attention_dropout=None):
    """ Sparse Attention
    Args:
        q, k, v: inputs, [b, num_heads, s, hn], k is padded to n * query_window
        pivot_idx: [b, num_pivots]
        pivot_attention_mask: [b, s, num_pivots]
        query_window: .
        key_window_times: key_window = query_window * key_window_times
    """
    b, n_head, s, hn = q.shape
    b, n_piv = pivot_idx.shape
    w = query_window
    pivot_idx_dummy = pivot_idx.view(b, 1, n_piv, 1).expand(b, n_head, n_piv, hn)
    pivot_k, pivot_v = torch.gather(k, 2, pivot_idx_dummy), torch.gather(v, 2, pivot_idx_dummy)
    attention_scores = torch.matmul(q, pivot_k.transpose(-1, -2))
    pivot_attention_mask = pivot_attention_mask.unsqueeze(1)
    attention_scores_pivot = torch.mul(attention_scores, pivot_attention_mask / math.sqrt(hn)) - 10000.0 * (1.0 - pivot_attention_mask)
    attention_scores_pivot = attention_scores_pivot + math.log(s // n_piv)
    window_k = _chunk(k, query_window, key_window_times)
    window_v = _chunk(v, query_window, key_window_times)
    if s % w == 0:
        assert k.shape[2] == s
        assert window_k.shape[2] == s // w
        window_q = q.view(b, n_head, s // w, w, hn)
        attention_scores = torch.matmul(window_q, window_k.transpose(-1, -2))
        window_attention_mask = torch.ones((w, w * key_window_times), dtype=attention_scores.dtype, device=q.device).tril_(diagonal=w * (key_window_times - 1))
        attention_scores_window = torch.mul(attention_scores, window_attention_mask / math.sqrt(hn)) - 10000.0 * (1.0 - window_attention_mask)
        for t in range(1, key_window_times):
            attention_scores_window[:, :, t - 1, :, :w * key_window_times - w * t] -= 10000.0
    else:
        raise ValueError('The seq_len must be exactly divided by window_size.')
    attention_scores_window = attention_scores_window.view(b, n_head, s, w * key_window_times)
    attention_scores = torch.cat((attention_scores_pivot, attention_scores_window), dim=-1)
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs = attention_dropout(attention_probs)
    context_layer = torch.matmul(attention_probs[..., :-w * key_window_times], pivot_v) + torch.einsum('bcgwk,bcgkh->bcgwh', attention_probs[..., -w * key_window_times:].view(b, n_head, s // w, w, w * key_window_times), window_v).view(b, n_head, s, hn)
    return context_layer


def sparse_attention_inference(q, k, v, pivot_and_window_idx, **kwargs):
    """the inference process of sparse attention.
    The Qs are in the same block, but seq_len mod window size might != 0.

    The Qs are the final tokens of Ks. the pivot_and_window_idx[-query_len] are Qs.

    """
    b, n_head, sq, hn = q.shape
    sk = k.shape[2]
    _b, n_piv = pivot_and_window_idx.shape
    pivot_and_window_idx_dummy = pivot_and_window_idx.view(b, 1, n_piv, 1).expand(b, n_head, n_piv, hn)
    pivot_k, pivot_v = torch.gather(k, 2, pivot_and_window_idx_dummy), torch.gather(v, 2, pivot_and_window_idx_dummy)
    attention_scores = torch.matmul(q / math.sqrt(hn), pivot_k.transpose(-1, -2))
    if sq > 1:
        query_part_scores = attention_scores[:, :, -sq:, -sq:]
        m = torch.ones((sq, sq), device=q.device, dtype=q.dtype) * -10000.0
        m.triu_(diagonal=1)
        query_part_scores += m
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
    context_layer = torch.matmul(attention_probs, pivot_v)
    return context_layer


def standard_attention(query_layer, key_layer, value_layer, attention_mask, attention_dropout=None):
    if len(attention_mask.shape) == 3:
        attention_mask = attention_mask.unsqueeze(1)
    attention_scores = torch.matmul(query_layer / math.sqrt(query_layer.shape[-1]), key_layer.transpose(-1, -2))
    attention_scores = torch.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs = attention_dropout(attention_probs)
    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer


class GPT2ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer for GPT2.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence length, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method, output_layer_init_method=None, query_window=128, key_window_times=6):
        super(GPT2ParallelSelfAttention, self).__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.query_window = query_window
        self.key_window_times = key_window_times
        self.query_key_value = ColumnParallelLinear(hidden_size, 3 * hidden_size, stride=3, gather_output=False, init_method=init_method)
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)
        self.dense = RowParallelLinear(hidden_size, hidden_size, input_is_parallel=True, init_method=output_layer_init_method)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)
        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ltor_mask, pivot_idx=None, is_sparse=0, mem=None):
        query_length = hidden_states.size(1)
        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            mixed_query_layer, mixed_key_layer, mixed_value_layer = split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            cat = torch.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            mixed_query_layer, mixed_key_layer, mixed_value_layer = split_tensor_along_last_dim(mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        if is_sparse == 1:
            context_layer = sparse_attention(query_layer, key_layer, value_layer, pivot_idx, ltor_mask, self.query_window, self.key_window_times, self.attention_dropout)
        elif is_sparse == 2:
            context_layer = sparse_attention_inference(query_layer, key_layer, value_layer, pivot_idx)
        else:
            context_layer = standard_attention(query_layer, key_layer, value_layer, ltor_mask, self.attention_dropout)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)
        output = self.output_dropout(output)
        return output


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class GPT2ParallelMLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob, init_method, output_layer_init_method=None):
        super(GPT2ParallelMLP, self).__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4 * hidden_size, gather_output=False, init_method=init_method)
        self.dense_4h_to_h = RowParallelLinear(4 * hidden_size, hidden_size, input_is_parallel=True, init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class GPT2ParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for GPT2.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, layernorm_epsilon, init_method, output_layer_init_method=None, query_window=128, key_window_times=6, scale_normalization=True):
        super(GPT2ParallelTransformerLayer, self).__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.attention = GPT2ParallelSelfAttention(hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method, output_layer_init_method=output_layer_init_method, query_window=query_window, key_window_times=key_window_times)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.scale_normalization = scale_normalization
        if scale_normalization:
            self.third_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
            self.fourth_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.mlp = GPT2ParallelMLP(hidden_size, output_dropout_prob, init_method, output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, ltor_mask, pivot_idx=None, is_sparse=0, mem=None):
        layernorm_output1 = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        attention_output = self.attention(layernorm_output1, ltor_mask, pivot_idx, is_sparse, mem)
        if self.scale_normalization:
            attention_output = self.third_layernorm(attention_output)
        layernorm_input = hidden_states + attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output)
        if self.scale_normalization:
            mlp_output = self.fourth_layernorm(mlp_output)
        output = layernorm_input + mlp_output
        return output


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_


class GPT2ParallelTransformer(torch.nn.Module):
    """GPT-2 transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """

    def __init__(self, num_layers, hidden_size, num_attention_heads, max_sequence_length, max_memory_length, embedding_dropout_prob, attention_dropout_prob, output_dropout_prob, checkpoint_activations, checkpoint_num_layers=1, layernorm_epsilon=1e-05, init_method_std=0.02, use_scaled_init_for_output_weights=True, query_window=128, key_window_times=6, num_pivot=768):
        super(GPT2ParallelTransformer, self).__init__()
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length
        self.max_sequence_length = max_sequence_length
        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(init_method_std, num_layers)
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        def get_layer(layer_id):
            return GPT2ParallelTransformerLayer(hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, layernorm_epsilon, unscaled_init_method(init_method_std), output_layer_init_method=output_layer_init_method, query_window=query_window, key_window_times=key_window_times, scale_normalization=True)
        self.query_window = query_window
        self.key_window_times = key_window_times
        self.num_pivot = num_pivot
        self.layers = torch.nn.ModuleList([get_layer(layer_id) for layer_id in range(num_layers)])
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint
        self.rmask = None

    def forward(self, hidden_states, position_ids, attention_mask, txt_indices_bool, img_indices_bool, is_sparse=0, *mems):
        batch_size, query_length = hidden_states.size()[:2]
        memory_length = mems[0].size(1) if mems else 0
        key_length = query_length + memory_length
        if isinstance(attention_mask, int) or attention_mask.numel() == 1:
            sep = attention_mask

            def build_mask_matrix(query_length, key_length, sep):
                m = torch.ones((1, query_length, key_length), device=hidden_states.device, dtype=hidden_states.dtype)
                assert query_length <= key_length
                m[0, :, -query_length:] = torch.tril(m[0, :, -query_length:])
                m[0, :, :sep + (key_length - query_length)] = 1
                m = m.unsqueeze(1)
                return m
            attention_mask = build_mask_matrix(query_length, key_length, sep)
        if is_sparse == 1 and self.rmask is None:
            w, times = self.query_window, self.key_window_times
            g = key_length // w
            tmp = torch.ones((g - times + 1, w, w), device=hidden_states.device, dtype=hidden_states.dtype)
            tmp = torch.tril(1 - torch.block_diag(*tmp))
            self.rmask = torch.nn.functional.pad(tmp, (0, (times - 1) * w, (times - 1) * w, 0))
        if is_sparse == 2:
            left_boundary = max(0, key_length - self.key_window_times * self.query_window)
            window_idx = torch.arange(left_boundary, key_length, device=hidden_states.device, dtype=torch.long).expand(batch_size, -1)
        elif is_sparse == 1:
            left_boundary = key_length
            num_pivot = self.num_pivot
        if is_sparse:
            img_indices = [img_indices_bool[i][:left_boundary].nonzero(as_tuple=False).view(-1) for i in range(batch_size)]
            txt_indices = [txt_indices_bool[i][:left_boundary].nonzero(as_tuple=False).view(-1) for i in range(batch_size)]
        if is_sparse == 2:
            ratio = self.num_pivot / self.max_sequence_length
            max_text_num = max(len(text_idx) for text_idx in txt_indices)
            num_pivot = max_text_num + int((left_boundary - max_text_num) * ratio)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)
        if self.max_memory_length > 0:
            mem_layers = [hidden_states.detach()]
        else:
            mem_layers = []

        def custom(start, end):

            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_, inputs = inputs[0], inputs[1:]
                if is_sparse > 0:
                    inputs, mems_ = inputs[:3], inputs[3:]
                else:
                    inputs, mems_ = inputs[:1], inputs[1:]
                for i, layer in enumerate(layers_):
                    mem_i_ = mems_[i] if mems_ else None
                    x_ = layer(x_, *inputs, mem=mem_i_)
                    if self.max_memory_length > 0:
                        mem_layers.append(x_.detach())
                return x_
            return custom_forward
        attention_mask_saved = attention_mask
        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                if is_sparse > 0:
                    pivot_idx = torch.stack([torch.cat((text_idx, img_indices[i][torch.tensor(random.sample(range(len(img_indices[i])), k=num_pivot - len(text_idx)), dtype=torch.long, device=text_idx.device)]), dim=0) for i, text_idx in enumerate(txt_indices)])
                    if is_sparse == 1:
                        assert key_length == query_length
                        b, s = batch_size, key_length
                        pivot_attention_mask = self.rmask.expand(b, s, s).gather(dim=-1, index=pivot_idx.unsqueeze(1).expand(b, s, self.num_pivot))
                        args = [hidden_states, pivot_attention_mask, pivot_idx, torch.tensor(is_sparse)]
                    elif is_sparse == 2:
                        pw_idx = torch.cat((pivot_idx, window_idx), dim=-1)
                        args = [hidden_states, attention_mask_saved, pw_idx, torch.tensor(is_sparse)]
                    else:
                        raise NotImplementedError
                else:
                    args = [hidden_states, attention_mask_saved]
                if mems:
                    args += mems[l:l + chunk_length]
                hidden_states = checkpoint(custom(l, l + chunk_length), *args)
                l += chunk_length
        else:
            assert is_sparse != 1, 'Please use checkpoint_activations for sparse attention training.'
            for i, layer in enumerate(self.layers):
                if is_sparse == 0:
                    args = [hidden_states, attention_mask_saved]
                elif is_sparse == 2:
                    pivot_idx = torch.stack([torch.cat((text_idx, img_indices[i][torch.tensor(random.sample(range(len(img_indices[i])), k=num_pivot - len(text_idx)), dtype=torch.long, device=text_idx.device)]), dim=0) for i, text_idx in enumerate(txt_indices)])
                    pw_idx = torch.cat((pivot_idx, window_idx), dim=-1)
                    args = [hidden_states, attention_mask_saved, pw_idx, torch.tensor(is_sparse)]
                mem_i = mems[i] if mems else None
                hidden_states = layer(*args, mem=mem_i)
                if self.max_memory_length > 0:
                    mem_layers.append(hidden_states.detach())
        output = self.final_layernorm(hidden_states)
        if self.max_memory_length > 0:
            mem_layers = self.update_mems(mem_layers, mems)
        return output, *mem_layers

    def update_mems(self, hiddens, mems):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = min(self.max_memory_length, memory_length + query_length)
        new_mems = []
        with torch.no_grad():
            for i in range(len(hiddens)):
                if new_memory_length <= query_length:
                    new_mems.append(hiddens[i][:, -new_memory_length:])
                else:
                    new_mems.append(torch.cat((mems[i][:, -new_memory_length + query_length:], hiddens[i]), dim=1))
        return new_mems


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    """
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if not torch.jit.is_scripting():
        if type(logits) is not Tensor and has_torch_function((logits,)):
            return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn('`eps` parameter is deprecated and has no effect.')
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret, index
    else:
        ret = y_soft
        index = y_soft.max(dim, keepdim=True)[1]
        return ret, index


class Quantize(nn.Module):

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-05):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed)
        torch.nn.init.xavier_uniform_(embed, gain=torch.nn.init.calculate_gain('tanh'))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward_(self, input, continuous_relax=False, temperature=1.0, hard=False):
        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        if not continuous_relax:
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        elif not hard:
            embed_soft, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=False)
            embed_ind = embed_ind.view(*input.shape[:-1])
            embed_soft = embed_soft.view(*input.shape[:-1], self.n_embed)
            quantize = embed_soft @ self.embed.transpose(0, 1)
        else:
            embed_onehot, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=True)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        if self.training and (continuous_relax and hard or not continuous_relax):
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        if not continuous_relax:
            diff = (quantize.detach() - input).pow(2).mean()
            quantize = input + (quantize - input).detach()
        else:
            qy = (-dist).softmax(-1)
            diff = torch.sum(qy * torch.log(qy * self.n_embed + 1e-20), dim=-1).mean()
            quantize = quantize
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):

    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(in_channel, channel, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, in_channel, 1))

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):

    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, embed_dim, n_embed, simple):
        super().__init__()
        if stride == 6:
            if simple:
                blocks = [nn.Conv2d(in_channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 4, stride=2, padding=1)]
            else:
                blocks = [nn.Conv2d(in_channel, channel // 4, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1)]
        elif stride == 4:
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 3, padding=1)]
        elif stride == 2:
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 2, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input).permute(0, 2, 3, 1)


class Decoder(nn.Module):

    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, simple):
        super().__init__()
        blocks = [nn.ConvTranspose2d(in_channel, channel, 4, stride=2, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4 and simple:
            blocks.extend([nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, out_channel, 1)])
        elif stride == 4:
            blocks.extend([nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(channel, channel // 2, 1), nn.ReLU(inplace=True), nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)])
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):

    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=1024, stride=4, simple=True, decay=0.99):
        super().__init__()
        if channel == 2048:
            n_res_block = 0
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride, embed_dim, n_embed, simple)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec = Decoder(in_channel=embed_dim, out_channel=in_channel, channel=channel, n_res_block=n_res_block, n_res_channel=n_res_channel, stride=stride - 2, simple=simple)

    def forward(self, input, continuous_relax=False, temperature=1.0, hard=False, KL=False):
        quant_t, diff, _ = self.encode(input, continuous_relax, temperature, hard, KL)
        dec = self.dec(quant_t)
        return dec, diff

    def encode(self, input, continuous_relax=False, temperature=1.0, hard=False, KL=False):
        logits = self.enc_b(input)
        quant_t, diff_t, id_t = self.quantize_t.forward_(logits, continuous_relax, temperature, hard)
        quant_t = quant_t.permute(0, 3, 1, 2)
        if not continuous_relax or KL:
            diff_t = diff_t.unsqueeze(0)
        else:
            diff_t = torch.zeros_like(diff_t).unsqueeze(0)
        return quant_t, diff_t, id_t

    def decode(self, code):
        return self.dec(code)

    def decode_code(self, code_t):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        dec = self.dec(quant_t)
        return dec


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'channel': 4, 'n_res_block': 4, 'n_res_channel': 4, 'stride': 1, 'simple': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FP16_Module,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (InceptionV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'in_channel': 4, 'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VQVAE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (tofp16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_THUDM_CogView(_paritybench_base):
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

