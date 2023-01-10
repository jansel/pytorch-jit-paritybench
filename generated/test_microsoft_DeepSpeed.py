import sys
_module = sys.modules[__name__]
del sys
benchmarks = _module
communication = _module
all_gather = _module
all_reduce = _module
all_to_all = _module
broadcast = _module
constants = _module
pt2pt = _module
run_all = _module
utils = _module
collect_results = _module
aio_bench_generate_param = _module
aio_bench_perf_sweep = _module
ds_aio_basic = _module
ds_aio_handle = _module
parse_aio_stats = _module
perf_sweep_utils = _module
test_ds_aio = _module
test_ds_aio_utils = _module
validate_async_io = _module
deepspeed = _module
accelerator = _module
abstract_accelerator = _module
cuda_accelerator = _module
real_accelerator = _module
autotuning = _module
autotuner = _module
config = _module
scheduler = _module
tuner = _module
base_tuner = _module
cost_model = _module
index_based_tuner = _module
model_based_tuner = _module
checkpoint = _module
deepspeed_checkpoint = _module
reshape_3d_utils = _module
reshape_meg_2d = _module
reshape_utils = _module
universal_checkpoint = _module
zero_checkpoint = _module
comm = _module
backend = _module
comm = _module
utils = _module
compression = _module
basic_layer = _module
compress = _module
helper = _module
utils = _module
elasticity = _module
elastic_agent = _module
utils = _module
env_report = _module
git_version_info = _module
inference = _module
config = _module
engine = _module
launcher = _module
launch = _module
multinode_runner = _module
runner = _module
model_implementations = _module
diffusers = _module
unet = _module
vae = _module
transformers = _module
clip_encoder = _module
ds_transformer = _module
module_inject = _module
inject = _module
layers = _module
load_checkpoint = _module
module_quantize = _module
replace_module = _module
replace_policy = _module
moe = _module
experts = _module
layer = _module
mappings = _module
sharded_moe = _module
utils = _module
monitor = _module
csv_monitor = _module
tensorboard = _module
utils = _module
wandb = _module
nebula = _module
constants = _module
ops = _module
adagrad = _module
cpu_adagrad = _module
adam = _module
cpu_adam = _module
fused_adam = _module
multi_tensor_apply = _module
aio = _module
lamb = _module
fused_lamb = _module
quantizer = _module
quantizer = _module
random_ltd = _module
dropping_utils = _module
sparse_attention = _module
bert_sparse_self_attention = _module
matmul = _module
softmax = _module
sparse_attention_utils = _module
sparse_self_attention = _module
sparsity_config = _module
trsrc = _module
transformer = _module
bias_add = _module
diffusers_2d_transformer = _module
diffusers_attention = _module
diffusers_transformer_block = _module
ds_attention = _module
ds_mlp = _module
moe_inference = _module
op_binding = _module
base = _module
linear = _module
qkv_gemm = _module
softmax = _module
softmax_context = _module
vector_matmul = _module
triton_ops = _module
transformer = _module
pipe = _module
profiling = _module
flops_profiler = _module
profiler = _module
runtime = _module
activation_checkpointing = _module
checkpointing = _module
bf16_optimizer = _module
checkpoint_engine = _module
nebula_checkpoint_engine = _module
torch_checkpoint_engine = _module
coalesced_collectives = _module
mpi = _module
nccl = _module
cupy = _module
config = _module
config_utils = _module
data_pipeline = _module
curriculum_scheduler = _module
data_routing = _module
basic_layer = _module
utils = _module
data_sampling = _module
data_analyzer = _module
data_sampler = _module
indexed_dataset = _module
dataloader = _module
eigenvalue = _module
engine = _module
fp16 = _module
fused_optimizer = _module
loss_scaler = _module
onebit = _module
adam = _module
lamb = _module
zoadam = _module
unfused_optimizer = _module
lr_schedules = _module
engine = _module
module = _module
p2p = _module
schedule = _module
topology = _module
progressive_layer_drop = _module
quantize = _module
sparse_tensor = _module
state_dict_factory = _module
swap_tensor = _module
aio_config = _module
async_swapper = _module
optimizer_utils = _module
partitioned_optimizer_swapper = _module
partitioned_param_swapper = _module
pipelined_optimizer_swapper = _module
utils = _module
utils = _module
weight_quantizer = _module
zero = _module
contiguous_memory_allocator = _module
linear = _module
offload_config = _module
parameter_offload = _module
partition_parameters = _module
partitioned_param_coordinator = _module
stage3 = _module
stage_1_and_2 = _module
test = _module
tiling = _module
utils = _module
comms_logging = _module
debug = _module
exceptions = _module
groups = _module
init_on_device = _module
logging = _module
mixed_precision_linkage = _module
nvtx = _module
tensor_fragment = _module
timer = _module
types = _module
zero_to_fp32 = _module
conf = _module
op_builder = _module
all_ops = _module
async_io = _module
builder = _module
builder_names = _module
cpu_adagrad = _module
cpu_adam = _module
random_ltd = _module
sparse_attn = _module
spatial_inference = _module
stochastic_transformer = _module
transformer = _module
transformer_inference = _module
bump_patch_version = _module
setup = _module
flatten_bench = _module
unflatten_bench = _module
conftest = _module
test_simple = _module
BingBertSquad_run_func_test = _module
BingBertSquad_test_common = _module
BingBertSquad = _module
test_e2e_squad = _module
Megatron_GPT2 = _module
run_checkpoint_test = _module
run_func_test = _module
run_perf_baseline = _module
run_perf_test = _module
test_common = _module
run_sanity_check = _module
test_mpi_backend = _module
test_mpi_perf = _module
test_nccl_backend = _module
test_nccl_perf = _module
adam_test = _module
adam_test1 = _module
stage3_test = _module
test = _module
test_model = _module
unit = _module
alexnet_model = _module
test_autotuning = _module
common = _module
test_latest_checkpoint = _module
test_lr_scheduler = _module
test_moe_checkpoint = _module
test_other_optimizer = _module
test_pipeline = _module
test_reshape_checkpoint = _module
test_sparse = _module
test_tag_validation = _module
test_zero_optimizer = _module
test_dist = _module
common = _module
test_compression = _module
test_elastic = _module
test_checkpoint_sharding = _module
test_inference = _module
test_inference_config = _module
test_model_profiling = _module
test_ds_arguments = _module
test_multinode_runner = _module
test_run = _module
megatron_model = _module
test_configurable_parallel_mp = _module
test_configurable_parallel_pp = _module
modeling = _module
modelingpreln = _module
test_moe = _module
test_moe_tp = _module
test_monitor = _module
multi_output_model = _module
test_cpu_adagrad = _module
test_adamw = _module
test_cpu_adam = _module
test_aio = _module
test_cuda_backward = _module
test_cuda_forward = _module
test_dequantize = _module
test_fake_quantization = _module
test_quantize = _module
test_sparse_attention = _module
test_nhwc_bias_add = _module
test_bias_add = _module
test_bias_geglu = _module
test_bias_gelu = _module
test_bias_relu = _module
test_layer_norm = _module
test_moe_res_matmult = _module
test_residual_add = _module
test_pipe_module = _module
test_flops_profiler = _module
test_activation_checkpointing = _module
test_coalesced_collectives = _module
test_onebit = _module
test_bf16 = _module
test_dynamic_loss_scale = _module
test_fp16 = _module
test_pipe = _module
test_pipe_schedule = _module
test_topology = _module
test_averaging_sparse_gradients = _module
test_csr = _module
test_sparse_grads = _module
test_autocast = _module
test_data = _module
test_data_efficiency = _module
test_ds_config_dict = _module
test_ds_config_model = _module
test_ds_initialize = _module
test_lr_schedulers = _module
test_multi_output_model = _module
test_pld = _module
test_runtime_utils = _module
test_partition = _module
test_ignore_unused_parameters = _module
test_zero = _module
test_zero_config = _module
test_zero_context = _module
test_zero_tiled = _module
simple_model = _module
util = _module
test_get_optim_files = _module
test_groups = _module
test_init_on_device = _module

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


import time


import torch


import math


import types


from typing import Optional


from typing import Union


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


import torch.cuda


from typing import Dict


from collections import OrderedDict


from enum import Enum


import inspect


from torch import nn


from torch.nn import init


import re


from torch import autograd


from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent


from typing import Any


from typing import Tuple


from torch.distributed.elastic.agent.server.api import log


from torch.distributed.elastic.agent.server.api import _get_socket_with_port


from torch.distributed.elastic.metrics import put_metric


from torch.distributed.elastic.agent.server.api import RunResult


from torch.distributed.elastic.agent.server.api import WorkerGroup


from torch.distributed.elastic.agent.server.api import WorkerSpec


from torch.distributed.elastic.agent.server.api import WorkerState


from torch.distributed import Store


from torch.distributed.elastic.multiprocessing import start_processes


from torch.distributed.elastic.utils import macros


import copy


from torch.nn.modules import Module


from collections import defaultdict


import collections


from copy import deepcopy


import torch.nn as nn


from torch.nn import functional as F


from torch.nn.parameter import Parameter


from abc import ABC


import typing


from typing import Callable


from typing import TYPE_CHECKING


from torch import Tensor


from torch.nn import Module


import torch.nn.functional as F


from typing import List


from torch import distributed as dist


import random


from torch.autograd import Function


from functools import partial


import numpy as np


from torch import _C


from torch.cuda import _lazy_call


from torch.cuda import device as device_ctx_manager


from torch.distributed import ProcessGroup


import torch.nn.functional


from torch.utils.dlpack import to_dlpack


from torch.utils.dlpack import from_dlpack


from torch.utils.data import BatchSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from functools import lru_cache


from itertools import accumulate


from torch.utils.data import RandomSampler


from torch.utils.data.distributed import DistributedSampler


import logging


from collections import deque


from typing import Iterable


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from types import MethodType


import re as regex


from abc import abstractmethod


from collections.abc import Iterable


from math import sqrt


from math import floor


from torch._six import inf


from numpy import prod


from torch.nn.modules.module import Module


from torch.cuda import Stream


import functools


import itertools


from torch.nn import Parameter


from collections import UserDict


from typing import Deque


from typing import Set


from torch.cuda import Event


from numpy import mean


import warnings


from torch.utils.data import Dataset


import numbers


import torch.multiprocessing as mp


from torch.multiprocessing import Process


from torch.nn import CrossEntropyLoss


from torch.utils import checkpoint


import torch.nn.init as init


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim.lr_scheduler import LambdaLR


from torch.nn import Linear


from torch.nn.modules.container import ModuleList


from torch.nn.modules.loss import L1Loss


from types import SimpleNamespace


class AsymQuantizer(torch.autograd.Function):
    """
    Asymmetric quantization
    """

    @staticmethod
    def forward(ctx, input, num_bits, min_value=None, max_value=None, num_groups=1):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input which needs to be quantized
            num_bits (int, >=4)
                Number of bits to use for quantization
            min_value/max_vlue (torch.FloatTensor)
                Used for static activation quantization
            num_groups (int)
                How many groups to partition the quantization into
        Returns:
            quantized_input (`torch.FloatTensor`)
                Quantized input
        """
        assert min_value is None and max_value is None or min_value is not None and max_value is not None and num_groups == 1
        q_range = 2 ** num_bits
        input_shape = input.shape
        if min_value is None:
            input = input.reshape(num_groups, -1)
            min_value = input.amin(dim=-1, keepdim=True)
            max_value = input.amax(dim=-1, keepdim=True)
        scale = (max_value - min_value) / q_range
        zero_point = (min_value / scale).round() * scale
        output = ((input - zero_point) / scale).round().clamp(0, q_range - 1) * scale + zero_point
        output = output.reshape(input_shape).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class SymQuantizer(torch.autograd.Function):
    """
    Symmetric quantization
    """

    @staticmethod
    def forward(ctx, input, num_bits, min_value=None, max_value=None, num_groups=1):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input which needs to be quantized
            num_bits (int, >=4)
                Number of bits to use for quantization
            min_value/max_vlue (torch.FloatTensor)
                Used for static activation quantization
            num_groups (int)
                How many groups to partition the quantization into
        Returns:
            quantized_input (`torch.FloatTensor`)
                Quantized input
        """
        assert min_value is None and max_value is None or min_value is not None and max_value is not None and num_groups == 1
        q_range = 2 ** num_bits
        input_shape = input.shape
        if min_value is None:
            input = input.reshape(num_groups, -1)
            max_input = torch.amax(torch.abs(input), dim=-1).view(num_groups, -1)
        else:
            max_input = torch.max(min_value.abs(), max_value).view(-1)
        scale = 2 * max_input / q_range
        output = (input / scale).round().clamp(-q_range // 2, q_range // 2 - 1) * scale
        output = output.reshape(input_shape).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class QuantAct(nn.Module):
    """
    Class to quantize given activations. Note that when using this function, the input acttivation quantization range will be fixed for all
    tokens/images for inference. This generally will affect some accuracy but achieve better latency performance.
    Parameters:
    ----------
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    quant_mode : str, default 'symmetric'
    """

    def __init__(self, act_range_momentum=0.95, quant_mode='symmetric'):
        super(QuantAct, self).__init__()
        self.act_range_momentum = act_range_momentum
        self.quant_mode = quant_mode
        if quant_mode == 'symmetric':
            self.act_function = SymQuantizer.apply
        else:
            self.act_function = AsymQuantizer.apply
        self.register_buffer('x_min_max', torch.zeros(2))

    def forward(self, x, num_bits, *args):
        """
        x: the activation that we need to quantize
        num_bits: the number of bits we need to quantize the activation to
        *args: some extra arguments that are useless but needed for align with the interface of other quantization functions
        """
        if self.training:
            x_min = x.data.min()
            x_max = x.data.max()
            if self.x_min_max[0] == self.x_min_max[1]:
                self.x_min_max[0] = x_min
                self.x_min_max[1] = x_max
            self.x_min_max[0] = self.x_min_max[0] * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
            self.x_min_max[1] = self.x_min_max[1] * self.act_range_momentum + x_max * (1 - self.act_range_momentum)
        x_q = self.act_function(x, num_bits, self.x_min_max[0], self.x_min_max[1])
        return x_q


class BinaryQuantizer(torch.autograd.Function):
    """
    Binary quantization
    """

    @staticmethod
    def forward(ctx, input, num_bits, min_value=None, max_value=None, num_groups=1):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input which needs to be quantized
            num_bits (int)
                Dummy variable
            min_value/max_vlue (torch.FloatTensor)
                Used for static activation quantization; for now they are dummy variable
            num_groups (int)
                How many groups to partition the quantization into
        Returns:
            quantized_input (`torch.FloatTensor`)
                Quantized input
        """
        assert min_value is None and max_value is None
        input_flat = input.reshape(num_groups, -1)
        n = input_flat.shape[1]
        m = input_flat.norm(p=1, dim=1, keepdim=True).div(n)
        output = input_flat.sign().mul(m)
        output = output.reshape(input.shape).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class TernaryQuantizer(torch.autograd.Function):
    """
    Ternary quantization
    """

    @staticmethod
    def forward(ctx, input, num_bits, min_value=None, max_value=None, num_groups=1):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input which needs to be quantized
            num_bits (int)
                Dummy variable
            min_value/max_vlue (torch.FloatTensor)
                Used for static activation quantization; for now they are dummy variable
            num_groups (int)
                How many groups to partition the quantization into
        Returns:
            quantized_input (`torch.FloatTensor`)
                Quantized input
        """
        assert min_value is None and max_value is None
        input_flat = input.reshape(num_groups, -1)
        n = input_flat.shape[1]
        m = input_flat.norm(p=1, dim=1).div(n)
        thres = (0.7 * m).view(-1, 1)
        pos = (input_flat > thres).type(input.type())
        neg = (input_flat < -thres).type(input.type())
        mask = (input_flat.abs() > thres).type(input.type())
        alpha = ((mask * input_flat).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
        output = alpha * pos - alpha * neg
        output = output.reshape(input.shape).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


logger = logging.getLogger(__name__)


class Embedding_Compress(nn.Embedding):

    def __init__(self, *kargs):
        super(Embedding_Compress, self).__init__(*kargs)
        self.weight.start_bits = None
        self.weight.target_bits = None
        self.weight.q_period = None
        self.weight_quantization_enabled_in_forward = False
        self.weight_quantization_enabled = False

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, weight_quantization={}'.format(self.num_embeddings, self.embedding_dim, self.weight.target_bits)

    def enable_weight_quantization(self, start_bits, target_bits, quantization_period, weight_quantization_enabled_in_forward, quantization_type, num_groups):
        self.weight.start_bits = start_bits
        self.weight.target_bits = target_bits
        self.weight.q_period = quantization_period
        self.weight_quantization_enabled_in_forward = weight_quantization_enabled_in_forward
        if self.weight_quantization_enabled_in_forward:
            logger.warning('************ A lot of MoQ features are not supported in quantize_weight_in_forward mode, please consider to use DS-FP16 optimizer************')
            if self.weight.target_bits >= 3:
                if quantization_type == 'symmetric':
                    self.weight_quantizer = SymQuantizer.apply
                else:
                    self.weight_quantizer = AsymQuantizer.apply
            elif self.weight.target_bits == 2:
                assert quantization_type == 'symmetric', 'Only symmetric quantization is supported for ternary weight quantization'
                self.weight_quantizer = TernaryQuantizer.apply
            elif self.weight.target_bits == 1:
                assert quantization_type == 'symmetric', 'Only symmetric quantization is supported for binary weight quantization'
                self.weight_quantizer = BinaryQuantizer.apply
            self.weight_quantize_num_groups = self.weight.size(0)

    def fix_weight_quantization(self):
        self.weight.data = self.weight_quantizer(self.weight, self.weight.target_bits, None, None, self.weight_quantize_num_groups).data
        self.weight_quantization_enabled_in_forward = False
        return None

    def forward(self, input):
        if self.weight_quantization_enabled_in_forward and self.weight_quantization_enabled:
            weight = self.weight_quantizer(self.weight, self.weight.target_bits, None, None, self.weight_quantize_num_groups)
        else:
            weight = self.weight
        out = nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out


class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.
    Implementation is inspired from:
        https://github.com/yaozhewei/MLPruning
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float, sigmoid: bool):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
            sigmoid (`bool`)
                Whether to apply a sigmoid on the threshold
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        if sigmoid:
            threshold = torch.sigmoid(threshold).item()
        ctx.sigmoid = sigmoid
        mask = inputs.clone()
        _, idx = inputs.flatten().sort(descending=True)
        j = math.ceil(threshold * inputs.numel())
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0.0
        flat_out[idx[:j]] = 1.0
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        mask, = ctx.saved_tensors
        if ctx.sigmoid:
            return gradOutput.clone(), (gradOutput * mask).sum().view(-1), None
        else:
            return gradOutput.clone(), None, None


class LinearLayer_Compress(nn.Linear):
    """
    Linear layer with compression.
    """

    def __init__(self, *kargs, bias=True):
        super(LinearLayer_Compress, self).__init__(*kargs, bias=bias)
        self.sparse_pruning_method = None
        self.row_pruning_method = None
        self.head_pruning_method = None
        self.activation_quantization_method = None
        self.weight.start_bits = None
        self.weight.target_bits = None
        self.weight.q_period = None
        self.weight_quantization_enabled_in_forward = False
        self.weight_quantization_enabled = False
        self.sparse_pruning_enabled = False
        self.row_pruning_enabled = False
        self.head_pruning_enabled = False
        self.activation_quantization_enabled = False

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, sparse pruning={}, row pruning={}, head pruning={}, activation quantization={}, weight_quantization={}'.format(self.in_features, self.out_features, self.bias is not None, self.sparse_pruning_method is not None, self.row_pruning_method is not None, self.head_pruning_method is not None, self.activation_quantization_method is not None, self.weight.target_bits)

    def enable_sparse_pruning(self, ratio, method):
        self.sparse_pruning_ratio = ratio
        self.sparse_pruning_method = method
        if method == 'l1':
            weight_norm = torch.abs(self.weight.data)
            mask = TopKBinarizer.apply(weight_norm, self.sparse_pruning_ratio, False)
            mask = mask.view(self.weight.size())
            mask = mask
        elif method == 'topk':
            self.sparse_mask_scores = nn.Parameter(torch.Tensor(self.weight.size()))
            self.sparse_mask_scores.data = self.sparse_mask_scores.data
            init.kaiming_uniform_(self.sparse_mask_scores, a=math.sqrt(5))
            mask = None
        else:
            raise NotImplementedError
        self.register_buffer('sparse_pruning_mask', mask)

    def enable_row_pruning(self, ratio, method):
        self.row_pruning_ratio = ratio
        self.row_pruning_method = method
        if method == 'l1':
            weight_norm = torch.norm(self.weight.data, p=1, dim=1)
            mask = TopKBinarizer.apply(weight_norm, self.row_pruning_ratio, False)
            mask = mask.view(-1, 1)
            mask = mask
        elif method == 'topk':
            self.row_mask_scores = nn.Parameter(torch.Tensor(self.weight.size(0), 1))
            self.row_mask_scores.data = self.row_mask_scores.data
            init.kaiming_uniform_(self.row_mask_scores, a=math.sqrt(5))
            mask = None
        else:
            raise NotImplementedError
        self.register_buffer('row_pruning_mask', mask)

    def enable_head_pruning(self, ratio, method, num_heads):
        self.num_heads = num_heads
        self.head_pruning_ratio = ratio
        self.head_pruning_method = method
        if method not in ['topk']:
            raise NotImplementedError
        else:
            self.head_pruning_ratio = ratio
            self.head_pruning_scores = nn.Parameter(torch.Tensor(1, self.num_heads))
            self.head_pruning_scores.data = self.head_pruning_scores.data
            init.kaiming_uniform_(self.head_pruning_scores, a=math.sqrt(5))

    def fix_sparse_pruning_helper(self):
        mask = self.get_mask(pruning_type='sparse')
        self.weight.data = self.weight.data * mask
        del self.sparse_pruning_mask
        if self.sparse_pruning_method == 'topk':
            del self.sparse_mask_scores
        self.sparse_pruning_method = None
        self.sparse_pruning_enabled = False
        return None

    def fix_row_col_pruning_helper(self, mask=None, dim_reduction=False):
        if mask is None:
            mask = self.get_mask(pruning_type='row').bool()
            if dim_reduction:
                start_bits = self.weight.start_bits
                target_bits = self.weight.target_bits
                q_period = self.weight.q_period
                self.weight = nn.Parameter(self.weight.data[mask.view(-1), :])
                self.weight.start_bits = start_bits
                self.weight.target_bits = target_bits
                self.weight.q_period = q_period
                if self.bias is not None:
                    self.bias = nn.Parameter(self.bias.data[mask.view(-1)])
                self.out_features = self.weight.size(0)
            else:
                self.weight.data = self.weight.data * mask.view(-1, 1)
                if self.bias is not None:
                    self.bias.data = self.bias.data * mask.view(-1)
            del self.row_pruning_mask
            if self.row_pruning_method == 'topk':
                del self.row_mask_scores
            self.row_pruning_method = None
        else:
            start_bits = self.weight.start_bits
            target_bits = self.weight.target_bits
            q_period = self.weight.q_period
            self.weight = nn.Parameter(self.weight.data[:, mask.view(-1)])
            self.weight.start_bits = start_bits
            self.weight.target_bits = target_bits
            self.weight.q_period = q_period
            self.in_features = self.weight.size(1)
            mask = None
        self.row_pruning_enabled = False
        return mask

    def fix_head_pruning_helper(self, mask=None, num_heads=None, dim_reduction=False):
        num_heads = num_heads if num_heads else self.num_heads
        if mask is None:
            if self.head_pruning_method == 'topk':
                mask = self.get_mask(pruning_type='head').bool()
                if dim_reduction:
                    shape = self.weight.size(0)
                    start_bits = self.weight.start_bits
                    target_bits = self.weight.target_bits
                    q_period = self.weight.q_period
                    self.weight = nn.Parameter(self.weight.data.t().reshape(num_heads, -1)[mask.view(-1), :].reshape(-1, shape).t())
                    self.weight.start_bits = start_bits
                    self.weight.target_bits = target_bits
                    self.weight.q_period = q_period
                else:
                    shape = self.weight.size()
                    self.weight.data = (self.weight.data.t().reshape(self.num_heads, -1) * mask.view(-1, 1)).reshape(shape[1], shape[0]).t()
                if self.head_pruning_method == 'topk':
                    del self.head_pruning_scores
                self.head_pruning_method = None
            else:
                raise NotImplementedError
        else:
            start_bits = self.weight.start_bits
            target_bits = self.weight.target_bits
            q_period = self.weight.q_period
            shape = self.weight.size(1)
            self.weight = nn.Parameter(self.weight.data.reshape(num_heads, -1)[mask.view(-1), :].reshape(-1, shape))
            self.weight.start_bits = start_bits
            self.weight.target_bits = target_bits
            self.weight.q_period = q_period
            if self.bias is not None:
                self.bias = nn.Parameter(self.bias.data.reshape(num_heads, -1)[mask.view(-1), :].reshape(-1))
        self.head_pruning_enabled = False
        return mask

    def get_mask(self, pruning_type='row'):
        if pruning_type == 'sparse':
            if self.sparse_pruning_method == 'l1':
                return self.sparse_pruning_mask
            elif self.sparse_pruning_method == 'topk':
                return TopKBinarizer.apply(self.sparse_mask_scores, self.sparse_pruning_ratio, False)
            else:
                raise NotImplementedError
        if pruning_type == 'row':
            if self.row_pruning_method == 'l1':
                return self.row_pruning_mask
            elif self.row_pruning_method == 'topk':
                return TopKBinarizer.apply(self.row_mask_scores, self.row_pruning_ratio, False)
            else:
                raise NotImplementedError
        elif pruning_type == 'head':
            if self.head_pruning_method == 'topk':
                return TopKBinarizer.apply(self.head_pruning_scores, self.head_pruning_ratio, False)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def enable_weight_quantization(self, start_bits, target_bits, quantization_period, weight_quantization_enabled_in_forward, quantization_type, num_groups):
        self.weight.start_bits = start_bits
        self.weight.target_bits = target_bits
        self.weight.q_period = quantization_period
        self.weight_quantization_enabled_in_forward = weight_quantization_enabled_in_forward
        if self.weight_quantization_enabled_in_forward:
            logger.warning('************ A lot of MoQ features are not supported in quantize_weight_in_forward mode, please consider to use DS-FP16 optimizer************')
            if self.weight.target_bits >= 3:
                if quantization_type == 'symmetric':
                    self.weight_quantizer = SymQuantizer.apply
                else:
                    self.weight_quantizer = AsymQuantizer.apply
            elif self.weight.target_bits == 2:
                assert quantization_type == 'symmetric', 'Only symmetric quantization is supported for ternary weight quantization'
                self.weight_quantizer = TernaryQuantizer.apply
            elif self.weight.target_bits == 1:
                assert quantization_type == 'symmetric', 'Only symmetric quantization is supported for binary weight quantization'
                self.weight_quantizer = BinaryQuantizer.apply
            self.weight_quantize_num_groups = num_groups

    def fix_weight_quantization(self):
        self.weight.data = self.weight_quantizer(self.weight, self.weight.target_bits, None, None, self.weight_quantize_num_groups).data
        self.weight_quantization_enabled_in_forward = False
        return None

    def enable_activation_quantization(self, bits, quantization_type, range_calibration):
        assert bits in [4, 8], 'Only 4/8 bits activation quantization are supported for now'
        self.activation_quantization_bits = bits
        self.activation_quantization_method = f'{quantization_type}_{range_calibration}'
        if range_calibration == 'static':
            self.activation_quantizer = QuantAct(quant_mode=quantization_type)
        elif quantization_type == 'symmetric':
            self.activation_quantizer = SymQuantizer.apply
        else:
            self.activation_quantizer = AsymQuantizer.apply

    def head_pruning_reshape(self, w, mask):
        shape = w.shape
        return (w.t().reshape(self.num_heads, -1) * mask.view(-1, 1)).reshape(shape[1], shape[0]).t()

    def forward(self, input, skip_bias_add=False):
        if self.weight_quantization_enabled_in_forward and self.weight_quantization_enabled:
            weight = self.weight_quantizer(self.weight, self.weight.target_bits, None, None, self.weight_quantize_num_groups)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        if self.sparse_pruning_enabled and self.sparse_pruning_method:
            mask = self.get_mask(pruning_type='sparse')
            weight = weight * mask.view(self.weight.size())
        if self.row_pruning_enabled and self.row_pruning_method:
            mask = self.get_mask(pruning_type='row')
            weight = weight * mask.view(-1, 1)
            if bias is not None:
                bias = bias * mask.view(-1)
        if self.head_pruning_enabled and self.head_pruning_method:
            mask = self.get_mask(pruning_type='head')
            weight = self.head_pruning_reshape(weight, mask)
        if self.activation_quantization_enabled:
            if 'dynamic' in self.activation_quantization_method:
                num_groups = input.numel() // input.size(-1)
            else:
                num_groups = 1
            input = self.activation_quantizer(input, self.activation_quantization_bits, None, None, num_groups)
        if skip_bias_add:
            output = nn.functional.linear(input, weight, None)
            return output, bias
        else:
            output = nn.functional.linear(input, weight, bias)
            return output


class Conv2dLayer_Compress(nn.Conv2d):
    """
    Conv2D layer with compression.
    """

    def __init__(self, *kargs):
        super(Conv2dLayer_Compress, self).__init__(*kargs)
        self.sparse_pruning_method = None
        self.channel_pruning_method = None
        self.activation_quantization_method = None
        self.weight.start_bits = None
        self.weight.target_bits = None
        self.weight.q_period = None
        self.weight_quantization_enabled_in_forward = False
        self.sparse_pruning_enabled = False
        self.channel_pruning_enabled = False
        self.activation_quantization_enabled = False

    def __repr__(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        output = s.format(**self.__dict__)
        return output + ' sparse pruning={}, channel pruning={}, activation quantization={}, weight_quantization={}'.format(self.sparse_pruning_method is not None, self.channel_pruning_method is not None, self.activation_quantization_method is not None, self.weight.target_bits)

    def enable_sparse_pruning(self, ratio, method):
        self.sparse_pruning_ratio = ratio
        self.sparse_pruning_method = method
        if method == 'l1':
            weight_norm = torch.abs(self.weight.data)
            mask = TopKBinarizer.apply(weight_norm, self.sparse_pruning_ratio, False)
            mask = mask.view(self.weight.size())
            mask = mask
        elif method == 'topk':
            self.sparse_mask_scores = nn.Parameter(torch.Tensor(self.weight.size()))
            self.sparse_mask_scores.data = self.sparse_mask_scores.data
            init.kaiming_uniform_(self.sparse_mask_scores, a=math.sqrt(5))
            mask = None
        else:
            raise NotImplementedError
        self.register_buffer('sparse_pruning_mask', mask)

    def enable_channel_pruning(self, ratio, method):
        self.channel_pruning_ratio = ratio
        self.channel_pruning_method = method
        if method == 'l1':
            weight_norm = torch.norm(self.weight.data, p=1, dim=[1, 2, 3])
            mask = TopKBinarizer.apply(weight_norm, self.channel_pruning_ratio, False)
            mask = mask.view(-1, 1, 1, 1)
            mask = mask
        elif method == 'topk':
            self.channel_mask_scores = nn.Parameter(torch.Tensor(self.weight.size(0), 1, 1, 1))
            self.channel_mask_scores.data = self.channel_mask_scores.data
            init.kaiming_uniform_(self.channel_mask_scores, a=math.sqrt(5))
            mask = None
        else:
            raise NotImplementedError
        self.register_buffer('channel_pruning_mask', mask)

    def fix_sparse_pruning_helper(self):
        mask = self.get_mask(pruning_type='sparse')
        self.weight.data = self.weight.data * mask
        del self.sparse_pruning_mask
        if self.sparse_pruning_method == 'topk':
            del self.sparse_mask_scores
        self.sparse_pruning_method = None
        self.sparse_pruning_enabled = False
        return None

    def fix_channel_pruning_helper(self, mask=None, dim_reduction=False):
        if mask is None:
            if self.channel_pruning_method in ['l1', 'topk']:
                mask = self.get_mask(pruning_type='channel').bool()
                if dim_reduction:
                    start_bits = self.weight.start_bits
                    target_bits = self.weight.target_bits
                    q_period = self.weight.q_period
                    self.weight = nn.Parameter(self.weight.data[mask.view(-1), ...])
                    self.weight.start_bits = start_bits
                    self.weight.target_bits = target_bits
                    self.weight.q_period = q_period
                    if self.bias is not None:
                        self.bias = nn.Parameter(self.bias.data[mask.view(-1)])
                else:
                    self.weight.data = self.weight.data * mask.view(-1, 1, 1, 1)
                    if self.bias is not None:
                        self.bias.data = self.bias.data * mask.view(-1)
                del self.channel_pruning_mask
                if self.channel_pruning_method == 'topk':
                    del self.channel_mask_scores
                self.channel_pruning_method = None
            else:
                raise NotImplementedError
        else:
            start_bits = self.weight.start_bits
            target_bits = self.weight.target_bits
            q_period = self.weight.q_period
            self.weight = nn.Parameter(self.weight.data[:, mask.view(-1), ...])
            self.weight.start_bits = start_bits
            self.weight.target_bits = target_bits
            self.weight.q_period = q_period
            mask = None
        self.channel_pruning_enabled = False
        return mask

    def get_mask(self, pruning_type='sparse'):
        if pruning_type == 'sparse':
            if self.sparse_pruning_method == 'l1':
                return self.sparse_pruning_mask
            elif self.sparse_pruning_method == 'topk':
                return TopKBinarizer.apply(self.sparse_mask_scores, self.sparse_pruning_ratio, False)
            else:
                raise NotImplementedError
        elif pruning_type == 'channel':
            if self.channel_pruning_method == 'l1':
                return self.channel_pruning_mask
            elif self.channel_pruning_method == 'topk':
                return TopKBinarizer.apply(self.channel_mask_scores, self.channel_pruning_ratio, False)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def fix_weight_quantization(self):
        self.weight.data = self.weight_quantizer(self.weight, self.weight.target_bits, None, None, self.weight_quantize_num_groups).data
        self.weight_quantization_enabled_in_forward = False
        return None

    def enable_weight_quantization(self, start_bits, target_bits, quantization_period, weight_quantization_enabled_in_forward, quantization_type, num_groups):
        self.weight.start_bits = start_bits
        self.weight.target_bits = target_bits
        self.weight.q_period = quantization_period
        self.weight_quantization_enabled_in_forward = weight_quantization_enabled_in_forward
        if self.weight_quantization_enabled_in_forward:
            assert self.weight.target_bits >= 4, 'Only >=4 bits weight quantization are supported during forward pass for now'
            logger.warning('************ A lot of MoQ features are not supported in quantize_weight_in_forward mode, please consider to use DS-FP16 optimizer************')
            if quantization_type == 'symmetric':
                self.weight_quantizer = SymQuantizer.apply
            else:
                self.weight_quantizer = AsymQuantizer.apply
            self.weight_quantize_num_groups = num_groups

    def enable_activation_quantization(self, bits, quantization_type, range_calibration):
        assert bits in [4, 8], 'Only 4/8 bits activation quantization are supported for now'
        self.activation_quantization_bits = bits
        self.activation_quantization_method = f'{quantization_type}_{range_calibration}'
        if range_calibration == 'static':
            self.activation_quantizer = QuantAct(quant_mode=quantization_type)
        elif quantization_type == 'symmetric':
            self.activation_quantizer = SymQuantizer.apply
        else:
            self.activation_quantizer = AsymQuantizer.apply

    def forward(self, input):
        if self.weight_quantization_enabled_in_forward and self.weight_quantization_enabled:
            weight = self.weight_quantizer(self.weight, self.weight.target_bits, None, None, self.weight_quantize_num_groups)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        if self.sparse_pruning_enabled and self.sparse_pruning_method:
            mask = self.get_mask(pruning_type='sparse')
            weight = weight * mask.view(self.weight.size())
        if self.channel_pruning_enabled:
            mask = self.get_mask(pruning_type='channel')
            weight = weight * mask.view(-1, 1, 1, 1)
            if bias is not None:
                bias = bias * mask.view(-1)
        if self.activation_quantization_enabled:
            if 'dynamic' in self.activation_quantization_method:
                num_groups = input.numel() // input[0].numel()
            else:
                num_groups = 1
            input = self.activation_quantizer(input, self.activation_quantization_bits, None, None, num_groups)
        return nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class BNLayer_Compress(nn.BatchNorm2d):

    def fix_channel_pruning_helper(self, mask, dim_reduction=True):
        self.weight = nn.Parameter(self.weight.data[mask.view(-1)])
        self.bias = nn.Parameter(self.bias.data[mask.view(-1)])
        self.running_mean = self.running_mean[mask.view(-1)]
        self.running_var = self.running_var[mask.view(-1)]


g_mpu = None


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = g_mpu.get_model_parallel_group()
    if dist.get_world_size(group=group) == 1:
        return input_
    dist.all_reduce(input_, group=group)
    return input_


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
    group = g_mpu.get_model_parallel_group()
    if dist.get_world_size(group=group) == 1:
        return input_
    last_dim = input_.dim() - 1
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    dist.all_gather(tensor_list, input_, group=group)
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


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


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = g_mpu.get_model_parallel_group()
    if dist.get_world_size(group=group) == 1:
        return input_
    world_size = dist.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = dist.get_rank(group=group)
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


class ColumnParallelLinear_Compress(LinearLayer_Compress):

    def __init__(self, mpu, input_size, output_size, bias=True, gather_output=True, skip_bias_add=False):
        global g_mpu
        g_mpu = mpu
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        world_size = mpu.get_model_parallel_world_size()
        assert output_size % world_size == 0
        self.output_size_per_partition = output_size // world_size
        super(ColumnParallelLinear_Compress, self).__init__(self.input_size, self.output_size_per_partition, bias=bias)

    def forward(self, input_):
        input_parallel = copy_to_model_parallel_region(input_)
        if self.skip_bias_add:
            output_parallel, bias = super().forward(input_parallel, True)
        else:
            output_parallel = super().forward(input_parallel)
            bias = None
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output, bias


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


class RowParallelLinear_Compress(LinearLayer_Compress):

    def __init__(self, mpu, input_size, output_size, bias=True, input_is_parallel=False, skip_bias_add=False):
        global g_mpu
        g_mpu = mpu
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        world_size = mpu.get_model_parallel_world_size()
        assert input_size % world_size == 0
        self.input_size_per_partition = input_size // world_size
        super(RowParallelLinear_Compress, self).__init__(self.input_size_per_partition, self.output_size, bias=bias)

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        output_parallel, bias = super().forward(input_parallel, True)
        output_ = reduce_from_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            if bias is not None:
                output = output_ + bias
            else:
                output = output_
            output_bias = None
        else:
            output = output_
            output_bias = bias
        return output, output_bias


class DSPolicy(ABC):
    _orig_layer_class = None

    def __init__(self):
        self.cuda_graph_supported = False

    def attention(self):
        """
        Returns attention qkv and dense parameters
        weight: (3*hidden, hidden) and (hidden, hidden)
        bias: (3*hidden) and (hidden)
        """
        raise NotImplementedError


INFERENCE_MODEL_TIMER = 'model-forward-inference'


class LinearAllreduce(nn.Module):

    def __init__(self, weight, bias=None, mp_group=None):
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output


class LinearLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(LinearLayer, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.weight = Parameter(torch.empty(weight_shape, dtype=dtype, device=torch.cuda.current_device()))
            self.bias = Parameter(torch.empty(weight_shape[0], dtype=dtype, device=torch.cuda.current_device())) if bias is not None else None

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output


class Normalize(nn.Module):

    def __init__(self, dim, dtype=torch.float, eps=1e-05):
        super(Normalize, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps).to(dtype)
        self.weight = self.norm.weight
        self.bias = self.norm.bias

    def forward(self, input):
        return self.norm(input)


class LayerSpec:
    """Building block for specifying pipeline-parallel modules.

    LayerSpec stores the type information and parameters for each stage in a
    PipelineModule. For example:

    .. code-block:: python

        nn.Sequence(
            torch.nn.Linear(self.in_dim, self.hidden_dim, bias=False),
            torch.nn.Linear(self.hidden_hidden, self.out_dim)
        )

    becomes

    .. code-block:: python

        layer_specs = [
            LayerSpec(torch.nn.Linear, self.in_dim, self.hidden_dim, bias=False),
            LayerSpec(torch.nn.Linear, self.hidden_hidden, self.out_dim)]
        ]
    """

    def __init__(self, typename, *module_args, **module_kwargs):
        self.typename = typename
        self.module_args = module_args
        self.module_kwargs = module_kwargs
        if not issubclass(typename, nn.Module):
            raise RuntimeError('LayerSpec only supports torch.nn.Module types.')
        if dist.is_initialized():
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = -1

    def __repr__(self):
        return ds_utils.call_to_str(self.typename.__name__, self.module_args, self.module_kwargs)

    def build(self, log=False):
        """Build the stored specification."""
        if log:
            logger.info(f'RANK={self.global_rank} building {repr(self)}')
        return self.typename(*self.module_args, **self.module_kwargs)


class ProcessTopology:
    """ Manages the mapping of n-dimensional Cartesian coordinates to linear
    indices. This mapping is used to map the rank of processes to the grid
    for various forms of parallelism.

    Each axis of the tensor is accessed by its name. The provided ordering
    of the axes defines the layout of the topology. ProcessTopology uses a "row-major"
    layout of the tensor axes, and so axes=['x', 'y'] would map coordinates (x,y) and
    (x,y+1) to adjacent linear indices. If instead axes=['y', 'x'] was used, coordinates
    (x,y) and (x+1,y) would be adjacent.

    Some methods return ProcessCoord namedtuples.
    """

    def __init__(self, axes, dims):
        """Create a mapping of n-dimensional tensor coordinates to linear indices.

        Arguments:
            axes (list): the names of the tensor axes
            dims (list): the dimension (length) of each axis of the topology tensor
        """
        self.axes = axes
        self.dims = dims
        self.ProcessCoord = namedtuple('ProcessCoord', axes)
        self.mapping = {}
        ranges = [range(d) for d in dims]
        for global_rank, coord in enumerate(cartesian_product(*ranges)):
            key = {axis: coord[self.axes.index(axis)] for axis in self.axes}
            key = self.ProcessCoord(**key)
            self.mapping[key] = global_rank

    def get_rank(self, **coord_kwargs):
        """Return the global rank of a process via its coordinates.

        Coordinates are specified as kwargs. For example:

            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_rank(x=0, y=1)
            1
        """
        if len(coord_kwargs) != len(self.axes):
            raise ValueError('get_rank() does not support slices. Use filter_match())')
        key = self.ProcessCoord(**coord_kwargs)
        assert key in self.mapping, f'key {coord_kwargs} invalid'
        return self.mapping[key]

    def get_axis_names(self):
        """Return a list of the axis names in the ordering of the topology. """
        return self.axes

    def get_rank_repr(self, rank, omit_axes=['data', 'pipe'], inner_sep='_', outer_sep='-'):
        """Return a string representation of a rank.

        This method is primarily used for checkpointing model data.

        For example:
            >>> topo = Topo(axes=['a', 'b'], dims=[2, 2])
            >>> topo.get_rank_repr(rank=3)
            'a_01-b_01'
            >>> topo.get_rank_repr(rank=3, omit_axes=['a'])
            'b_01'

        Args:
            rank (int): A rank in the topology.
            omit_axes (list, optional): Axes that should not be in the representation. Defaults to ['data', 'pipe'].
            inner_sep (str, optional): [description]. Defaults to '_'.
            outer_sep (str, optional): [description]. Defaults to '-'.

        Returns:
            str: A string representation of the coordinate owned by ``rank``.
        """
        omit_axes = frozenset(omit_axes)
        axes = [a for a in self.get_axis_names() if a not in omit_axes]
        names = []
        for ax in axes:
            ax_rank = getattr(self.get_coord(rank=rank), ax)
            names.append(f'{ax}{inner_sep}{ax_rank:02d}')
        return outer_sep.join(names)

    def get_dim(self, axis):
        """Return the number of processes along the given axis.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_dim('y')
            3
        """
        if axis not in self.axes:
            return 0
        return self.dims[self.axes.index(axis)]

    def get_coord(self, rank):
        """Return the coordinate owned by a process rank.

        The axes of the returned namedtuple can be directly accessed as members. For
        example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> coord = X.get_coord(rank=1)
            >>> coord.x
            0
            >>> coord.y
            1
        """
        for coord, idx in self.mapping.items():
            if idx == rank:
                return coord
        raise ValueError(f'rank {rank} not found in topology.')

    def get_axis_comm_lists(self, axis):
        """ Construct lists suitable for a communicator group along axis ``axis``.

        Example:
            >>> topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> topo.get_axis_comm_lists('pipe')
            [
                [0, 4], # data=0, model=0
                [1, 5], # data=0, model=1
                [2, 6], # data=1, model=0
                [3, 7], # data=1, model=1
            ]

        Returns:
            A list of lists whose coordinates match in all axes *except* ``axis``.
        """
        if axis not in self.axes:
            return []
        other_axes = [a for a in self.axes if a != axis]
        lists = []
        ranges = [range(self.get_dim(a)) for a in other_axes]
        for coord in cartesian_product(*ranges):
            other_keys = {a: coord[other_axes.index(a)] for a in other_axes}
            sub_list = []
            for axis_key in range(self.get_dim(axis)):
                key = self.ProcessCoord(**other_keys, **{axis: axis_key})
                sub_list.append(self.mapping[key])
            lists.append(sub_list)
        return lists

    def filter_match(self, **filter_kwargs):
        """Return the list of ranks whose coordinates match the provided criteria.

        Example:
            >>> X = ProcessTopology(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> X.filter_match(pipe=0, data=1)
            [2, 3]
            >>> [X.get_coord(rank) for rank in X.filter_match(pipe=0, data=1)]
            [ProcessCoord(pipe=0, data=1, model=0), ProcessCoord(pipe=0, data=1, model=1)]

        Arguments:
            **filter_kwargs (dict): criteria used to select coordinates.

        Returns:
            The list of ranks whose coordinates match filter_kwargs.
        """

        def _filter_helper(x):
            for key, val in filter_kwargs.items():
                if getattr(x, key) != val:
                    return False
            return True
        coords = filter(_filter_helper, self.mapping.keys())
        return [self.mapping[coord] for coord in coords]

    def get_axis_list(self, axis, idx):
        """Returns the list of global ranks whose coordinate in an axis is idx.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_axis_list(axis='x', idx=0)
            [0, 1, 2]
            >>> X.get_axis_list(axis='y', idx=0)
            [0, 3]
        """
        axis_num = self.axes.index(axis)
        ranks = [self.mapping[k] for k in self.mapping.keys() if k[axis_num] == idx]
        return ranks

    def world_size(self):
        return len(self.mapping)

    def __str__(self):
        return str(self.mapping)


class PipeDataParallelTopology(ProcessTopology):
    """ A topology specialization for hybrid data and pipeline parallelism.

        Uses data parallelism on the last dimension to encourage gradient
        reductions to use high-bandwidth intra-node links and lower-volume
        pipeline communications to use low-bandwidth inter-node links.
    """

    def __init__(self, num_pp, num_dp):
        super().__init__(axes=['pipe', 'data'], dims=[num_pp, num_dp])


def _prime_factors(N):
    """ Returns the prime factorization of positive integer N. """
    if N <= 0:
        raise ValueError('Values must be strictly positive.')
    primes = []
    while N != 1:
        for candidate in range(2, N + 1):
            if N % candidate == 0:
                primes.append(candidate)
                N //= candidate
                break
    return primes


class PipelineParallelGrid:
    """Implements a grid object that stores the data parallel ranks
    corresponding to each of the model parallel stages

    The grid object organizes the processes in a distributed pytorch job
    into a 2D grid, of stage_id and data_parallel_id.

    self.stage_id and self.data_parallel_id stores the stage id
    and the data parallel id of current process.

    self.dp_group groups the processes by stage_id.
    self.dp_group[i], is a list containing all process ranks whose
    stage_id is i.

    self.p2p_groups stores a list of tuple, where each tuple
    stores process ranks of adjacent stages for a given data_parallel_id.
    For example if num_stage is 5 then a tuple [7,8] represents stages [3, 4],
    with data_parallel id = 1. A stage wrap around will appear as non-adjacent ranks,
    for example tuple [4,0] with representing wrap-around stage 4 and 0, for
    data_parallel_id = 0, or similarly [9,5] represents wrapped around stages [4,0]
    for data_parallel_id = 1.
    """

    def __init__(self, topology=None, process_group=None):
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if topology is not None:
            if self.global_rank == 0:
                None
            self._topo = topology
        else:
            num_pp = 1
            num_dp = 1
            for idx, prime in enumerate(_prime_factors(self.world_size)):
                if idx % 2 == 0:
                    num_pp *= prime
                else:
                    num_dp *= prime
            self._topo = PipeDataParallelTopology(num_dp=num_dp, num_pp=num_pp)
        self.data_parallel_size = max(self._topo.get_dim('data'), 1)
        self.pipe_parallel_size = max(self._topo.get_dim('pipe'), 1)
        self.model_parallel_size = max(self._topo.get_dim('model'), 1)
        self.slice_parallel_size = self.model_parallel_size
        assert self._is_grid_valid(), 'Invalid Grid'
        self.stage_id = self.get_stage_id()
        self.data_parallel_id = self.get_data_parallel_id()
        self.ds_model_proc_group = None
        self.ds_model_rank = -1
        for dp in range(self.data_parallel_size):
            ranks = sorted(self._topo.get_axis_list(axis='data', idx=dp))
            if self.global_rank == 0:
                pass
            proc_group = dist.new_group(ranks=ranks)
            if self.global_rank in ranks:
                self.ds_model_proc_group = proc_group
                self.ds_model_world_size = len(ranks)
                self.ds_model_rank = ranks.index(self.global_rank)
        assert self.ds_model_rank > -1
        assert self.ds_model_proc_group is not None
        self.dp_group = []
        self.dp_groups = self._topo.get_axis_comm_lists('data')
        for g in self.dp_groups:
            proc_group = dist.new_group(ranks=g)
            if self.global_rank in g:
                self.dp_group = g
                self.dp_proc_group = proc_group
        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = self.stage_id == self.pipe_parallel_size - 1
        self.p2p_groups = self._build_p2p_groups()
        self.pp_group = []
        self.pp_proc_group = None
        self.pipe_groups = self._topo.get_axis_comm_lists('pipe')
        for ranks in self.pipe_groups:
            if self.global_rank == 0:
                pass
            proc_group = dist.new_group(ranks=ranks)
            if self.global_rank in ranks:
                self.pp_group = ranks
                self.pp_proc_group = proc_group
        assert self.pp_proc_group is not None
        if self.model_parallel_size == 1:
            for group_rank in range(self.world_size):
                group_rank = [group_rank]
                group = dist.new_group(ranks=group_rank)
                if group_rank[0] == self.global_rank:
                    self.slice_group = group_rank
                    self.slice_proc_group = group
            return
        else:
            self.mp_group = []
            self.model_groups = self._topo.get_axis_comm_lists('model')
            for g in self.model_groups:
                proc_group = dist.new_group(ranks=g)
                if self.global_rank in g:
                    self.slice_group = g
                    self.slice_proc_group = proc_group

    def get_stage_id(self):
        return self._topo.get_coord(rank=self.global_rank).pipe

    def get_data_parallel_id(self):
        return self._topo.get_coord(rank=self.global_rank).data

    def _build_p2p_groups(self):
        """Groups for sending and receiving activations and gradients across model
        parallel stages.
        """
        comm_lists = self._topo.get_axis_comm_lists('pipe')
        p2p_lists = []
        for rank in range(self.world_size):
            for l in comm_lists:
                assert len(l) == self.pipe_parallel_size
                if rank in l:
                    idx = l.index(rank)
                    buddy_rank = l[(idx + 1) % self.pipe_parallel_size]
                    p2p_lists.append([rank, buddy_rank])
                    break
        assert len(p2p_lists) == self.world_size
        return p2p_lists

    def _is_grid_valid(self):
        ranks = 1
        for ax in self._topo.get_axis_names():
            ranks *= self._topo.get_dim(ax)
        return ranks == dist.get_world_size()

    def stage_to_global(self, stage_id, **kwargs):
        me = self._topo.get_coord(self.global_rank)
        transform = me._replace(pipe=stage_id, **kwargs)._asdict()
        return self._topo.get_rank(**transform)

    def topology(self):
        return self._topo

    def get_global_rank(self):
        return self.global_rank

    def get_pipe_parallel_rank(self):
        """ The stage of the pipeline this rank resides in. """
        return self.get_stage_id()

    def get_pipe_parallel_world_size(self):
        """ The number of stages in the pipeline. """
        return self.pipe_parallel_size

    def get_pipe_parallel_group(self):
        """ The group of ranks within the same pipeline. """
        return self.pp_proc_group

    def get_data_parallel_rank(self):
        """ Which pipeline this rank resides in. """
        return self.data_parallel_id

    def get_data_parallel_world_size(self):
        """ The number of pipelines. """
        return self.data_parallel_size

    def get_data_parallel_group(self):
        """ The group of ranks within the same stage of all pipelines. """
        return self.dp_proc_group

    def get_model_parallel_rank(self):
        return self.ds_model_rank

    def get_model_parallel_world_size(self):
        return self.ds_model_world_size

    def get_model_parallel_group(self):
        return self.ds_model_proc_group

    def get_slice_parallel_rank(self):
        if 'model' in self._topo.get_axis_names():
            return self._topo.get_coord(rank=self.global_rank).model
        else:
            return 0

    def get_slice_parallel_world_size(self):
        return self.slice_parallel_size

    def get_slice_parallel_group(self):
        return self.slice_proc_group


AUTO_MODULE_KEY = 'auto'


class CheckpointEngine(object):

    def __init__(self, config_params=None):
        pass

    def create(self, tag):
        pass

    def save(self, state_dict, path: str):
        pass

    def load(self, path: str, map_location=None):
        pass

    def commit(self, tag):
        pass


def log_dist(message, ranks=None, level=logging.INFO):
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


class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)

    def create(self, tag):
        log_dist(f'[Torch] Checkpoint {tag} is begin to save!', ranks=[0])

    def save(self, state_dict, path: str):
        logger.info(f'[Torch] Saving {path}...')
        torch.save(state_dict, path)
        logger.info(f'[Torch] Saved {path}.')
        return None

    def load(self, path: str, map_location=None):
        logger.info(f'[Torch] Loading checkpoint from {path}...')
        partition = torch.load(path, map_location=map_location)
        logger.info(f'[Torch] Loaded checkpoint from {path}.')
        return partition

    def commit(self, tag):
        logger.info(f'[Torch] Checkpoint {tag} is ready now!')
        return True


class WeightQuantization(object):

    def __init__(self, mlp_extra_grouping=True, mp_size=1):
        self.dense_scales = []
        self.qkv_scales = []
        self.mlp4hh_scales = []
        self.mlph4h_scales = []
        self.mlp_extra_grouping = mlp_extra_grouping
        self.mp_size = mp_size

    def quantize_data(self, data, quantize_bits, groups, key=None):
        data_groups = torch.split(data.float().view(-1), data.numel() // groups)
        max_d = [max(g.max(), g.min().abs()) for g in data_groups]
        data_scale = [(float(1 << quantize_bits) / (2 * mx + 1e-05)) for mx in max_d]
        data_int = [(g * s) for g, s in zip(data_groups, data_scale)]
        data_int = [di.round().clamp(-(1 << quantize_bits - 1), (1 << quantize_bits - 1) - 1) for di in data_int]
        data_int = torch.cat(data_int).reshape(data.shape)
        data_int = data_int
        data_scale = torch.cat([s.unsqueeze(0).unsqueeze(0) for s in data_scale])
        return data_int, data_scale

    def is_mlp(self, data, merge_count=1):
        return self.mp_size * data.shape[0] * merge_count / data.shape[1] == 4 or self.mp_size * data.shape[1] * merge_count / data.shape[0] == 4

    def is_qkv(self, data):
        return self.mp_size * data.shape[0] / data.shape[1] == 3 or self.mp_size * data.shape[1] / data.shape[0] == 3

    def Quantize(self, value_list, quantize_bits, groups, key, merge_dim=0):
        if self.mlp_extra_grouping and self.is_mlp(value_list[0], merge_count=len(value_list)):
            groups *= 2
        q_scale = []
        index = 0
        for data in value_list:
            data_int, data_scale = self.quantize_data(data, quantize_bits, groups, key)
            q_scale.append(data_scale)
            value_list[index] = data_int
            index += 1
        q_scale = 1 / torch.cat(q_scale, dim=merge_dim).view(-1).unsqueeze(0)
        if 'mlp.dense_4h_to_h.weight' in key:
            self.mlp4hh_scales.append(q_scale)
        elif 'mlp.dense_h_to_4h.weight' in key:
            self.mlph4h_scales.append(q_scale)
        elif 'attention.query_key_value.weight' in key:
            self.qkv_scales.append(q_scale)
        else:
            self.dense_scales.append(q_scale)
        return value_list

    def merge_layer_scales(self, layer_scales):
        max_dim = max([s.shape[-1] for s in layer_scales])
        layer_scales = [(torch.cat((s, torch.zeros((1, max_dim - s.shape[-1]), device=torch.cuda.current_device())), dim=-1) if s.shape[-1] < max_dim else s) for s in layer_scales]
        return torch.cat(layer_scales).unsqueeze(0)

    def merge_scales(self):
        all_scales = []
        for dense_scale, qkv_scale, m4hh_scale, mh4h_scale in zip(self.dense_scales, self.qkv_scales, self.mlp4hh_scales, self.mlph4h_scales):
            all_scales.append(self.merge_layer_scales([qkv_scale, dense_scale, mh4h_scale, m4hh_scale]))
        return torch.cat(all_scales)

    def merge_scales_split(self, split_count):
        all_scales = [[] for _ in range(split_count)]
        for dense_scale, qkv_scale, m4hh_scale, mh4h_scale in zip(self.dense_scales, self.qkv_scales, self.mlp4hh_scales, self.mlph4h_scales):
            dense_scale = torch.split(dense_scale, dense_scale.numel() // split_count)
            qkv_scale = torch.split(qkv_scale, qkv_scale.numel() // split_count)
            m4hh_scale = torch.split(m4hh_scale, m4hh_scale.numel() // split_count)
            mh4h_scale = torch.split(mh4h_scale, mh4h_scale.numel() // split_count)
            for s in range(split_count):
                all_scales[s].append(torch.cat([torch.cat((qkv_scale[s], torch.zeros_like(qkv_scale[s])), dim=1), torch.cat((dense_scale[s], torch.zeros_like(dense_scale[s])), dim=1), mh4h_scale[s], m4hh_scale[s]]).unsqueeze(0))
            for scales_a in all_scales:
                torch.cat(scales_a)
        return all_scales

    def sd_quantize_megatron(self, sd, quantize_bits, groups):
        keys = sd.keys()
        for key in keys:
            value_list = [sd[key]]
            if 'attention.dense.weight' in key or 'mlp.dense_4h_to_h.weight' in key or 'mlp.dense_h_to_4h.weight' in key or 'attention.query_key_value.weight' in key:
                value_list = self.Quantize(value_list, quantize_bits, groups, key=key)
            sd[key] = value_list[0]
        all_scales = self.merge_scales()
        return sd, all_scales

    def model_quantize(self, model, quantize_policy, quantize_bits, groups):
        all_scales = []

        def quantize_fn(layer, policy_cls):
            policy = policy_cls(layer)
            _, qkvw, _, dense_w, _, _ = policy.attention()
            _, _h4h_w, _, _4hh_w, _ = policy.mlp()
            keys = [qkvw, dense_w, _h4h_w, _4hh_w]
            layer_scales = []
            for key in range(len(keys)):
                if self.mlp_extra_grouping and self.is_mlp(keys[key]):
                    data_quantized, data_scale = self.quantize_data(keys[key], quantize_bits, groups * 2)
                elif policy_cls is HFBertLayerPolicy and self.is_qkv(keys[key]):
                    data_quantized, data_scale = self.quantize_data(keys[key], quantize_bits, groups * 3)
                else:
                    data_quantized, data_scale = self.quantize_data(keys[key], quantize_bits, groups)
                keys[key].copy_(data_quantized)
                layer_scales.append(1 / data_scale.view(-1).unsqueeze(0))
            all_scales.append(self.merge_layer_scales(layer_scales))
            return layer

        def _quantize_module(model, policies):
            for name, child in model.named_children():
                if child.__class__ in policies:
                    quantize_fn, replace_policy = policies[child.__class__]
                    setattr(model, name, quantize_fn(child, replace_policy))
                else:
                    _quantize_module(child, policies)
            return model
        policy = {}
        if quantize_policy is not None:
            for layer_name, replace_policy in quantize_policy.items():
                policy.update({layer_name: (quantize_fn, replace_policy)})
        else:
            for plcy in replace_policies:
                policy.update({plcy._orig_layer_class: (quantize_fn, plcy)})
        quantized_module = _quantize_module(model, policy)
        return quantized_module, torch.cat(all_scales)


class SDLoaderBase(ABC):

    def __init__(self, ckpt_list, version, checkpoint_engine):
        self.module_key = None
        self.ckpt_list = ckpt_list
        self.version = version
        self.checkpoint_engine = TorchCheckpointEngine() if checkpoint_engine is None else checkpoint_engine
        self.check_ckpt_list()

    def load(self, mp_world_size, mp_rank, module_key=AUTO_MODULE_KEY, is_pipe_parallel=False, quantize=False, quantize_bits=8, quantize_groups=64, mlp_extra_grouping=True):
        self.module_key = module_key
        num_ckpt = len(self.ckpt_list)
        idx = mp_rank * num_ckpt // mp_world_size
        """ We have multiple cases to handle here for both training and inference:
            1. PipeModule loading mp_rank_*.pt files, is_pipe_parallel=True, module_key is not None
                a. if no mp_size/pp_size resizing occurs, for both training & inference, loading
                   the mp_rank related checkpoint directly.
                b. if has mp_size/pp_size resizing, only Megatron model inference is supported,
                   in this case each mp_rank_*.pt have same content, we will load the first checkpoint
                   file (idx=0), to avoid idx exceeding file list boundary.

            2. PipeModule loading layer_*.pt files, is_pipe_parallel=True, module_key is None
                a. if no mp_size resizing occurs, for both training & inference, loading
                   the mp_rank related checkpoint directly.
                b. if has mp_size resizing, only Megatron model inference is supported,
                   checkpoint file(s) will be merged/split according to mp_rank, mp_world_size and
                   checkpoint file list.

            3. Non-PipeModule loading mp_rank_*.pt files, is_pipe_parallel=False
                Same with case (2).
        """
        if is_pipe_parallel and module_key is not None and mp_world_size != num_ckpt:
            mp_world_size = num_ckpt
            idx = 0
        load_path = self.ckpt_list[idx]
        merge_count = 1
        if num_ckpt == mp_world_size:
            assert os.path.exists(load_path)
            sd = self.checkpoint_engine.load(load_path, map_location=lambda storage, loc: storage)
            if quantize:
                quantizer = WeightQuantization(mlp_extra_grouping=mlp_extra_grouping, mp_size=mp_world_size)
                sd_module, all_scales = quantizer.sd_quantize_megatron(self.get_module(sd), quantize_bits, quantize_groups)
                self.set_module(sd, sd_module)
            else:
                all_scales = None
        elif num_ckpt > mp_world_size:
            sd, all_scales, merge_count = self.merge_state_dict(mp_world_size, mp_rank, quantize, quantize_bits, quantize_groups, mlp_extra_grouping)
        else:
            sd, all_scales = self.split_state_dict(mp_world_size, mp_rank, quantize, quantize_bits, quantize_groups, mlp_extra_grouping)
        return load_path, sd, (all_scales, merge_count)

    def get_merge_state_dicts(self, mp_world_size, mp_rank):
        num_ckpt = len(self.ckpt_list)
        assert num_ckpt % mp_world_size == 0, 'Invalid checkpoints and world size for sd merge'
        num_to_merge = num_ckpt // mp_world_size
        ckpt_list = [self.ckpt_list[i] for i in range(num_to_merge * mp_rank, num_to_merge * (mp_rank + 1))]
        logger.info(f'mp_rank: {mp_rank}, ckpt_list: {ckpt_list}')
        sd_list = [self.checkpoint_engine.load(ckpt, map_location=lambda storage, loc: storage) for ckpt in ckpt_list]
        return sd_list

    def get_split_state_dict(self, mp_world_size, mp_rank):
        num_ckpt = len(self.ckpt_list)
        assert mp_world_size % num_ckpt == 0, 'Invalid checkpoints and world size for sd split'
        num_to_split = mp_world_size // num_ckpt
        ckpt_index = mp_rank // num_to_split
        ckpt_offset = mp_rank % num_to_split
        logger.info(f'mp_rank: {mp_rank}, ckpt_list: {self.ckpt_list[ckpt_index]}, offset: {ckpt_offset}')
        sd = self.checkpoint_engine.load(self.ckpt_list[ckpt_index], map_location=lambda storage, loc: storage)
        return sd, num_to_split, ckpt_offset

    def _choose_module_key(self, sd):
        assert not ('module' in sd and 'model' in sd), "checkpoint has both 'model' and 'module' keys, not sure how to proceed"
        assert 'module' in sd or 'model' in sd, "checkpoint contains neither 'model' or 'module' keys, not sure how to proceed"
        if 'module' in sd:
            return 'module'
        elif 'model' in sd:
            return 'model'

    def get_module(self, sd):
        if self.module_key is None:
            return sd
        elif self.module_key == AUTO_MODULE_KEY:
            return sd[self._choose_module_key(sd)]
        else:
            return sd[self.module_key]

    def set_module(self, sd, module):
        if self.module_key is None:
            sd = module
        elif self.module_key == AUTO_MODULE_KEY:
            sd[self._choose_module_key(sd)] = module
        else:
            sd[self.module_key] = module
        return sd

    def check_ckpt_list(self):
        assert len(self.ckpt_list) > 0
        sd = self.checkpoint_engine.load(self.ckpt_list[0], map_location=lambda storage, loc: storage)
        if 'mp_world_size' in sd.keys():
            assert len(self.ckpt_list) == sd['mp_world_size'], f"checkpoint count {len(self.ckpt_list)} is different from saved mp_world_size {sd['mp_world_size']}"

    @abstractmethod
    def merge_state_dict(self, mp_world_size, mp_rank, quantize, quantize_bits, groups, mlp_extra_grouping):
        pass

    @abstractmethod
    def split_state_dict(self, mp_world_size, mp_rank, quantize, quantize_bits, groups, mlp_extra_grouping):
        pass

    @abstractmethod
    def sanity_check(self, ckpt_file_name):
        pass


class MegatronSDLoader(SDLoaderBase):

    def __init__(self, ckpt_list, version, checkpoint_engine):
        super().__init__(ckpt_list, version, checkpoint_engine)
        """
        ## Q/K/V data need special processing
        key: transformer.layers.0.attention.query_key_value.weight, shape: torch.Size([3192, 4256])
        key: transformer.layers.0.attention.query_key_value.bias, shape: torch.Size([3192])

        ## merge or split on axis=0
        key: word_embeddings.weight, shape: torch.Size([12672, 4256])
        key: transformer.layers.0.mlp.dense_h_to_4h.bias, shape: torch.Size([4256])
        key: transformer.layers.0.mlp.dense_h_to_4h.weight, shape: torch.Size([4256, 4256])

        ## merge or split on axis=1
        key: transformer.layers.0.attention.dense.weight, shape: torch.Size([4256, 1064])
        key: transformer.layers.0.mlp.dense_4h_to_h.weight, shape: torch.Size([4256, 4256])

        ## no change required
        key: transformer.layers.0.mlp.dense_4h_to_h.bias, shape: torch.Size([4256])
        key: transformer.final_layernorm.weight, shape: torch.Size([4256])
        key: transformer.final_layernorm.bias, shape: torch.Size([4256])
        key: transformer.layers.0.attention.dense.bias, shape: torch.Size([4256])
        key: transformer.layers.0.post_attention_layernorm.weight, shape: torch.Size([4256])
        key: transformer.layers.0.post_attention_layernorm.bias, shape: torch.Size([4256])
        key: transformer.layers.0.input_layernorm.weight, shape: torch.Size([4256])
        key: transformer.layers.0.input_layernorm.bias, shape: torch.Size([4256])
        key: position_embeddings.weight, shape: torch.Size([1024, 4256])
        """

    def merge_query_key_value(self, param_list, ckpt_ver):
        """
        Up to now we found 3 Q/K/V parameter formats in different Megatron checkpoint versions:

        1. version 0, there is no version information saved in checkpoint.
            format: [(3 * np * hn), h]
        2. version 1.0
            format: [(np * hn * 3), h]
        3. version 2.0
            format: [(np * 3 * hn), h]

        h: hidden size
        n: number of attention heads
        p: number of model parallel partitions
        np: n/p
        hn: h/n
        """
        new_qkv = None
        if ckpt_ver == 0:
            assert param_list[0].shape[0] % 3 == 0
            size_qkv = param_list[0].shape[0] // 3
            split_tensors = [torch.split(param, size_qkv, dim=0) for param in param_list]
            tensors = []
            for i in range(3):
                tensor_tuple = [t[i] for t in split_tensors]
                tensors.append(torch.cat(tensor_tuple, axis=0))
            new_qkv = torch.cat(tensors, axis=0)
        elif ckpt_ver == 1.0 or ckpt_ver == 2.0:
            new_qkv = torch.cat(param_list, axis=0)
        else:
            assert False, f'checkpoint version: {ckpt_ver} is not supported'
        return new_qkv

    def split_query_key_value(self, param, num_to_split, offset, ckpt_ver):
        """
        Up to now we found 3 Q/K/V parameter formats in different Megatron checkpoint versions:

        1. version 0, there is no version information saved in checkpoint.
            format: [(3 * np * hn), h]
        2. version 1.0
            format: [(np * hn * 3), h]
        3. version 2.0
            format: [(np * 3 * hn), h]

        h: hidden size
        n: number of attention heads
        p: number of model parallel partitions
        np: n/p
        hn: h/n
        """
        new_qkv = None
        if ckpt_ver == 0:
            assert param.shape[0] % 3 == 0
            size_qkv = param.shape[0] // 3
            split_tensors = torch.split(param, size_qkv, dim=0)
            assert split_tensors[0].shape[0] % num_to_split == 0
            split_size = split_tensors[0].shape[0] // num_to_split
            tensors = []
            for i in range(3):
                tensors.append(torch.split(split_tensors[i], split_size, dim=0)[offset])
            new_qkv = torch.cat(tensors, axis=0)
        elif ckpt_ver == 1.0 or ckpt_ver == 2.0:
            assert param.shape[0] % num_to_split == 0
            size_qkv = param.shape[0] // num_to_split
            split_tensors = torch.split(param, size_qkv, dim=0)
            new_qkv = split_tensors[offset]
        else:
            assert False, f'checkpoint version: {ckpt_ver} is not supported'
        return new_qkv

    def merge_state_dict(self, mp_world_size, mp_rank, quantize=False, quantize_bits=8, groups=64, mlp_extra_grouping=True):
        self.sanity_check(self.ckpt_list[0])
        sd_list = self.get_merge_state_dicts(mp_world_size, mp_rank)
        ds_sd = copy.deepcopy(sd_list[0])
        new_client_sd = collections.OrderedDict()
        client_sd_list = [self.get_module(sd) for sd in sd_list]
        keys = client_sd_list[0].keys()
        ckpt_ver = self.get_checkpoint_version(ds_sd)
        logger.info(f'checkpoint version: {ckpt_ver}')
        if quantize:
            quantizer = WeightQuantization(mlp_extra_grouping=mlp_extra_grouping, mp_size=mp_world_size)
        for key in keys:
            value_list = [sd[key] for sd in client_sd_list]
            if 'attention.dense.weight' in key or 'mlp.dense_4h_to_h.weight' in key:
                if quantize:
                    value_list = quantizer.Quantize(value_list, quantize_bits, groups, key=key, merge_dim=1)
                new_client_sd[key] = torch.cat(value_list, axis=1)
            elif 'attention.query_key_value' in key:
                if quantize and 'attention.query_key_value.weight' in key:
                    value_list = quantizer.Quantize(value_list, quantize_bits, groups, key=key)
                    new_client_sd[key] = torch.cat(value_list, axis=0)
                elif quantize:
                    new_client_sd[key] = torch.cat(value_list, axis=0)
                else:
                    new_client_sd[key] = self.merge_query_key_value(value_list, ckpt_ver)
            elif 'mlp.dense_h_to_4h.weight' in key or 'word_embeddings.weight' in key or 'mlp.dense_h_to_4h.bias' in key:
                if quantize and 'mlp.dense_h_to_4h.weight' in key:
                    value_list = quantizer.Quantize(value_list, quantize_bits, groups, key=key)
                new_client_sd[key] = torch.cat(value_list, axis=0)
            else:
                new_client_sd[key] = value_list[0]
        if quantize:
            all_scales = quantizer.merge_scales()
        ds_sd = self.set_module(ds_sd, new_client_sd)
        return ds_sd, all_scales if quantize else None, len(client_sd_list)

    def split_state_dict(self, mp_world_size, mp_rank, quantize=False, quantize_bits=8, groups=64, mlp_extra_grouping=True):
        sd, num_to_split, ckpt_offset = self.get_split_state_dict(mp_world_size, mp_rank)
        ds_sd = copy.deepcopy(sd)
        new_client_sd = collections.OrderedDict()
        client_sd = self.get_module(sd)
        ckpt_ver = self.get_checkpoint_version(ds_sd)
        logger.info(f'checkpoint version: {ckpt_ver}')
        if quantize:
            quantizer = WeightQuantization(mlp_extra_grouping=mlp_extra_grouping, mp_size=mp_world_size)
        for key in client_sd.keys():
            value = client_sd[key]
            if 'attention.dense.weight' in key or 'mlp.dense_4h_to_h.weight' in key:
                assert value.shape[1] % num_to_split == 0
                split_size = value.shape[1] // num_to_split
                if quantize:
                    q_vals = quantizer.Quantize([value], quantize_bits, groups, key)
                    value = q_vals[0]
                new_client_sd[key] = torch.split(value, split_size, dim=1)[ckpt_offset]
            elif 'attention.query_key_value' in key:
                if quantize and 'attention.query_key_value.weight' in key:
                    q_vals = quantizer.Quantize([value], quantize_bits, groups, key)
                    value = q_vals[0]
                new_client_sd[key] = self.split_query_key_value(value, num_to_split, ckpt_offset, ckpt_ver)
            elif 'mlp.dense_h_to_4h.weight' in key or 'word_embeddings.weight' in key or 'mlp.dense_h_to_4h.bias' in key or 'final_linear.weight' in key:
                assert value.shape[0] % num_to_split == 0
                split_size = value.shape[0] // num_to_split
                if quantize and 'mlp.dense_h_to_4h.weight' in key:
                    q_vals = quantizer.Quantize([value], quantize_bits, groups, key)
                    value = q_vals[0]
                new_client_sd[key] = torch.split(value, split_size, dim=0)[ckpt_offset]
            else:
                new_client_sd[key] = value
        if quantize:
            all_scales = quantizer.merge_scales_split(num_to_split)
        ds_sd = self.set_module(ds_sd, new_client_sd)
        return ds_sd, all_scales if quantize else None

    def sanity_check(self, ckpt_file_name):
        keys_to_check = ['attention.dense.weight', 'mlp.dense_4h_to_h.weight', 'attention.query_key_value', 'mlp.dense_h_to_4h.weight', 'mlp.dense_h_to_4h.bias']
        sd = self.checkpoint_engine.load(ckpt_file_name, map_location=lambda storage, loc: storage)

        def check_key_exist(partial_key, sd):
            keys = sd.keys()
            found = False
            for k in keys:
                if partial_key in k:
                    found = True
                    break
            return found
        for key in keys_to_check:
            assert check_key_exist(key, self.get_module(sd)), f'key: {key} is not found in the checkpoint {ckpt_file_name}'

    def get_checkpoint_version(self, state_dict):
        return self.version if self.version is not None else state_dict.get('checkpoint_version', 0)


class SDLoaderFactory:

    @staticmethod
    def get_sd_loader_json(json_file, checkpoint_engine):
        if isinstance(json_file, str):
            with open(json_file) as f:
                data = json.load(f)
        else:
            assert isinstance(json_file, dict)
            data = json_file
        sd_type = data['type']
        ckpt_list = data['checkpoints']
        version = data['version']
        ckpt_type = data.get('parallelization', 'pp')
        mp_size = data.get('mp_size', 0)
        if sd_type.lower() in ['bloom', 'ds_model']:
            return data
        return SDLoaderFactory.get_sd_loader(ckpt_list, checkpoint_engine, sd_type, version)

    @staticmethod
    def get_sd_loader(ckpt_list, checkpoint_engine, sd_type='Megatron', version=None):
        if sd_type == 'Megatron':
            return MegatronSDLoader(ckpt_list, version, checkpoint_engine)
        else:
            assert False, '{} checkpoint type is not supported'.format(sd_type)


class TiedLayerSpec(LayerSpec):

    def __init__(self, key, typename, *module_args, forward_fn=None, tied_weight_attr='weight', **module_kwargs):
        super().__init__(typename, *module_args, **module_kwargs)
        self.key = key
        self.forward_fn = forward_fn
        self.tied_weight_attr = tied_weight_attr


class PipelineModule(nn.Module):
    """Modules to be parallelized with pipeline parallelism.

    The key constraint that enables pipeline parallelism is the
    representation of the forward pass as a sequence of layers
    and the enforcement of a simple interface between them. The
    forward pass is implicitly defined by the module ``layers``. The key
    assumption is that the output of each layer can be directly fed as
    input to the next, like a ``torch.nn.Sequence``. The forward pass is
    implicitly:

    .. code-block:: python

        def forward(self, inputs):
            x = inputs
            for layer in self.layers:
                x = layer(x)
            return x

    .. note::
        Pipeline parallelism is not compatible with ZeRO-2 and ZeRO-3.

    Args:
        layers (Iterable): A sequence of layers defining pipeline structure. Can be a ``torch.nn.Sequential`` module.
        num_stages (int, optional): The degree of pipeline parallelism. If not specified, ``topology`` must be provided.
        topology (``deepspeed.runtime.pipe.ProcessTopology``, optional): Defines the axes of parallelism axes for training. Must be provided if ``num_stages`` is ``None``.
        loss_fn (callable, optional): Loss is computed ``loss = loss_fn(outputs, label)``
        seed_layers(bool, optional): Use a different seed for each layer. Defaults to False.
        seed_fn(type, optional): The custom seed generating function. Defaults to random seed generator.
        base_seed (int, optional): The starting seed. Defaults to 1234.
        partition_method (str, optional): The method upon which the layers are partitioned. Defaults to 'parameters'.
        activation_checkpoint_interval (int, optional): The granularity activation checkpointing in terms of number of layers. 0 disables activation checkpointing.
        activation_checkpoint_func (callable, optional): The function to use for activation checkpointing. Defaults to ``deepspeed.checkpointing.checkpoint``.
        checkpointable_layers(list, optional): Checkpointable layers may not be checkpointed. Defaults to None which does not additional filtering.
    """

    def __init__(self, layers, num_stages=None, topology=None, loss_fn=None, seed_layers=False, seed_fn=None, base_seed=1234, partition_method='parameters', activation_checkpoint_interval=0, activation_checkpoint_func=checkpointing.checkpoint, checkpointable_layers=None):
        super().__init__()
        if num_stages is None and topology is None:
            raise RuntimeError('must provide num_stages or topology')
        self.micro_offset = 0
        self.loss_fn = loss_fn
        self.checkpointable_layers = checkpointable_layers
        if checkpointable_layers is not None:
            assert isinstance(checkpointable_layers, list), 'param `checkpointable_layers` must be type of list.'
        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            None
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get('LOCAL_RANK', None))
        assert self.local_rank != None
        if topology:
            self._topo = topology
            self.num_stages = self._topo.get_dim('pipe')
        else:
            self.num_stages = num_stages
            if topology is None:
                if self.world_size % self.num_stages != 0:
                    raise RuntimeError(f'num_stages ({self.num_stages}) must divide distributed world size ({self.world_size})')
                dp = self.world_size // num_stages
                topology = PipeDataParallelTopology(num_pp=num_stages, num_dp=dp)
                self._topo = topology
        self._grid = PipelineParallelGrid(process_group=self.world_group, topology=self._topo)
        self.stage_id = self._topo.get_coord(self.global_rank).pipe
        self._layer_specs = list(layers)
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=partition_method)
        self.forward_funcs = []
        self.fwd_map = {}
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}
        self._build()
        self
        self.tied_comms = self._index_tied_modules()
        self._synchronize_tied_weights()
        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.activation_checkpoint_func = activation_checkpoint_func

    def _build(self):
        specs = self._layer_specs
        for local_idx, layer in enumerate(specs[self._local_start:self._local_stop]):
            layer_idx = local_idx + self._local_start
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    ds_utils.set_random_seed(self.base_seed + layer_idx)
            if isinstance(layer, PipelineModule):
                raise NotImplementedError('RECURSIVE BUILD NOT YET IMPLEMENTED')
            elif isinstance(layer, nn.Module):
                name = str(layer_idx)
                self.forward_funcs.append(layer)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, layer)
            elif isinstance(layer, TiedLayerSpec):
                if layer.key not in self.tied_modules:
                    self.tied_modules[layer.key] = layer.build()
                    self.tied_weight_attrs[layer.key] = layer.tied_weight_attr
                if layer.forward_fn is None:
                    self.forward_funcs.append(self.tied_modules[layer.key])
                else:
                    self.forward_funcs.append(partial(layer.forward_fn, self.tied_modules[layer.key]))
            elif isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)
                self.forward_funcs.append(module)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, module)
            else:
                self.forward_funcs.append(layer)
        for p in self.parameters():
            p.ds_pipe_replicated = False

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self._layer_specs)
        for idx, layer in enumerate(self._layer_specs):
            if isinstance(layer, LayerSpec):
                l = layer.build()
                params = filter(lambda p: p.requires_grad, l.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
            elif isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def _find_layer_type(self, layername):
        idxs = []
        typeregex = regex.compile(layername, regex.IGNORECASE)
        for idx, layer in enumerate(self._layer_specs):
            name = None
            if isinstance(layer, LayerSpec):
                name = layer.typename.__name__
            elif isinstance(layer, nn.Module):
                name = layer.__class__.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    continue
            if typeregex.search(name):
                idxs.append(idx)
        if len(idxs) == 0:
            raise RuntimeError(f"Partitioning '{layername}' found no valid layers to partition.")
        return idxs

    def forward(self, forward_input):
        self.micro_offset += 1

        def exec_range_func(start, end):
            """ Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            """
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = self.base_seed * local_micro_offset + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            ds_utils.set_random_seed(new_seed)
                    inputs = layer(inputs)
                return inputs
            return exec_func
        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.forward_funcs))
            x = func(forward_input)
        else:
            num_layers = len(self.forward_funcs)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(start_idx + self.activation_checkpoint_interval, num_layers)
                funcs = self.forward_funcs[start_idx:end_idx]
                if not isinstance(x, tuple):
                    x = x,
                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(exec_range_func(start_idx, end_idx), *x)
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x

    def _partition_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe
        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages with method {method}')
        method = method.lower()
        if method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            self.parts = ds_utils.partition_balanced(weights=binary_weights, num_parts=num_stages)
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                None
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    None
            if self.loss_fn:
                try:
                    None
                except AttributeError:
                    None
        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

    def allreduce_tied_weight_gradients(self):
        """All reduce the gradients of the tied weights between tied stages"""
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            dist.all_reduce(weight.grad, group=comm['group'])

    def get_tied_weights_and_groups(self):
        weight_group_list = []
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            weight_group_list.append((weight, comm['group']))
        return weight_group_list

    def _synchronize_tied_weights(self):
        for key, comm in self.tied_comms.items():
            dist.broadcast(getattr(comm['module'], comm['weight_attr']), src=min(comm['ranks']), group=comm['group'])

    def _index_tied_modules(self):
        """ Build communication structures for tied modules. """
        tied_comms = {}
        if self._topo.get_dim('pipe') == 1:
            return tied_comms
        specs = self._layer_specs
        tie_keys = set(s.key for s in specs if isinstance(s, TiedLayerSpec))
        for key in tie_keys:
            tied_layers = []
            for idx, layer in enumerate(specs):
                if isinstance(layer, TiedLayerSpec) and layer.key == key:
                    tied_layers.append(idx)
            tied_stages = set(self.stage_owner(idx) for idx in tied_layers)
            for dp in range(self._grid.data_parallel_size):
                for mp in range(self._grid.get_slice_parallel_world_size()):
                    tied_ranks = []
                    for s in sorted(tied_stages):
                        if self._grid.get_slice_parallel_world_size() > 1:
                            tied_ranks.append(self._grid.stage_to_global(stage_id=s, data=dp, model=mp))
                        else:
                            tied_ranks.append(self._grid.stage_to_global(stage_id=s, data=dp))
                    group = dist.new_group(ranks=tied_ranks)
                    if self.global_rank in tied_ranks:
                        assert key in self.tied_modules
                        if key in self.tied_modules:
                            tied_comms[key] = {'ranks': tied_ranks, 'group': group, 'weight_attr': self.tied_weight_attrs[key], 'module': self.tied_modules[key]}
                            if self.global_rank != tied_ranks[0]:
                                for p in self.tied_modules[key].parameters():
                                    p.ds_pipe_replicated = True
        """
        if len(tied_comms) > 0:
            print(f'RANK={self.global_rank} tied_comms={tied_comms}')
        """
        return tied_comms

    def partitions(self):
        return self.parts

    def stage_owner(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers
        for stage in range(self._topo.get_dim('pipe')):
            if self.parts[stage] <= layer_idx < self.parts[stage + 1]:
                return stage
        raise RuntimeError(f'Layer {layer_idx} not owned? parts={self.parts}')

    def _set_bounds(self, start=None, stop=None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive and stop is
        exclusive. The default of None for both results in all layers being built
        locally.
        """
        self._local_start = start
        self._local_stop = stop

    def set_checkpoint_interval(self, interval):
        assert interval >= 0
        self.checkpoint_interval = interval

    def topology(self):
        """ ProcessTopology object to query process mappings. """
        return self._topo

    def mpu(self):
        return self._grid

    def num_pipeline_stages(self):
        return self._topo.get_dim('pipe')

    def ckpt_prefix(self, checkpoints_path, tag):
        """Build a prefix for all checkpoint files written by this module. """
        rank_name = 'module'
        omit_dims = frozenset(['data'])
        axes = [a for a in self._grid._topo.get_axis_names() if a not in omit_dims]
        for dim in axes:
            rank = getattr(self._grid._topo.get_coord(rank=self.global_rank), dim)
            rank_name += f'-{dim}_{rank:02d}'
        ckpt_name = os.path.join(checkpoints_path, str(tag), rank_name)
        return ckpt_name

    def ckpt_layer_path(self, ckpt_dir, local_layer_idx):
        """Customize a prefix for a specific pipeline module layer. """
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f'layer_{idx:02d}')
        rank_repr = self._grid._topo.get_rank_repr(rank=self.global_rank)
        if rank_repr != '':
            layer_ckpt_path += f'-{rank_repr}'
        layer_ckpt_path += '-model_states.pt'
        return layer_ckpt_path

    def ckpt_layer_path_list(self, ckpt_dir, local_layer_idx):
        """Get all ckpt file list for a specific pipeline module layer. """
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f'layer_{idx:02d}-')
        layer_ckpt_path += '*model_states.pt'
        ckpt_files = glob.glob(layer_ckpt_path)
        ckpt_files.sort()
        return ckpt_files

    def save_state_dict(self, save_dir, checkpoint_engine):
        dp_rank = self._grid.data_parallel_id
        dp_size = self._grid.data_parallel_size
        num_layers = len(self.forward_funcs)
        if self.checkpoint_parallel_write_pipeline:
            offsets = ds_utils.partition_uniform(num_layers, dp_size)
            start, end = offsets[dp_rank], offsets[dp_rank + 1]
        else:
            if dp_rank != 0:
                return
            start, end = 0, num_layers
        layer_list = self.forward_funcs[start:end]
        os.makedirs(save_dir, exist_ok=True)
        for idx, layer in enumerate(layer_list):
            model_ckpt_path = self.ckpt_layer_path(save_dir, start + idx)
            if not hasattr(layer, 'state_dict'):
                continue
            orig_state_dict = layer.state_dict()
            final_state_dict = type(orig_state_dict)({k: v.clone() for k, v in orig_state_dict.items()})
            checkpoint_engine.save(final_state_dict, model_ckpt_path)

    def load_state_dir(self, load_dir, checkpoint_engine, strict=True):
        for idx, layer in enumerate(self.forward_funcs):
            if not hasattr(layer, 'load_state_dict'):
                continue
            model_ckpt_list = self.ckpt_layer_path_list(load_dir, idx)
            mp_rank = self._grid.get_slice_parallel_rank()
            mp_world_size = self._grid.get_slice_parallel_world_size()
            sd_loader = SDLoaderFactory.get_sd_loader(model_ckpt_list, version=2.0, checkpoint_engine=checkpoint_engine)
            load_path, checkpoint, _ = sd_loader.load(mp_world_size, mp_rank, module_key=None, is_pipe_parallel=True)
            layer.load_state_dict(checkpoint)
        self._synchronize_tied_weights()

    def _is_checkpointable(self, funcs):
        if self.__class__.__name__ in ('GPTModelPipe', 'GPT2ModelPipe'):
            return all('ParallelTransformerLayerPipe' in f.__class__.__name__ for f in funcs)
        if self.checkpointable_layers is not None:
            return all(f.__class__.__name__ in self.checkpointable_layers for f in funcs)
        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)


class ReplaceWithTensorSlicing:

    def __init__(self, mp_group=None, mp_size=1, out_dim=1, in_dim=0):
        if mp_group is not None:
            self.gpu_index = dist.get_rank(group=mp_group)
        else:
            self.gpu_index = 0
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mp_size = mp_size

    def merge_assert(self, dim1, dim2):
        assert dim1 > dim2, 'Merging tensors is not allowed here! Please use deepspeed load_checkpoint            for merging your checkpoints before replacing the transformer layer with            inference-kernels'

    def qkv_copy(self, dst, src):
        if src is None:
            return src
        src_shape = src.shape
        dst_shape = dst.shape
        if self.out_dim == 0:
            src_split = torch.split(src.data, src_shape[self.out_dim] // self.mp_size, dim=0)
        else:
            src_split = torch.split(src.data, src.shape[-1] // 3, dim=-1)
        if len(src_shape) == 2 and len(dst_shape) == 2:
            if src_shape[self.out_dim] == dst_shape[self.out_dim]:
                return torch.nn.parameter.Parameter(src)
            if self.out_dim == 1:
                self.merge_assert(src_shape[self.out_dim], dst_shape[self.out_dim])
                qkv_size = dst_shape[self.out_dim] // 3
                qkv_split = [torch.split(src_s, qkv_size, dim=self.out_dim) for src_s in src_split]
                weight_split = [torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=self.out_dim) for i in range(len(qkv_split[0]))]
                dst.data.copy_(weight_split[self.gpu_index].contiguous())
            else:
                dst.data.copy_(src_split[self.gpu_index].contiguous())
        else:
            if src_shape[0] == dst_shape[0]:
                return torch.nn.parameter.Parameter(src)
            if self.out_dim == 1:
                qkv_size = dst_shape[0] // 3
                qkv_split = [torch.split(src_s, qkv_size, dim=0) for src_s in src_split]
                bias_split = [torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=0) for i in range(len(qkv_split[0]))]
                dst.data.copy_(bias_split[self.gpu_index].contiguous())
            else:
                dst.data.copy_(src_split[self.gpu_index].contiguous())
        return torch.nn.parameter.Parameter(dst)

    def copy(self, dst, src):
        if src is None:
            return src
        src_shape = src.shape
        dst_shape = dst.shape
        if len(src_shape) == 2 and len(dst_shape) == 2:
            if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
                dst.data.copy_(src)
            else:
                if src_shape[self.in_dim] != dst_shape[self.in_dim]:
                    self.merge_assert(src_shape[self.in_dim], dst_shape[self.in_dim])
                    weight_split = torch.split(src, dst_shape[self.in_dim], dim=self.in_dim)[self.gpu_index].contiguous()
                else:
                    self.merge_assert(src_shape[self.out_dim], dst_shape[self.out_dim])
                    weight_split = torch.split(src.data, dst_shape[self.out_dim], dim=self.out_dim)[self.gpu_index].contiguous()
                dst.data.copy_(weight_split.contiguous())
        elif src_shape[0] == dst_shape[0]:
            dst.data.copy_(src)
        else:
            bias_split = torch.split(src.data, dst_shape[-1])[self.gpu_index].contiguous()
            dst.data.copy_(bias_split)
        dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
        if hasattr(src, 'scale'):
            dst.scale = src.scale
        return dst


class CudaEventTimer(object):

    def __init__(self, start_event: torch.cuda.Event, end_event: torch.cuda.Event):
        self.start_event = start_event
        self.end_event = end_event

    def get_elapsed_msec(self):
        torch.cuda.current_stream().wait_event(self.end_event)
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def trim_mean(data, trim_percent):
    """Compute the trimmed mean of a list of numbers.

    Args:
        data (list): List of numbers.
        trim_percent (float): Percentage of data to trim.

    Returns:
        float: Trimmed mean.
    """
    assert trim_percent >= 0.0 and trim_percent <= 1.0
    n = len(data)
    if len(data) == 0:
        return 0
    data.sort()
    k = int(round(n * trim_percent))
    return mean(data[k:n - k])


class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""


    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.started_ = False
            self.event_timers = []
            self.start_event = None
            self.elapsed_records = None

        def start(self):
            """Start the timer."""
            assert not self.started_, f'{self.name_} timer has already been started'
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.started_ = True

        def stop(self, reset=False, record=False):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.event_timers.append(CudaEventTimer(self.start_event, end_event))
            self.start_event = None
            self.started_ = False

        def _get_elapsed_msec(self):
            self.elapsed_records = [et.get_elapsed_msec() for et in self.event_timers]
            self.event_timers.clear()
            return sum(self.elapsed_records)

        def reset(self):
            """Reset timer."""
            self.started_ = False
            self.start_event = None
            self.elapsed_records = None
            self.event_timers.clear()

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            if self.started_:
                self.stop()
            elapsed_ = self._get_elapsed_msec()
            if reset:
                self.reset()
            if started_:
                self.start()
            return elapsed_

        def mean(self):
            self.elapsed(reset=False)
            return trim_mean(self.elapsed_records, 0.1)

    def __init__(self):
        self.timers = {}

    def get_timers(self):
        return self.timers

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    @staticmethod
    def memory_usage():
        alloc = 'mem_allocated: {:.4f} GB'.format(torch.cuda.memory_allocated() / (1024 * 1024 * 1024))
        max_alloc = 'max_mem_allocated: {:.4f} GB'.format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024))
        cache = 'cache_allocated: {:.4f} GB'.format(torch.cuda.memory_cached() / (1024 * 1024 * 1024))
        max_cache = 'max_cache_allocated: {:.4f} GB'.format(torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))
        return ' | {} | {} | {} | {}'.format(alloc, max_alloc, cache, max_cache)

    def log(self, names, normalizer=1.0, reset=True, memory_breakdown=False, ranks=None):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = f'rank={dist.get_rank()} time (ms)'
        for name in names:
            if name in self.timers:
                elapsed_time = self.timers[name].elapsed(reset=reset) / normalizer
                string += ' | {}: {:.2f}'.format(name, elapsed_time)
        log_dist(string, ranks=ranks or [0])

    def get_mean(self, names, normalizer=1.0, reset=True):
        """Get the mean of a group of timers."""
        assert normalizer > 0.0
        means = {}
        for name in names:
            if name in self.timers:
                elapsed_time = self.timers[name].mean() * 1000.0 / normalizer
                means[name] = elapsed_time
        return means


inference_cuda_module = None


class DeepSpeedDiffusersAttentionFunction(Function):

    @staticmethod
    def forward(ctx, input, context, input_mask, config, attn_qkvw, attn_qw, attn_kw, attn_vw, attn_qkvb, num_attention_heads_per_partition, norm_factor, hidden_size_per_partition, attn_ow, attn_ob, do_out_bias, score_context_func, linear_func, triton_flash_attn_kernel):

        def _transpose_for_context(x):
            x = x.permute(0, 2, 1, 3)
            new_x_layer_shape = x.size()[:-2] + (hidden_size_per_partition,)
            return x.reshape(*new_x_layer_shape)

        def _transpose_for_scores(x):
            attention_head_size = x.shape[-1] // num_attention_heads_per_partition
            new_x_shape = x.size()[:-1] + (num_attention_heads_per_partition, attention_head_size)
            x = x.reshape(*new_x_shape)
            x = x.permute(0, 2, 1, 3)
            return x.contiguous()

        def selfAttention_fp(input, context, input_mask):
            if config.fp16 and input.dtype == torch.float32:
                input = input.half()
            head_size = input.shape[-1] // config.heads
            do_flash_attn = head_size <= 128
            scale = 1 / norm_factor * (1 / norm_factor)
            if do_flash_attn and context == None:
                qkv_out = linear_func(input, attn_qkvw, attn_qkvb if attn_qkvb is not None else attn_qkvw, attn_qkvb is not None, do_flash_attn, config.heads)
                context_layer = triton_flash_attn_kernel(qkv_out[0], qkv_out[1], qkv_out[2], scale, input.shape[-2] % 128 == 0)
                context_layer = _transpose_for_context(context_layer[:, :, :, :head_size])
            else:
                do_flash_attn = False
                if context is not None:
                    query = torch.matmul(input, attn_qw)
                    key = torch.matmul(context, attn_kw)
                    value = torch.matmul(context, attn_vw)
                else:
                    qkv = torch.matmul(input, attn_qkvw)
                    query, key, value = qkv.chunk(3, dim=-1)
                    query = query.contiguous()
                    key = key.contiguous()
                    value = value.contiguous()
                query, key, value = inference_cuda_module.pad_transform_fp16(query, key, value, config.heads, do_flash_attn)
                attention_scores = (torch.matmul(query, key.transpose(-1, -2)) * scale).softmax(dim=-1)
                context_layer = _transpose_for_context(torch.matmul(attention_scores, value))
            output = linear_func(context_layer, attn_ow, attn_ob, do_out_bias, False, config.heads)
            return output
        output = selfAttention_fp(input, context, input_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2, grad_output3):
        raise RuntimeError('You are running with DeepSpeed Inference mode.                             Please switch to Training mode for running backward!')


class triton_flash_attn(torch.nn.Module):

    def __init__(self):
        super(triton_flash_attn, self).__init__()

    def forward(self, q, k, v, sm_scale, block_128=True):
        BLOCK = 128 if block_128 else 64
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        o = torch.empty_like(q)
        grid = triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1]
        tmp = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](q, k, v, sm_scale, tmp, o, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), o.stride(0), o.stride(1), o.stride(2), o.stride(3), k.shape[0], k.shape[1], k.shape[2], BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=Lk, num_warps=num_warps, num_stages=1)
        return o


class DeepSpeedDiffusersAttention(nn.Module):
    """Initialize the DeepSpeed Transformer Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepSpeedInferenceConfig
    """
    layer_id = 0

    def __init__(self, config):
        super(DeepSpeedDiffusersAttention, self).__init__()
        self.config = config
        self.config.layer_id = DeepSpeedDiffusersAttention.layer_id
        DeepSpeedDiffusersAttention.layer_id += 1
        device = torch.cuda.current_device() if config.bigscience_bloom else 'cpu'
        qkv_size_per_partition = self.config.hidden_size // self.config.mp_size * 3
        data_type = torch.int8 if config.q_int8 else torch.half if config.fp16 else torch.float
        data_type_fp = torch.half if config.fp16 else torch.float
        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()
        if DeepSpeedDiffusersAttention.layer_id == 1:
            log_dist(f'DeepSpeed-Attention config: {self.config.__dict__}', [0])
        self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size, qkv_size_per_partition, dtype=data_type, device=device), requires_grad=False)
        self.attn_kw = nn.Parameter(torch.empty(self.config.hidden_size, self.config.hidden_size, dtype=data_type, device=device), requires_grad=False)
        self.attn_vw = nn.Parameter(torch.empty(self.config.hidden_size, self.config.hidden_size, dtype=data_type, device=device), requires_grad=False)
        self.attn_qw = nn.Parameter(torch.empty(self.config.hidden_size, self.config.hidden_size, dtype=data_type, device=device), requires_grad=False)
        self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition, dtype=data_type_fp, device=device), requires_grad=False)
        out_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition, self.config.hidden_size, dtype=data_type, device=device), requires_grad=False)
        self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device), requires_grad=False)
        self.do_out_bias = True
        if triton_flash_attn is None:
            load_triton_flash_attn()
        self.triton_flash_attn_kernel = triton_flash_attn()
        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads
        self.norm_factor = math.sqrt(math.sqrt(self.config.hidden_size // self.config.heads))
        if self.config.scale_attn_by_inverse_layer_idx is True:
            self.norm_factor *= math.sqrt(self.config.layer_id + 1)
        self.score_context_func = inference_cuda_module.softmax_context_fp32 if not config.fp16 else inference_cuda_module.softmax_context_fp16
        self.linear_func = inference_cuda_module.linear_layer_fp16 if config.fp16 else inference_cuda_module.linear_layer_fp32
        self.allocate_workspace = inference_cuda_module.allocate_workspace_fp32 if not config.fp16 else inference_cuda_module.allocate_workspace_fp16

    def forward(self, input, context=None, input_mask=None):
        if self.config.layer_id == 0:
            self.allocate_workspace(self.config.hidden_size, self.config.heads, input.size()[1], input.size()[0], DeepSpeedDiffusersAttention.layer_id, self.config.mp_size, False, 0, self.config.max_out_tokens)
        output = DeepSpeedDiffusersAttentionFunction.apply(input, context, input_mask, self.config, self.attn_qkvw, self.attn_qw, self.attn_kw, self.attn_vw, self.attn_qkvb, self.num_attention_heads_per_partition, self.norm_factor, self.hidden_size_per_partition, self.attn_ow, self.attn_ob, self.do_out_bias, self.score_context_func, self.linear_func, self.triton_flash_attn_kernel)
        return output


class Diffusers2DTransformerConfig:

    def __init__(self, int8_quantization=False):
        self.int8_quantization = int8_quantization


def load_spatial_module():
    global spatial_cuda_module
    if spatial_cuda_module is None:
        spatial_cuda_module = op_builder.SpatialInferenceBuilder().load()
    return spatial_cuda_module


def load_transformer_module():
    global transformer_cuda_module
    if transformer_cuda_module is None:
        transformer_cuda_module = op_builder.InferenceBuilder().load()
    return transformer_cuda_module


class TransformerConfig:

    def __init__(self, batch_size, hidden_size, intermediate_size, heads, attn_dropout_ratio, hidden_dropout_ratio, num_hidden_layers, initializer_range):
        self.layer_id = -1
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.heads = heads
        self.attn_dropout_ratio = attn_dropout_ratio
        self.hidden_dropout_ratio = hidden_dropout_ratio
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range


class DeepSpeedTransformerConfig(TransformerConfig):
    """Initialize the DeepSpeed Transformer Config.

        Arguments:
            batch_size: The maximum batch size used for running the kernel on each GPU

            hidden_size: The hidden size of the transformer layer

            intermediate_size: The intermediate size of the feed-forward part of transformer layer

            heads: The number of heads in the self-attention of the transformer layer

            attn_dropout_ratio: The ratio of dropout for the attention's output

            hidden_dropout_ratio: The ratio of dropout for the transformer's output

            num_hidden_layers: The number of transformer layers

            initializer_range: BERT model's initializer range for initializing parameter data

            local_rank: Optional: The rank of GPU running the transformer kernel, it is not required
                to use if the model already set the current device, otherwise need to set it
                so that the transformer kernel can work on the right device

            seed: The random seed for the dropout layers

            fp16: Enable half-precision computation

            pre_layer_norm: Select between Pre-LN or Post-LN transformer architecture

            normalize_invertible: Optional: Enable invertible LayerNorm execution (dropping the input activation),
                default is False

            gelu_checkpoint: Optional: Enable checkpointing of Gelu activation output to save memory,
                default is False

            adjust_init_range: Optional: Set as True (default) if the model adjusts the weight initial values of
                its self-attention output and layer output, False keeps the initializer_range no change.
                See the adjustment below:
                    output_std = self.config.initializer_range / math.sqrt(2.0 * num_layers)

            attn_dropout_checkpoint: Optional: Enable checkpointing of attention dropout to save memory,
                default is False

            stochastic_mode:  Enable for high performance, please note that this flag has some level of
                non-determinism and can produce different results on different runs.  However, we have seen
                that by enabling it, the pretraining tasks such as BERT are not affected and can obtain
                a high accuracy level. On the other hand, for the downstream tasks, such as fine-tuning, we recommend
                to turn it off in order to be able to reproduce the same result through the regular kernel execution.

            return_tuple: Enable if using the return_tuple interface style for sending out the forward results.

            training: Enable for training rather than inference.
    """

    def __init__(self, batch_size=-1, hidden_size=-1, intermediate_size=-1, heads=-1, attn_dropout_ratio=-1, hidden_dropout_ratio=-1, num_hidden_layers=-1, initializer_range=-1, layer_norm_eps=1e-12, local_rank=-1, seed=-1, fp16=False, pre_layer_norm=True, normalize_invertible=False, gelu_checkpoint=False, adjust_init_range=True, attn_dropout_checkpoint=False, stochastic_mode=False, return_tuple=False, training=True):
        super(DeepSpeedTransformerConfig, self).__init__(batch_size, hidden_size, intermediate_size if intermediate_size > 0 else 4 * hidden_size, heads, attn_dropout_ratio, hidden_dropout_ratio, num_hidden_layers, initializer_range)
        self.fp16 = fp16
        self.pre_layer_norm = pre_layer_norm
        self.local_rank = local_rank
        self.seed = seed
        self.normalize_invertible = normalize_invertible
        self.gelu_checkpoint = gelu_checkpoint
        self.adjust_init_range = adjust_init_range
        self.test_gemm = False
        self.layer_norm_eps = layer_norm_eps
        self.training = training
        self.is_grad_enabled = True
        self.attn_dropout_checkpoint = attn_dropout_checkpoint
        self.stochastic_mode = stochastic_mode
        self.return_tuple = return_tuple

    @classmethod
    def from_dict(cls, json_object):
        config = DeepSpeedTransformerConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r', encoding='utf-16') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))


stochastic_transformer_cuda_module = None


transformer_cuda_module = None


class DeepSpeedTransformerFunction(Function):

    @staticmethod
    def forward(ctx, input, input_mask, self, grads, layer_id, attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, inter_w, inter_b, output_w, output_b, norm_w, norm_b, config):
        cuda_module = stochastic_transformer_cuda_module if config.stochastic_mode else transformer_cuda_module
        forward_func = cuda_module.forward_fp16 if config.fp16 else cuda_module.forward_fp32
        inp_size = input.size()
        if inp_size[1] % 16 != 0:
            input = torch.cat((input, torch.randn((inp_size[0], 16 - inp_size[1] % 16, inp_size[2]), device=input.device, dtype=input.dtype)), 1)
            input_mask = torch.cat((input_mask, torch.ones((inp_size[0], input_mask.shape[1], input_mask.shape[2], 16 - inp_size[1] % 16), device=input_mask.device, dtype=input_mask.dtype) * -10000), 3)
        output, inp_norm, qkv_tf, soft_inp, ctx_bufB, attn_o_inp, add_res, ff1_inp, gelu_inp, ff2_inp, attn_prob_dropout_mask, attn_output_dropout_mask, layer_output_dropout_mask, attn_layer_norm_var, attn_layer_norm_mean, layer_norm_var, layer_norm_mean = forward_func(config.layer_id, input, input_mask, attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, inter_w, inter_b, output_w, output_b, norm_w, norm_b, config.training and config.is_grad_enabled, config.pre_layer_norm, config.attn_dropout_checkpoint, config.normalize_invertible, config.gelu_checkpoint)
        if grads is not None:
            for i in [2]:
                attn_qkvw.register_hook(lambda x, i=i, self=self: grads.append([x[i * attn_ow.size(0):(i + 1) * attn_ow.size(0)], 'Q_W' if i == 0 else 'K_W' if i == 1 else 'V_W']))
            for i in [2]:
                attn_qkvb.register_hook(lambda x, i=i, self=self: grads.append([x[i * attn_ow.size(0):(i + 1) * attn_ow.size(0)], 'Q_B' if i == 0 else 'K_B' if i == 1 else 'V_B']))
            attn_ow.register_hook(lambda x, self=self: grads.append([x, 'O_W']))
            attn_ob.register_hook(lambda x, self=self: grads.append([x, 'O_B']))
            attn_nw.register_hook(lambda x, self=self: grads.append([x, 'N2_W']))
            attn_nb.register_hook(lambda x, self=self: grads.append([x, 'N2_B']))
            inter_w.register_hook(lambda x, self=self: grads.append([x, 'int_W']))
            inter_b.register_hook(lambda x, self=self: grads.append([x, 'int_B']))
            output_w.register_hook(lambda x, self=self: grads.append([x, 'out_W']))
            output_b.register_hook(lambda x, self=self: grads.append([x, 'out_B']))
            norm_w.register_hook(lambda x, self=self: grads.append([x, 'norm_W']))
            norm_b.register_hook(lambda x, self=self: grads.append([x, 'norm_B']))
        if config.is_grad_enabled and config.training:
            if config.pre_layer_norm and config.normalize_invertible:
                ctx.save_for_backward(input_mask, attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, inter_w, inter_b, output_w, output_b, norm_w, norm_b)
            else:
                ctx.save_for_backward(output, input, input_mask, attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, inter_w, inter_b, output_w, output_b, norm_w, norm_b)
            ctx.config = config
            if config.pre_layer_norm or not config.normalize_invertible:
                ctx.inp_norm = inp_norm
            ctx.qkv_tf = qkv_tf
            ctx.soft_inp = soft_inp
            if not config.attn_dropout_checkpoint:
                ctx.ctx_bufB = ctx_bufB
            ctx.attn_o_inp = attn_o_inp
            if not config.normalize_invertible:
                ctx.add_res = add_res
            ctx.attn_layer_norm_mean = attn_layer_norm_mean
            ctx.layer_norm_mean = layer_norm_mean
            ctx.ff1_inp = ff1_inp
            if not config.gelu_checkpoint:
                ctx.gelu_inp = gelu_inp
            ctx.ff2_inp = ff2_inp
            ctx.attn_prob_dropout_mask = attn_prob_dropout_mask
            ctx.attn_output_dropout_mask = attn_output_dropout_mask
            ctx.layer_output_dropout_mask = layer_output_dropout_mask
            ctx.attn_layer_norm_var = attn_layer_norm_var
            ctx.layer_norm_var = layer_norm_var
        if inp_size[1] % 16 != 0:
            output = torch.narrow(output, 1, 0, inp_size[1])
        if config.return_tuple:
            return output,
        else:
            return output

    @staticmethod
    def backward(ctx, grad_output):
        bsz = grad_output.shape[0]
        grad_output_shape = grad_output.size()
        if grad_output_shape[1] % 16 != 0:
            grad_output = torch.cat((grad_output, torch.zeros((bsz, 16 - grad_output_shape[1] % 16, grad_output_shape[2]), device=grad_output.device, dtype=grad_output.dtype)), 1)
        assert ctx.config.training
        if ctx.config.pre_layer_norm and ctx.config.normalize_invertible:
            input_mask, attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, inter_w, inter_b, output_w, output_b, norm_w, norm_b = ctx.saved_tensors
        else:
            output, input, input_mask, attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, inter_w, inter_b, output_w, output_b, norm_w, norm_b = ctx.saved_tensors
        cuda_module = stochastic_transformer_cuda_module if ctx.config.stochastic_mode else transformer_cuda_module
        backward_func = cuda_module.backward_fp16 if ctx.config.fp16 else cuda_module.backward_fp32
        grad_input, grad_attn_qkvw, grad_attn_qkvb, grad_attn_ow, grad_attn_ob, grad_attn_nw, grad_attn_nb, grad_inter_w, grad_inter_b, grad_output_w, grad_output_b, grad_norm_w, grad_norm_b = backward_func(ctx.config.layer_id, grad_output, ctx.inp_norm if ctx.config.pre_layer_norm and ctx.config.normalize_invertible else output, ctx.inp_norm if ctx.config.pre_layer_norm or not ctx.config.normalize_invertible else input, ctx.qkv_tf, ctx.soft_inp, ctx.soft_inp if ctx.config.attn_dropout_checkpoint else ctx.ctx_bufB, ctx.attn_o_inp, ctx.ff1_inp if ctx.config.normalize_invertible else ctx.add_res, ctx.ff1_inp, ctx.ff2_inp if ctx.config.gelu_checkpoint else ctx.gelu_inp, ctx.ff2_inp, ctx.attn_prob_dropout_mask, ctx.attn_output_dropout_mask, ctx.layer_output_dropout_mask, ctx.attn_layer_norm_var, ctx.attn_layer_norm_mean, ctx.layer_norm_var, ctx.layer_norm_mean, ctx.inp_norm if ctx.config.pre_layer_norm and ctx.config.normalize_invertible else input, input_mask, attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, inter_w, inter_b, output_w, output_b, norm_w, norm_b)
        ctx.qkv_tf = None
        ctx.soft_inp = None
        ctx.ctx_bufB = None
        ctx.gelu_inp = None
        ctx.ff2_inp = None
        ctx.attn_o_inp = None
        ctx.ff1_inp = None
        ctx.add_res = None
        ctx.inp_norm = None
        ctx.config = None
        ctx.attn_layer_norm_mean = None
        ctx.layer_norm_mean = None
        ctx.attn_prob_dropout_mask = None
        ctx.attn_output_dropout_mask = None
        ctx.layer_output_dropout_mask = None
        ctx.attn_layer_norm_var = None
        ctx.layer_norm_var = None
        if grad_output_shape[1] % 16 != 0:
            grad_input = torch.narrow(grad_input, 1, 0, grad_output_shape[1])
        return grad_input, None, None, None, None, grad_attn_qkvw, grad_attn_qkvb, grad_attn_ow, grad_attn_ob, grad_attn_nw, grad_attn_nb, grad_inter_w, grad_inter_b, grad_output_w, grad_output_b, grad_norm_w, grad_norm_b, None


TORCH_MAJOR = int(torch.__version__.split('.')[0])


TORCH_MINOR = int(torch.__version__.split('.')[1])


END = '\x1b[0m'


YELLOW = '\x1b[93m'


WARNING = f'{YELLOW} [WARNING] {END}'


cuda_minor_mismatch_ok = {(10): ['10.0', '10.1', '10.2'], (11): ['11.0', '11.1', '11.2', '11.3', '11.4', '11.5', '11.6', '11.7', '11.8']}


def installed_cuda_version():
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    assert cuda_home is not None, 'CUDA_HOME does not exist, unable to compile CUDA op(s)'
    output = subprocess.check_output([cuda_home + '/bin/nvcc', '-V'], universal_newlines=True)
    output_split = output.split()
    release_idx = output_split.index('release')
    release = output_split[release_idx + 1].replace(',', '').split('.')
    cuda_major, cuda_minor = release[:2]
    installed_cuda_version = '.'.join(release[:2])
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


def get_default_compute_capabilities():
    compute_caps = DEFAULT_COMPUTE_CAPABILITIES
    import torch.utils.cpp_extension
    if torch.utils.cpp_extension.CUDA_HOME is not None and installed_cuda_version()[0] >= 11:
        if installed_cuda_version()[0] == 11 and installed_cuda_version()[1] == 0:
            compute_caps += ';8.0'
        else:
            compute_caps += ';8.0;8.6'
    return compute_caps


class DeepSpeedTransformerLayer(nn.Module):
    """Initialize the DeepSpeed Transformer Layer.

        Static variable:
            layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
            e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.
        Arguments:
            config: An object of DeepSpeedTransformerConfig

            initial_weights: Optional: Only used for unit test

            initial_biases: Optional: Only used for unit test
    """
    layer_id = 0

    def __init__(self, config, initial_weights=None, initial_biases=None):
        super(DeepSpeedTransformerLayer, self).__init__()
        self.config = config
        self.config.layer_id = DeepSpeedTransformerLayer.layer_id
        DeepSpeedTransformerLayer.layer_id = DeepSpeedTransformerLayer.layer_id + 1
        None
        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)
        if initial_weights is None and initial_biases is None:
            self.attn_qkvw = nn.Parameter(torch.Tensor(self.config.hidden_size * 3, self.config.hidden_size))
            self.attn_qkvb = nn.Parameter(torch.Tensor(self.config.hidden_size * 3))
            self.attn_ow = nn.Parameter(torch.Tensor(self.config.hidden_size, self.config.hidden_size))
            self.attn_ob = nn.Parameter(torch.Tensor(self.config.hidden_size))
            self.attn_nw = nn.Parameter(torch.Tensor(self.config.hidden_size))
            self.attn_nb = nn.Parameter(torch.Tensor(self.config.hidden_size))
            self.inter_w = nn.Parameter(torch.Tensor(self.config.intermediate_size, self.config.hidden_size))
            self.inter_b = nn.Parameter(torch.Tensor(self.config.intermediate_size))
            self.output_w = nn.Parameter(torch.Tensor(self.config.hidden_size, self.config.intermediate_size))
            self.output_b = nn.Parameter(torch.Tensor(self.config.hidden_size))
            self.norm_w = nn.Parameter(torch.Tensor(self.config.hidden_size))
            self.norm_b = nn.Parameter(torch.Tensor(self.config.hidden_size))
            self.init_transformer_weights(self.config.adjust_init_range)
        else:
            q = initial_weights[0].data
            k = initial_weights[1].data
            v = initial_weights[2].data
            self.attn_qkvw = nn.Parameter(torch.cat((q, k, v)))
            self.attn_qkvb = nn.Parameter(torch.Tensor(self.config.hidden_size * 3))
            self.attn_qkvb.data.zero_()
            self.attn_ow = initial_weights[3]
            self.attn_ob = initial_biases[3]
            self.attn_nw = initial_weights[4]
            self.attn_nb = initial_biases[4]
            self.inter_w = initial_weights[5]
            self.inter_b = initial_biases[5]
            self.output_w = initial_weights[6]
            self.output_b = initial_biases[6]
            self.norm_w = initial_weights[7]
            self.norm_b = initial_biases[7]
        global transformer_cuda_module, stochastic_transformer_cuda_module
        if transformer_cuda_module is None and not self.config.stochastic_mode:
            transformer_cuda_module = TransformerBuilder().load()
        if stochastic_transformer_cuda_module is None and self.config.stochastic_mode:
            stochastic_transformer_cuda_module = StochasticTransformerBuilder().load()
        cuda_module = stochastic_transformer_cuda_module if self.config.stochastic_mode else transformer_cuda_module
        create_layer_func = cuda_module.create_transformer_layer_fp16 if self.config.fp16 else cuda_module.create_transformer_layer_fp32
        create_layer_func(self.config.layer_id, self.config.batch_size, self.config.hidden_size, self.config.heads, self.config.intermediate_size, self.config.attn_dropout_ratio, self.config.hidden_dropout_ratio, self.config.layer_norm_eps, self.config.seed, self.config.pre_layer_norm, self.config.test_gemm, self.config.attn_dropout_checkpoint, self.config.normalize_invertible, self.config.gelu_checkpoint, self.config.stochastic_mode)

    def init_transformer_weights(self, adjust_init_range=False):
        num_layers = self.config.num_hidden_layers
        output_std = self.config.initializer_range
        if adjust_init_range and self.config.local_rank == 0:
            None
            output_std = self.config.initializer_range / math.sqrt(2.0 * num_layers)
        self.attn_qkvw.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.attn_qkvb.data.zero_()
        self.attn_ow.data.normal_(mean=0.0, std=output_std)
        self.attn_ob.data.zero_()
        self.attn_nw.data.fill_(1.0)
        self.attn_nb.data.zero_()
        self.inter_w.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.inter_b.data.zero_()
        self.output_w.data.normal_(mean=0.0, std=output_std)
        self.output_b.data.zero_()
        self.norm_w.data.fill_(1.0)
        self.norm_b.data.zero_()

    def forward(self, hidden_states, attention_mask=None, head_mask=None, layer_head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, grads=None):
        self.config.is_grad_enabled = torch.is_grad_enabled()
        return DeepSpeedTransformerFunction.apply(hidden_states, attention_mask, self, grads, self.config.layer_id, self.attn_qkvw, self.attn_qkvb, self.attn_ow, self.attn_ob, self.attn_nw, self.attn_nb, self.inter_w, self.inter_b, self.output_w, self.output_b, self.norm_w, self.norm_b, self.config)


def module_inject(layer_obj, model, config, micro_batch_size, max_seq_length, seed, preln, fp16=True):
    for name, child in model.named_children():
        if isinstance(child, layer_obj):
            None
            cuda_config = DeepSpeedTransformerConfig(batch_size=micro_batch_size, max_seq_length=max_seq_length, hidden_size=config.hidden_size, heads=config.num_attention_heads, attn_dropout_ratio=config.attention_probs_dropout_prob, hidden_dropout_ratio=config.hidden_dropout_prob, num_hidden_layers=config.num_hidden_layers, initializer_range=config.initializer_range, seed=seed, fp16=fp16, pre_layer_norm=preln)
            new_module = DeepSpeedTransformerLayer(cuda_config)
            qw = child.attention.self.query.weight
            qb = child.attention.self.query.bias
            kw = child.attention.self.key.weight
            kb = child.attention.self.key.bias
            vw = child.attention.self.value.weight
            vb = child.attention.self.value.bias
            qkvw = torch.cat((qw, kw, vw), 0)
            qkvb = torch.cat((qb, kb, vb), 0)
            new_module.attn_qkvw.data = qkvw
            new_module.attn_qkvb.data = qkvb
            new_module.attn_ow.data = child.attention.output.dense.weight
            new_module.attn_ob.data = child.attention.output.dense.bias
            if preln:
                attention_layerNorm = child.PostAttentionLayerNorm
            else:
                attention_layerNorm = child.attention.output.LayerNorm
            new_module.attn_nw.data = attention_layerNorm.weight
            new_module.attn_nb.data = attention_layerNorm.bias
            if preln:
                intermediate_FF = child.intermediate.dense_act
            else:
                intermediate_FF = child.intermediate.dense
            new_module.inter_w.data = intermediate_FF.weight
            new_module.inter_b.data = intermediate_FF.bias
            new_module.output_w.data = child.output.dense.weight
            new_module.output_b.data = child.output.dense.bias
            if preln:
                transformer_LayerNorm = child.PreAttentionLayerNorm
            else:
                transformer_LayerNorm = child.output.LayerNorm
            new_module.norm_w.data = transformer_LayerNorm.weight
            new_module.norm_b.data = transformer_LayerNorm.bias
            setattr(model, name, copy.deepcopy(new_module))
        else:
            module_inject(layer_obj, child, config, micro_batch_size, max_seq_length, seed, preln, fp16)
    return model


def nhwc_bias_add(activation: torch.Tensor, bias: torch.Tensor, other: Optional[torch.Tensor]=None, other_bias: Optional[torch.Tensor]=None) ->torch.Tensor:
    global spatial_cuda_module
    if spatial_cuda_module is None:
        spatial_cuda_module = op_builder.SpatialInferenceBuilder().load()
    if other is None:
        return spatial_cuda_module.nhwc_bias_add(activation, bias)
    elif other_bias is None:
        return spatial_cuda_module.nhwc_bias_add_add(activation, bias, other)
    else:
        return spatial_cuda_module.nhwc_bias_add_bias_add(activation, bias, other, other_bias)


class DeepSpeedDiffusersTransformerBlock(nn.Module):

    def __init__(self, equivalent_module: nn.Module, config: Diffusers2DTransformerConfig):
        super(DeepSpeedDiffusersTransformerBlock, self).__init__()
        self.quantizer = module_inject.GroupQuantizer(q_int8=config.int8_quantization)
        self.config = config
        self.ff1_w = self.quantizer.quantize(nn.Parameter(equivalent_module.ff.net[0].proj.weight.data, requires_grad=False))
        self.ff1_b = nn.Parameter(equivalent_module.ff.net[0].proj.bias.data, requires_grad=False)
        self.ff2_w = self.quantizer.quantize(nn.Parameter(equivalent_module.ff.net[2].weight.data, requires_grad=False))
        self.ff2_b = nn.Parameter(equivalent_module.ff.net[2].bias.data, requires_grad=False)
        self.norm1_g = nn.Parameter(equivalent_module.norm1.weight.data, requires_grad=False)
        self.norm1_b = nn.Parameter(equivalent_module.norm1.bias.data, requires_grad=False)
        self.norm1_eps = equivalent_module.norm1.eps
        self.norm2_g = nn.Parameter(equivalent_module.norm2.weight.data, requires_grad=False)
        self.norm2_b = nn.Parameter(equivalent_module.norm2.bias.data, requires_grad=False)
        self.norm2_eps = equivalent_module.norm2.eps
        self.norm3_g = nn.Parameter(equivalent_module.norm3.weight.data, requires_grad=False)
        self.norm3_b = nn.Parameter(equivalent_module.norm3.bias.data, requires_grad=False)
        self.norm3_eps = equivalent_module.norm3.eps
        self.attn_1 = equivalent_module.attn1
        self.attn_2 = equivalent_module.attn2
        if isinstance(self.attn_1, DeepSpeedDiffusersAttention):
            self.attn_1.do_out_bias = False
            self.attn_1_bias = self.attn_1.attn_ob
        else:
            self.attn_1_bias = nn.Paramaeter(torch.zeros_like(self.norm2_g), requires_grad=False)
        if isinstance(self.attn_2, DeepSpeedDiffusersAttention):
            self.attn_2.do_out_bias = False
            self.attn_2_bias = self.attn_2.attn_ob
        else:
            self.attn_2_bias = nn.Paramaeter(torch.zeros_like(self.norm3_g), requires_grad=False)
        self.transformer_cuda_module = load_transformer_module()
        load_spatial_module()

    def forward(self, hidden_states, context=None, timestep=None):
        out_norm_1 = self.transformer_cuda_module.layer_norm(hidden_states, self.norm1_g, self.norm1_b, self.norm1_eps)
        out_attn_1 = self.attn_1(out_norm_1)
        out_norm_2, out_attn_1 = self.transformer_cuda_module.layer_norm_residual_store_pre_ln_res(out_attn_1, self.attn_1_bias, hidden_states, self.norm2_g, self.norm2_b, self.norm2_eps)
        out_attn_2 = self.attn_2(out_norm_2, context=context)
        out_norm_3, out_attn_2 = self.transformer_cuda_module.layer_norm_residual_store_pre_ln_res(out_attn_2, self.attn_2_bias, out_attn_1, self.norm3_g, self.norm3_b, self.norm3_eps)
        out_ff1 = nn.functional.linear(out_norm_3, self.ff1_w)
        out_geglu = self.transformer_cuda_module.bias_geglu(out_ff1, self.ff1_b)
        out_ff2 = nn.functional.linear(out_geglu, self.ff2_w)
        return nhwc_bias_add(out_ff2, self.ff2_b, other=out_attn_2)


class UNetPolicy(DSPolicy):

    def __init__(self):
        super().__init__()
        try:
            self._orig_layer_class = diffusers.models.unet_2d_condition.UNet2DConditionModel
        except ImportError:
            self._orig_layer_class = None

    def match(self, module):
        return isinstance(module, self._orig_layer_class)

    def apply(self, module, enable_cuda_graph=True):
        return DSUNet(module, enable_cuda_graph=enable_cuda_graph)

    def attention(self, client_module):
        qw = client_module.to_q.weight
        kw = client_module.to_k.weight
        vw = client_module.to_v.weight
        if qw.shape[1] == kw.shape[1]:
            qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)
            return qkvw, client_module.to_out[0].weight, client_module.to_out[0].bias, qw.shape[-1], client_module.heads
        else:
            return qw, kw, vw, client_module.to_out[0].weight, client_module.to_out[0].bias, qw.shape[-1], client_module.heads


class VAEPolicy(DSPolicy):

    def __init__(self):
        super().__init__()
        try:
            self._orig_layer_class = diffusers.models.vae.AutoencoderKL
        except ImportError:
            self._orig_layer_class = None

    def match(self, module):
        return isinstance(module, self._orig_layer_class)

    def apply(self, module, enable_cuda_graph=True):
        return DSVAE(module, enable_cuda_graph=enable_cuda_graph)


generic_policies = [UNetPolicy, VAEPolicy]


def _module_match(module):
    for policy in generic_policies:
        policy = policy()
        if policy.match(module):
            return policy
    return None


def generic_injection(module, fp16=False, enable_cuda_graph=True):

    def replace_attn(child, policy):
        policy_attn = policy.attention(child)
        if policy_attn is None:
            return child
        if len(policy_attn) == 5:
            qkvw, attn_ow, attn_ob, hidden_size, heads = policy_attn
        else:
            qw, kw, vw, attn_ow, attn_ob, hidden_size, heads = policy_attn
        config = transformer_inference.DeepSpeedInferenceConfig(hidden_size=hidden_size, heads=heads, fp16=fp16, triangular_masking=False, max_out_tokens=4096)
        attn_module = DeepSpeedDiffusersAttention(config)

        def transpose(data):
            data = data.contiguous()
            data.reshape(-1).copy_(data.transpose(-1, -2).contiguous().reshape(-1))
            data = data.reshape(data.shape[-1], data.shape[-2])
            data
            return data
        if len(policy_attn) == 5:
            attn_module.attn_qkvw.data = transpose(qkvw.data)
        else:
            attn_module.attn_qkvw = None
            attn_module.attn_qw.data = transpose(qw.data)
            attn_module.attn_kw.data = transpose(kw.data)
            attn_module.attn_vw.data = transpose(vw.data)
        attn_module.attn_qkvb = None
        attn_module.attn_ow.data = transpose(attn_ow.data)
        attn_module.attn_ob.data.copy_(attn_ob.data)
        return attn_module

    def replace_attn_block(child, policy):
        config = Diffusers2DTransformerConfig()
        return DeepSpeedDiffusersTransformerBlock(child, config)
    if isinstance(module, torch.nn.Module):
        pass
    else:
        if fp16 is False:
            raise ValueError('Generic injection only supported with FP16')
        try:
            cross_attention = diffusers.models.attention.CrossAttention
            attention_block = diffusers.models.attention.BasicTransformerBlock
            new_policies = {cross_attention: replace_attn, attention_block: replace_attn_block}
        except ImportError:
            new_policies = {}
        cg_encoder = DSClipEncoder(module.text_encoder, enable_cuda_graph=enable_cuda_graph)
        setattr(module, 'text_encoder', cg_encoder)
        for name in module.__dict__.keys():
            sub_module = getattr(module, name)
            policy = _module_match(sub_module)
            if policy is not None:

                def _replace_module(module, policy):
                    for name, child in module.named_children():
                        _replace_module(child, policy)
                        if child.__class__ in new_policies:
                            replaced_module = new_policies[child.__class__](child, policy)
                            setattr(module, name, replaced_module)
                _replace_module(sub_module, policy)
                new_module = policy.apply(sub_module, enable_cuda_graph=enable_cuda_graph)
                None
                setattr(module, name, new_module)


class Experts(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(Experts, self).__init__()
        self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        for expert in self.deepspeed_experts:
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.deepspeed_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]
            expert_outputs += [out]
        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output


class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: torch.distributed.ProcessGroup, input: Tensor) ->Tensor:
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) ->Tuple[None, Tensor]:
        return None, _AllToAll.apply(ctx.group, *grad_output)


def _drop_tokens(input_, dim=0):
    """Divide a tensor among the tensor parallel ranks"""
    mpu = deepspeed.utils.groups.mpu
    total_chunks = mpu.get_tensor_model_parallel_world_size()
    this_chunk = mpu.get_tensor_model_parallel_rank()
    assert input_.shape[dim] % total_chunks == 0, f'input dimension {dim} ({input_.shape[dim]}) is not divisible by tensor parallel world size ({total_chunks})'
    chunk_size = input_.shape[dim] // total_chunks
    return torch.narrow(input_, dim, this_chunk * chunk_size, chunk_size)


def _gather_tokens(input_, dim=0):
    """Gather tensors and concatenate them along a dimension"""
    mpu = deepspeed.utils.groups.mpu
    input_ = input_.contiguous()
    rank = mpu.get_tensor_model_parallel_rank()
    tensor_list = [torch.empty_like(input_) for _ in range(mpu.get_tensor_model_parallel_world_size())]
    tensor_list[rank] = input_
    deepspeed.comm.all_gather(tensor_list, input_, group=mpu.get_tensor_model_parallel_group())
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output


class _DropTokens(torch.autograd.Function):
    """Divide tokens equally among the tensor parallel ranks"""

    @staticmethod
    def symbolic(graph, input_, dim):
        return _drop_tokens(input_, dim)

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _drop_tokens(input_, dim)

    @staticmethod
    def backward(ctx, input_):
        return _gather_tokens(input_, ctx.dim), None


def drop_tokens(input_, dim=0):
    mpu = deepspeed.utils.groups.mpu
    if mpu is None or mpu.get_tensor_model_parallel_world_size() == 1:
        return input_
    return _DropTokens.apply(input_, dim)


USE_EINSUM = True


def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        a = a.t().unsqueeze(1)
        b = b.reshape(k, -1).t().reshape(s, m, k)
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


class _GatherTokens(torch.autograd.Function):
    """All gather tokens among the tensor parallel ranks"""

    @staticmethod
    def symbolic(graph, input_, dim):
        return _gather_tokens(input_, dim)

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _gather_tokens(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_tokens(grad_output, ctx.dim), None


def gather_tokens(input_, dim=0):
    mpu = deepspeed.utils.groups.mpu
    if mpu is None or mpu.get_tensor_model_parallel_world_size() == 1:
        return input_
    return _GatherTokens.apply(input_, dim)


def multiplicative_jitter(x, device: torch.device, epsilon=0.01):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=torch.tensor(1.0 - epsilon, device=device), high=torch.tensor(1.0 + epsilon, device=device)).rsample
        uniform_map[device] = uniform
    return x * uniform(x.shape)


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) ->Tensor:
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = torch.ceil(num_tokens / num_experts * capacity_factor)
    if capacity < min_capacity:
        capacity = min_capacity
    return capacity


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


def gumbel_rsample(shape: Tuple, device: torch.device) ->Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample
        gumbel_map[device] = gumbel
    return gumbel(shape)


def top1gating(logits: Tensor, capacity_factor: float, min_capacity: int, used_token: Tensor=None, noisy_gate_policy: Optional[str]=None, drop_tokens: bool=True, use_rts: bool=True, use_tutel: bool=False) ->Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    gates = F.softmax(logits, dim=1)
    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    if used_token is not None:
        mask1 = einsum('s,se->se', used_token, mask1)
    exp_counts = torch.sum(mask1, dim=0).detach()
    if not drop_tokens:
        new_capacity = torch.max(exp_counts)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        capacity = new_capacity
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device), high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform
        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1
    assert logits.shape[0] >= min_capacity, 'No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size.'
    top_idx = _top_idx(mask1_rand, capacity)
    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1
    if use_tutel:
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1
    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [indices1_s], [locations1_s], [gates1_s], exp_counts
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    mask1_float = mask1.float()
    gates = gates * mask1_float
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum('se,sc->sec', gates, locations1_sc)
    dispatch_mask = combine_weights.bool()
    return l_aux, combine_weights, dispatch_mask, exp_counts


def top2gating(logits: Tensor, capacity_factor: float, min_capacity: int) ->Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    gates = F.softmax(logits, dim=1)
    capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float('-inf'))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    exp_counts = torch.sum(mask1, dim=0).detach()
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum('se,se->s', gates, mask1_float)
    gates2_s = einsum('se,se->s', gates, mask2_float)
    denom_s = gates1_s + gates2_s
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    gates1 = einsum('s,se->se', gates1_s, mask1_float)
    gates2 = einsum('s,se->se', gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum('se,sc->sec', gates1, locations1_sc)
    combine2_sec = einsum('se,sc->sec', gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    return l_aux, combine_weights, dispatch_mask, exp_counts


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """
    wg: torch.nn.Linear

    def __init__(self, model_dim: int, num_experts: int, k: int=1, capacity_factor: float=1.0, eval_capacity_factor: float=1.0, min_capacity: int=8, noisy_gate_policy: Optional[str]=None, drop_tokens: bool=True, use_rts: bool=True) ->None:
        super().__init__()
        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts

    def forward(self, input: torch.Tensor, used_token: torch.Tensor=None, use_tutel: bool=False) ->Tuple[Tensor, Tensor, Tensor]:
        if self.wall_clock_breakdown:
            self.timers('TopKGate').start()
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(input_fp32)
        if self.k == 1:
            gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor, self.min_capacity, used_token, self.noisy_gate_policy if self.training else None, self.drop_tokens, self.use_rts, use_tutel)
        else:
            gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor, self.min_capacity)
        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(reset=False)
        return gate_output


class MoE(torch.nn.Module):
    """Initialize an MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        use_residual (bool, optional): default=False, make this MoE layer a Residual MoE (https://arxiv.org/abs/2201.05596) layer.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        enable_expert_tensor_parallelism (bool, optional): default=False, whether to use tensor parallelism for experts
    """

    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1, capacity_factor=1.0, eval_capacity_factor=1.0, min_capacity=4, use_residual=False, noisy_gate_policy: typing.Optional[str]=None, drop_tokens: bool=True, use_rts=True, use_tutel: bool=False, enable_expert_tensor_parallelism: bool=False):
        super(MoE, self).__init__()
        self.use_residual = use_residual
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        assert num_experts % ep_size == 0, f'Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})'
        self.ep_size = ep_size
        self.expert_group_name = f'ep_size_{self.ep_size}'
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        log_dist(f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}', [0])
        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], 'Unsupported noisy_gate_policy: ' + noisy_gate_policy
        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = MOELayer(TopKGate(hidden_size, num_experts, k, capacity_factor, eval_capacity_factor, min_capacity, noisy_gate_policy, drop_tokens, use_rts), experts, self.expert_group_name, self.ep_size, self.num_local_experts, use_tutel=use_tutel)
        if self.use_residual:
            self.mlp = expert
            self.coefficient = torch.nn.Linear(hidden_size, 2)

    def set_deepspeed_parallelism(self):
        self._create_process_groups()

    def _create_process_groups(self):
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            None
            if groups.mpu is None or not self.enable_expert_tensor_parallelism:
                groups._create_expert_and_data_parallel(self.ep_size)
            else:
                groups._create_expert_data_and_model_parallel(self.ep_size, mpu=groups.mpu)
        self.deepspeed_moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        if self.use_residual:
            output_mlp = self.mlp(hidden_states)
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts


def has_moe_layers(m):
    has_moe = False
    num_experts = 0
    for _, module in m.named_modules():
        if isinstance(module, MoE):
            has_moe = True
            num_experts = module.num_experts
            break
    return has_moe, num_experts


TORCH_DISTRIBUTED_DEFAULT_PORT = 29500


class Backend(object):

    def __init__(self, name='backend', rank=0, size=1):
        self.name = name
        self.world_group = None
        self.world_size = rank
        self.world_rank = size
        self.process_groups = []
        self.initialized = False

    def is_initialized(self):
        return self.initialized

    def new_group(self):
        pass

    def init_process_group(self):
        self.initialized = True


class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    AVG = 7
    UNUSED = 8


def has_allgather_base():
    """
        Helper to check if torch.distributed has _all_gather_base
    """
    return hasattr(torch.distributed, '_all_gather_base')


def has_reduce_scatter_base():
    """
        Helper to check if torch.distributed has _reduce_scatter_base
    """
    return hasattr(torch.distributed, '_reduce_scatter_base')


def older_torch():
    """
        Helper to lookup torch version. For versions less than 1.8, torch.dist
        used torch.distributed.group.WORLD as the default group argument instead of None.
        See more details at: https://github.com/pytorch/pytorch/pull/48767
    """
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
        return True
    else:
        return False


class TorchBackend(Backend):
    """
        A light-weight wrapper class for torch.distributed API.
        Only a subset of functions are wrapped. Once the init_process_group
        is initialized, standard torch.distributed.* can be used directly
        so no need to wrap all the functions. We can keep adding wrappers as
        needed.
    """

    def __init__(self, backend, timeout, init_method, name='torch'):
        super(TorchBackend, self).__init__()
        self.torch_version_before_18 = older_torch()
        self.has_allgather_base = has_allgather_base()
        self.has_reduce_scatter_base = has_reduce_scatter_base()
        self.initialized = True
        self.name = name
        self.single_gpu_mode = True
        self.init_process_group(backend, timeout, init_method)

    def init_process_group(self, backend, timeout, init_method):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend, timeout=timeout, init_method=init_method)
        self.using_mpi = torch.distributed.get_backend() == 'mpi'

    def all_reduce(self, tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        op = self._reduce_op(op)
        return torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
        return torch.distributed.reduce(tensor=tensor, dst=dst, op=self._reduce_op(op), group=group, async_op=async_op)

    def reduce_scatter(self, output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
        return torch.distributed.reduce_scatter(output=output, input_list=input_list, op=self._reduce_op(op), group=group, async_op=async_op)

    def broadcast(self, tensor, src, group=None, async_op=False):
        return torch.distributed.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        return torch.distributed.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)

    def all_gather_base(self, output_tensor, input_tensor, group=None, async_op=False):
        if self.has_allgather_base:
            return torch.distributed.distributed_c10d._all_gather_base(output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op)
        else:
            utils.logger.warning('unable to find torch.distributed._all_gather_base. will fall back to torch.distributed.reduce_scatter which will result in suboptimal performance. please consider upgrading your pytorch installation.')
            pass

    def reduce_scatter_base(self, output_tensor, input_tensor, op=ReduceOp.SUM, group=None, async_op=False):
        if self.has_reduce_scatter_base:
            return torch.distributed._reduce_scatter_base(output_tensor, input_tensor, op=self._reduce_op(op), group=group, async_op=async_op)
        else:
            utils.logger.warning('unable to find torch.distributed._reduce_scatter_base. will fall back to torch.distributed.reduce_scatter which will result in suboptimal performance. please consider upgrading your pytorch installation.')
            pass

    def all_to_all_single(self, output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
        return torch.distributed.all_to_all_single(output=output, input=input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group, async_op=async_op)

    def send(self, tensor, dst, group=None, tag=0):
        return torch.distributed.send(tensor=tensor, dst=dst, group=group, tag=tag)

    def recv(self, tensor, src=None, group=None, tag=0):
        return torch.distributed.recv(tensor=tensor, src=src, group=group, tag=tag)

    def isend(self, tensor, dst, group=None, tag=0):
        return torch.distributed.isend(tensor=tensor, dst=dst, group=group, tag=tag)

    def irecv(self, tensor, src=None, group=None, tag=0):
        return torch.distributed.irecv(tensor=tensor, src=src, group=group, tag=tag)

    def gather(self, tensor, gather_list=None, dst=0, group=None, async_op=False):
        return torch.distributed.gather(tensor=tensor, gather_list=gather_list, dst=dst, group=group, async_op=async_op)

    def scatter(self, tensor, scatter_list=None, src=0, group=None, async_op=False):
        return torch.distributed.scatter(tensor=tensor, scatter_list=scatter_list, src=src, group=group, async_op=async_op)

    def barrier(self, group=torch.distributed.GroupMember.WORLD, async_op=False, device_ids=None):
        if group is None:
            group = torch.distributed.GroupMember.WORLD
        return torch.distributed.barrier(group=group, async_op=async_op, device_ids=device_ids)

    def monitored_barrier(self, group=torch.distributed.GroupMember.WORLD, timeout=None, wait_all_ranks=False):
        if group is None:
            group = torch.distributed.GroupMember.WORLD
        return torch.distributed.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=wait_all_ranks)

    def get_rank(self, group=None):
        return torch.distributed.get_rank(group=group)

    def get_world_size(self, group=None):
        return torch.distributed.get_world_size(group=group)

    def is_initialized(self):
        return torch.distributed.is_initialized()

    def get_backend(self, group=None):
        return torch.distributed.get_backend(group=group)

    def new_group(self, ranks):
        return torch.distributed.new_group(ranks)

    def get_global_rank(self, group, group_rank):
        if hasattr(torch.distributed.distributed_c10d, 'get_global_rank'):
            from torch.distributed.distributed_c10d import get_global_rank as _get_global_rank
        else:
            from torch.distributed.distributed_c10d import _get_global_rank
        return _get_global_rank(group, group_rank)

    def get_world_group(self):
        return torch.distributed.group.WORLD

    def destroy_process_group(self, group=None):
        return torch.distributed.destroy_process_group(group=group)

    def _reduce_op(self, op):
        """
            Helper function. If the op provided is not a torch.dist.ReduceOp, convert it and return
        """
        if not isinstance(op, torch.distributed.ReduceOp):
            if op == ReduceOp.SUM:
                op = torch.distributed.ReduceOp.SUM
            elif op == ReduceOp.PRODUCT:
                op = torch.distributed.ReduceOp.PRODUCT
            elif op == ReduceOp.AVG:
                op = torch.distributed.ReduceOp.AVG
            elif op == ReduceOp.MIN:
                op = torch.distributed.ReduceOp.MIN
            elif op == ReduceOp.MAX:
                op = torch.distributed.ReduceOp.MAX
            elif op == ReduceOp.BAND:
                op = torch.distributed.ReduceOp.BAND
            elif op == ReduceOp.BOR:
                op = torch.distributed.ReduceOp.BOR
            elif op == ReduceOp.BXOR:
                op = torch.distributed.ReduceOp.BXOR
        return op


def _configure_defaults():
    global mpu, num_layers, deepspeed_checkpointing_enabled
    global PARTITION_ACTIVATIONS, CONTIGUOUS_CHECKPOINTING, CPU_CHECKPOINT, SYNCHRONIZE, PROFILE_TIME
    PARTITION_ACTIVATIONS = False
    CONTIGUOUS_CHECKPOINTING = False
    num_layers = False
    CPU_CHECKPOINT = False
    SYNCHRONIZE = False
    PROFILE_TIME = False
    deepspeed_checkpointing_enabled = True


ADAGRAD_OPTIMIZER = 'adagrad'


ADAMW_OPTIMIZER = 'adamw'


ADAM_OPTIMIZER = 'adam'


LAMB_OPTIMIZER = 'lamb'


ONEBIT_ADAM_OPTIMIZER = 'onebitadam'


ONEBIT_LAMB_OPTIMIZER = 'onebitlamb'


ZERO_ONE_ADAM_OPTIMIZER = 'zerooneadam'


DEEPSPEED_OPTIMIZERS = [ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER, ZERO_ONE_ADAM_OPTIMIZER]


ACT_CHKPT = 'activation_checkpointing'


ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION = 'contiguous_memory_optimization'


ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION_DEFAULT = False


ACT_CHKPT_CPU_CHECKPOINTING = 'cpu_checkpointing'


ACT_CHKPT_CPU_CHECKPOINTING_DEFAULT = False


ACT_CHKPT_NUMBER_CHECKPOINTS = 'number_checkpoints'


ACT_CHKPT_NUMBER_CHECKPOINTS_DEFAULT = None


ACT_CHKPT_PARTITION_ACTIVATIONS = 'partition_activations'


ACT_CHKPT_PARTITION_ACTIVATIONS_DEFAULT = False


ACT_CHKPT_PROFILE = 'profile'


ACT_CHKPT_PROFILE_DEFAULT = False


ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY = 'synchronize_checkpoint_boundary'


ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY_DEFAULT = False


ACT_CHKPT_DEFAULT = {ACT_CHKPT_PARTITION_ACTIVATIONS: ACT_CHKPT_PARTITION_ACTIVATIONS_DEFAULT, ACT_CHKPT_NUMBER_CHECKPOINTS: ACT_CHKPT_NUMBER_CHECKPOINTS_DEFAULT, ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION: ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION_DEFAULT, ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY: ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY_DEFAULT, ACT_CHKPT_PROFILE: ACT_CHKPT_PROFILE_DEFAULT, ACT_CHKPT_CPU_CHECKPOINTING: ACT_CHKPT_CPU_CHECKPOINTING_DEFAULT}


class DeepSpeedConfigObject(object):
    """
    For json serialization
    """

    def repr(self):
        return self.__dict__

    def __repr__(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4, cls=ScientificNotationEncoder)


def get_scalar_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


class DeepSpeedActivationCheckpointingConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        super(DeepSpeedActivationCheckpointingConfig, self).__init__()
        self.partition_activations = None
        self.contiguous_memory_optimization = None
        self.cpu_checkpointing = None
        self.number_checkpoints = None
        self.synchronize_checkpoint_boundary = None
        self.profile = None
        if ACT_CHKPT in param_dict.keys():
            act_chkpt_config_dict = param_dict[ACT_CHKPT]
        else:
            act_chkpt_config_dict = ACT_CHKPT_DEFAULT
        self._initialize(act_chkpt_config_dict)

    def _initialize(self, act_chkpt_config_dict):
        self.partition_activations = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_PARTITION_ACTIVATIONS, ACT_CHKPT_PARTITION_ACTIVATIONS_DEFAULT)
        self.contiguous_memory_optimization = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION, ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION_DEFAULT)
        self.cpu_checkpointing = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_CPU_CHECKPOINTING, ACT_CHKPT_CPU_CHECKPOINTING_DEFAULT)
        self.number_checkpoints = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_NUMBER_CHECKPOINTS, ACT_CHKPT_NUMBER_CHECKPOINTS_DEFAULT)
        self.profile = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_PROFILE, ACT_CHKPT_PROFILE_DEFAULT)
        self.synchronize_checkpoint_boundary = get_scalar_param(act_chkpt_config_dict, ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY, ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY_DEFAULT)


AUTOTUNING = 'autotuning'


AUTOTUNING_ARG_MAPPINGS = 'arg_mappings'


AUTOTUNING_ARG_MAPPINGS_DEFAULT = None


AUTOTUNING_ENABLED = 'enabled'


AUTOTUNING_ENABLED_DEFAULT = False


AUTOTUNING_END_PROFILE_STEP = 'end_profile_step'


AUTOTUNING_END_PROFILE_STEP_DEFAULT = 5


AUTOTUNING_EXPS_DIR = 'exps_dir'


AUTOTUNING_EXPS_DIR_DEFAULT = 'autotuning_exps'


AUTOTUNING_FAST = 'fast'


AUTOTUNING_FAST_DEFAULT = True


AUTOTUNING_MAX_TRAIN_BATCH_SIZE = 'max_train_batch_size'


AUTOTUNING_MAX_TRAIN_BATCH_SIZE_DEFAULT = None


AUTOTUNING_MAX_TRAIN_MICRO_BATCH_SIZE_PER_GPU = 'max_train_micro_batch_size_per_gpu'


AUTOTUNING_MAX_TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT = 1024


AUTOTUNING_METRIC = 'metric'


AUTOTUNING_METRIC_THROUGHPUT = 'throughput'


AUTOTUNING_METRIC_DEFAULT = AUTOTUNING_METRIC_THROUGHPUT


AUTOTUNING_METRIC_PATH = 'metric_path'


AUTOTUNING_METRIC_PATH_DEFAULT = None


AUTOTUNING_MIN_TRAIN_BATCH_SIZE = 'min_train_batch_size'


AUTOTUNING_MIN_TRAIN_BATCH_SIZE_DEFAULT = 1


AUTOTUNING_MIN_TRAIN_MICRO_BATCH_SIZE_PER_GPU = 'min_train_micro_batch_size_per_gpu'


AUTOTUNING_MIN_TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT = 1


AUTOTUNING_MODEL_INFO_PATH = 'model_info_path'


AUTOTUNING_MODEL_INFO_PATH_DEFAULT = None


AUTOTUNING_MP_SIZE = 'mp_size'


AUTOTUNING_MP_SIZE_DEFAULT = 1


AUTOTUNING_NUM_TUNING_MICRO_BATCH_SIZES = 'num_tuning_micro_batch_sizes'


AUTOTUNING_NUM_TUNING_MICRO_BATCH_SIZES_DEFAULT = 3


AUTOTUNING_OVERWRITE = 'overwrite'


AUTOTUNING_OVERWRITE_DEFAULT = True


AUTOTUNING_RESULTS_DIR = 'results_dir'


AUTOTUNING_RESULTS_DIR_DEFAULT = 'autotuning_results'


AUTOTUNING_START_PROFILE_STEP = 'start_profile_step'


AUTOTUNING_START_PROFILE_STEP_DEFAULT = 3


AUTOTUNING_TUNER_EARLY_STOPPING = 'tuner_early_stopping'


AUTOTUNING_TUNER_EARLY_STOPPING_DEFAULT = 5


AUTOTUNING_TUNER_NUM_TRIALS = 'tuner_num_trials'


AUTOTUNING_TUNER_NUM_TRIALS_DEFAULT = 50


AUTOTUNING_TUNER_TYPE = 'tuner_type'


AUTOTUNING_TUNER_GRIDSEARCH = 'gridsearch'


AUTOTUNING_TUNER_TYPE_DEFAULT = AUTOTUNING_TUNER_GRIDSEARCH


def get_dict_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


MODEL_INFO = 'model_info'


MODEL_INFO_HIDDEN_SIZE = 'hideen_size'


MODEL_INFO_HIDDEN_SIZE_DEFAULT = None


MODEL_INFO_NUM_LAYERS = 'num_layers'


MODEL_INFO_NUM_LAYERS_DEFAULT = None


MODEL_INFO_NUM_PARAMS = 'num_params'


MODEL_INFO_NUM_PARAMS_DEFAULT = None


MODEL_INFO_PROFILE = 'profile'


MODEL_INFO_PROFILE_DEFAULT = False


MODEL_INFO_KEY_DEFAULT_DICT = {MODEL_INFO_PROFILE: MODEL_INFO_PROFILE_DEFAULT, MODEL_INFO_NUM_PARAMS: MODEL_INFO_NUM_PARAMS_DEFAULT, MODEL_INFO_HIDDEN_SIZE: MODEL_INFO_HIDDEN_SIZE_DEFAULT, MODEL_INFO_NUM_LAYERS: MODEL_INFO_NUM_LAYERS_DEFAULT}


def get_model_info_config(param_dict):
    if MODEL_INFO in param_dict and param_dict[MODEL_INFO] is not None:
        model_info_config = {}
        for key, default_value in MODEL_INFO_KEY_DEFAULT_DICT.items():
            model_info_config[key] = get_scalar_param(param_dict[MODEL_INFO], key, default_value)
        return model_info_config
    return None


class DeepSpeedAutotuningConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        super(DeepSpeedAutotuningConfig, self).__init__()
        self.enabled = None
        self.start_step = None
        self.end_step = None
        self.metric_path = None
        self.arg_mappings = None
        self.metric = None
        self.model_info = None
        self.results_dir = None
        self.exps_dir = None
        self.overwrite = None
        if param_dict and AUTOTUNING in param_dict.keys():
            autotuning_dict = param_dict[AUTOTUNING]
        else:
            autotuning_dict = {}
        self._initialize(autotuning_dict)

    def _initialize(self, autotuning_dict):
        self.enabled = get_scalar_param(autotuning_dict, AUTOTUNING_ENABLED, AUTOTUNING_ENABLED_DEFAULT)
        self.fast = get_scalar_param(autotuning_dict, AUTOTUNING_FAST, AUTOTUNING_FAST_DEFAULT)
        self.results_dir = get_scalar_param(autotuning_dict, AUTOTUNING_RESULTS_DIR, AUTOTUNING_RESULTS_DIR_DEFAULT)
        assert self.results_dir, 'results_dir cannot be empty'
        self.exps_dir = get_scalar_param(autotuning_dict, AUTOTUNING_EXPS_DIR, AUTOTUNING_EXPS_DIR_DEFAULT)
        assert self.exps_dir, 'exps_dir cannot be empty'
        self.overwrite = get_scalar_param(autotuning_dict, AUTOTUNING_OVERWRITE, AUTOTUNING_OVERWRITE_DEFAULT)
        self.start_profile_step = get_scalar_param(autotuning_dict, AUTOTUNING_START_PROFILE_STEP, AUTOTUNING_START_PROFILE_STEP_DEFAULT)
        self.end_profile_step = get_scalar_param(autotuning_dict, AUTOTUNING_END_PROFILE_STEP, AUTOTUNING_END_PROFILE_STEP_DEFAULT)
        self.metric = get_scalar_param(autotuning_dict, AUTOTUNING_METRIC, AUTOTUNING_METRIC_DEFAULT)
        self.metric_path = get_scalar_param(autotuning_dict, AUTOTUNING_METRIC_PATH, AUTOTUNING_METRIC_PATH_DEFAULT)
        self.tuner_type = get_scalar_param(autotuning_dict, AUTOTUNING_TUNER_TYPE, AUTOTUNING_TUNER_TYPE_DEFAULT)
        self.tuner_early_stopping = get_scalar_param(autotuning_dict, AUTOTUNING_TUNER_EARLY_STOPPING, AUTOTUNING_TUNER_EARLY_STOPPING_DEFAULT)
        self.tuner_num_trials = get_scalar_param(autotuning_dict, AUTOTUNING_TUNER_NUM_TRIALS, AUTOTUNING_TUNER_NUM_TRIALS_DEFAULT)
        self.arg_mappings = get_dict_param(autotuning_dict, AUTOTUNING_ARG_MAPPINGS, AUTOTUNING_ARG_MAPPINGS_DEFAULT)
        self.model_info = get_model_info_config(autotuning_dict)
        self.model_info_path = get_scalar_param(autotuning_dict, AUTOTUNING_MODEL_INFO_PATH, AUTOTUNING_MODEL_INFO_PATH_DEFAULT)
        self.mp_size = get_scalar_param(autotuning_dict, AUTOTUNING_MP_SIZE, AUTOTUNING_MP_SIZE_DEFAULT)
        self.max_train_batch_size = get_dict_param(autotuning_dict, AUTOTUNING_MAX_TRAIN_BATCH_SIZE, AUTOTUNING_MAX_TRAIN_BATCH_SIZE_DEFAULT)
        self.min_train_batch_size = get_dict_param(autotuning_dict, AUTOTUNING_MIN_TRAIN_BATCH_SIZE, AUTOTUNING_MIN_TRAIN_BATCH_SIZE_DEFAULT)
        self.max_train_micro_batch_size_per_gpu = get_dict_param(autotuning_dict, AUTOTUNING_MAX_TRAIN_MICRO_BATCH_SIZE_PER_GPU, AUTOTUNING_MAX_TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT)
        self.min_train_micro_batch_size_per_gpu = get_dict_param(autotuning_dict, AUTOTUNING_MIN_TRAIN_MICRO_BATCH_SIZE_PER_GPU, AUTOTUNING_MIN_TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT)
        self.num_tuning_micro_batch_sizes = get_dict_param(autotuning_dict, AUTOTUNING_NUM_TUNING_MICRO_BATCH_SIZES, AUTOTUNING_NUM_TUNING_MICRO_BATCH_SIZES_DEFAULT)


COMMS_LOGGER_DEBUG_DEFAULT = False


COMMS_LOGGER_ENABLED_DEFAULT = False


COMMS_LOGGER_PROF_ALL_DEFAULT = True


COMMS_LOGGER_PROF_OPS_DEFAULT = []


COMMS_LOGGER_VERBOSE_DEFAULT = False


class DeepSpeedCommsConfig:

    def __init__(self, ds_config):
        self.comms_logger_enabled = 'comms_logger' in ds_config
        if self.comms_logger_enabled:
            self.comms_logger = CommsLoggerConfig(**ds_config['comms_logger'])


FLOPS_PROFILER = 'flops_profiler'


FLOPS_PROFILER_DETAILED = 'detailed'


FLOPS_PROFILER_DETAILED_DEFAULT = True


FLOPS_PROFILER_ENABLED = 'enabled'


FLOPS_PROFILER_ENABLED_DEFAULT = False


FLOPS_PROFILER_MODULE_DEPTH = 'module_depth'


FLOPS_PROFILER_MODULE_DEPTH_DEFAULT = -1


FLOPS_PROFILER_OUTPUT_FILE = 'output_file'


FLOPS_PROFILER_OUTPUT_FILE_DEFAULT = None


FLOPS_PROFILER_PROFILE_STEP = 'profile_step'


FLOPS_PROFILER_PROFILE_STEP_DEFAULT = 1


FLOPS_PROFILER_TOP_MODULES = 'top_modules'


FLOPS_PROFILER_TOP_MODULES_DEFAULT = 1


class DeepSpeedFlopsProfilerConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        super(DeepSpeedFlopsProfilerConfig, self).__init__()
        self.enabled = None
        self.profile_step = None
        self.module_depth = None
        self.top_modules = None
        if FLOPS_PROFILER in param_dict.keys():
            flops_profiler_dict = param_dict[FLOPS_PROFILER]
        else:
            flops_profiler_dict = {}
        self._initialize(flops_profiler_dict)

    def _initialize(self, flops_profiler_dict):
        self.enabled = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_ENABLED, FLOPS_PROFILER_ENABLED_DEFAULT)
        self.profile_step = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_PROFILE_STEP, FLOPS_PROFILER_PROFILE_STEP_DEFAULT)
        self.module_depth = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_MODULE_DEPTH, FLOPS_PROFILER_MODULE_DEPTH_DEFAULT)
        self.top_modules = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_TOP_MODULES, FLOPS_PROFILER_TOP_MODULES_DEFAULT)
        self.detailed = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_DETAILED, FLOPS_PROFILER_DETAILED_DEFAULT)
        self.output_file = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_OUTPUT_FILE, FLOPS_PROFILER_OUTPUT_FILE_DEFAULT)


CSV_MONITOR_ENABLED_DEFAULT = False


CSV_MONITOR_JOB_NAME_DEFAULT = 'DeepSpeedJobName'


CSV_MONITOR_OUTPUT_PATH_DEFAULT = ''


TENSORBOARD_ENABLED_DEFAULT = False


TENSORBOARD_JOB_NAME_DEFAULT = 'DeepSpeedJobName'


TENSORBOARD_OUTPUT_PATH_DEFAULT = ''


WANDB_ENABLED_DEFAULT = False


WANDB_GROUP_NAME_DEFAULT = None


WANDB_PROJECT_NAME_DEFAULT = 'deepspeed'


WANDB_TEAM_NAME_DEFAULT = None


class DeepSpeedMonitorConfig:

    def __init__(self, ds_config):
        self.tensorboard_enabled = 'tensorboard' in ds_config
        self.wandb_enabled = 'wandb' in ds_config
        self.csv_monitor_enabled = 'csv_monitor' in ds_config
        if self.tensorboard_enabled:
            self.tensorboard_config = TensorBoardConfig(**ds_config['tensorboard'])
        if self.wandb_enabled:
            self.wandb_config = WandbConfig(**ds_config['wandb'])
        if self.csv_monitor_enabled:
            self.csv_monitor_config = CSVConfig(**ds_config['csv_monitor'])


NEBULA = 'nebula'


NEBULA_ENABLED = 'enabled'


NEBULA_ENABLED_DEFAULT = False


NEBULA_ENABLE_NEBULA_LOAD = 'enable_nebula_load'


NEBULA_ENABLE_NEBULA_LOAD_DEFAULT = True


NEBULA_LOAD_PATH = 'nebula_load_path'


NEBULA_LOAD_PATH_DEFAULT = None


NEBULA_NUM_OF_VERSION_IN_RETENTION = 'num_of_version_in_retention'


NEBULA_NUM_OF_VERSION_IN_RETENTION_DEFAULT = 2


NEBULA_PERSISTENT_STORAGE_PATH = 'persistent_storage_path'


NEBULA_PERSISTENT_STORAGE_PATH_DEFAULT = None


NEBULA_PERSISTENT_TIME_INTERVAL = 'persistent_time_interval'


NEBULA_PERSISTENT_TIME_INTERVAL_DEFAULT = 100


class DeepSpeedNebulaConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        super(DeepSpeedNebulaConfig, self).__init__()
        self.enabled = None
        self.persistent_storage_path = None
        self.persistent_time_interval = None
        self.num_of_version_in_retention = None
        self.enable_nebula_load = None
        if NEBULA in param_dict.keys():
            nebula_dict = param_dict[NEBULA]
        else:
            nebula_dict = {}
        self._initialize(nebula_dict)

    def _initialize(self, nebula_dict):
        self.enabled = get_scalar_param(nebula_dict, NEBULA_ENABLED, NEBULA_ENABLED_DEFAULT)
        self.load_path = get_scalar_param(nebula_dict, NEBULA_LOAD_PATH, NEBULA_LOAD_PATH_DEFAULT)
        self.enable_nebula_load = get_scalar_param(nebula_dict, NEBULA_ENABLE_NEBULA_LOAD, NEBULA_ENABLE_NEBULA_LOAD_DEFAULT)
        self.persistent_storage_path = get_scalar_param(nebula_dict, NEBULA_PERSISTENT_STORAGE_PATH, NEBULA_PERSISTENT_STORAGE_PATH_DEFAULT)
        self.persistent_time_interval = get_scalar_param(nebula_dict, NEBULA_PERSISTENT_TIME_INTERVAL, NEBULA_PERSISTENT_TIME_INTERVAL_DEFAULT)
        self.num_of_version_in_retention = get_scalar_param(nebula_dict, NEBULA_NUM_OF_VERSION_IN_RETENTION, NEBULA_NUM_OF_VERSION_IN_RETENTION_DEFAULT)


ELASTICITY = 'elasticity'


class ElasticityError(Exception):
    """
    Base exception for all elasticity related errors
    """


class ElasticityConfigError(ElasticityError):
    """
    Elasticity configuration error
    """


GRADIENT_ACCUMULATION_STEPS = 'gradient_accumulation_steps'


GRAD_ACCUM_DTYPE = 'grad_accum_dtype'


GRAD_ACCUM_DTYPE_DEFAULT = None


IGNORE_NON_ELASTIC_BATCH_INFO = 'ignore_non_elastic_batch_info'


IGNORE_NON_ELASTIC_BATCH_INFO_DEFAULT = False


LOAD_UNIVERSAL_CHECKPOINT = 'load_universal'


LOAD_UNIVERSAL_CHECKPOINT_DEFAULT = False


MAX_GRAD_NORM = 'max_grad_norm'


MODEL_PARLLEL_SIZE = 'model_parallel_size'


MODEL_PARLLEL_SIZE_DEFAULT = 1


NUM_GPUS_PER_NODE = 'num_gpus_per_node'


NUM_GPUS_PER_NODE_DEFAULT = 1


TENSOR_CORE_ALIGN_SIZE = 8


TRAIN_BATCH_SIZE = 'train_batch_size'


TRAIN_MICRO_BATCH_SIZE_PER_GPU = """
TRAIN_MICRO_BATCH_SIZE_PER_GPU is defined in this format:
"train_micro_batch_size_per_gpu": 1
"""


USE_NODE_LOCAL_STORAGE_CHECKPOINT = 'use_node_local_storage'


USE_NODE_LOCAL_STORAGE_CHECKPOINT_DEFAULT = False


VOCABULARY_SIZE = 'vocabulary_size'


VOCABULARY_SIZE_DEFAULT = None


class ValidationMode:
    WARN = 'WARN'
    IGNORE = 'IGNORE'
    FAIL = 'FAIL'


class ZeroStageEnum(int, Enum):
    """ Enum class for possible zero stages """
    disabled = 0
    optimizer_states = 1
    gradients = 2
    weights = 3
    max_stage = 3


ENABLED = 'enabled'


ENABLED_DEFAULT = False


MAX_ACCEPTABLE_BATCH_SIZE = 'max_train_batch_size'


MAX_ACCEPTABLE_BATCH_SIZE_DEFAULT = 2000


MAX_GPUS = 'max_gpus'


MAX_GPUS_DEFAULT = 10000


MICRO_BATCHES = 'micro_batch_sizes'


MICRO_BATCHES_DEFAULT = [2, 4, 6]


MIN_GPUS = 'min_gpus'


MIN_GPUS_DEFAULT = 1


MIN_TIME = 'min_time'


MIN_TIME_DEFAULT = 0


PREFER_LARGER_BATCH = 'prefer_larger_batch'


PREFER_LARGER_BATCH_DEFAULT = True


VERSION = 'version'


LATEST_ELASTICITY_VERSION = 0.2


VERSION_DEFAULT = LATEST_ELASTICITY_VERSION


class ElasticityConfig:
    """
    Elastic config object, constructed from a param dictionary that only contains elastic
    config parameters, example below:

    If elasticity is enabled, user must specify (at least) max_train_batch_size
    and micro_batch_sizes.

    {
        "enabled": true,
        "max_train_batch_size": 2000,
        "micro_batch_sizes": [2,4,6],
        "min_gpus": 1,
        "max_gpus" : 10000
        "min_time": 20
        "ignore_non_elastic_batch_info": false
        "version": 0.1
    }
    """

    def __init__(self, param_dict):
        self.enabled = param_dict.get(ENABLED, ENABLED_DEFAULT)
        if self.enabled:
            if MAX_ACCEPTABLE_BATCH_SIZE in param_dict:
                self.max_acceptable_batch_size = param_dict[MAX_ACCEPTABLE_BATCH_SIZE]
            else:
                raise ElasticityConfigError(f'Elasticity config missing {MAX_ACCEPTABLE_BATCH_SIZE}')
            if MICRO_BATCHES in param_dict:
                self.micro_batches = param_dict[MICRO_BATCHES]
            else:
                raise ElasticityConfigError(f'Elasticity config missing {MICRO_BATCHES}')
        else:
            self.max_acceptable_batch_size = param_dict.get(MAX_ACCEPTABLE_BATCH_SIZE, MAX_ACCEPTABLE_BATCH_SIZE_DEFAULT)
            self.micro_batches = param_dict.get(MICRO_BATCHES, MICRO_BATCHES_DEFAULT)
        if not isinstance(self.micro_batches, list):
            raise ElasticityConfigError(f'Elasticity expected value of {MICRO_BATCHES} to be a list of micro batches, instead is: {type(self.micro_batches)}, containing: {self.micro_batches}')
        if not all(map(lambda m: isinstance(m, int), self.micro_batches)):
            raise ElasticityConfigError(f'Elasticity expected {MICRO_BATCHES} to only contain a list of integers, instead contains: f{self.micro_batches}')
        if not all(map(lambda m: m > 0, self.micro_batches)):
            raise ElasticityConfigError(f'Elasticity expected {MICRO_BATCHES} to only contain positive integers, instead contains: f{self.micro_batches}')
        self.min_gpus = param_dict.get(MIN_GPUS, MIN_GPUS_DEFAULT)
        self.max_gpus = param_dict.get(MAX_GPUS, MAX_GPUS_DEFAULT)
        if self.min_gpus < 1 or self.max_gpus < 1:
            raise ElasticityConfigError(f'Elasticity min/max gpus must be > 0, given min_gpus: {self.min_gpus}, max_gpus: {self.max_gpus}')
        if self.max_gpus < self.min_gpus:
            raise ElasticityConfigError(f'Elasticity min_gpus cannot be greater than max_gpus, given min_gpus: {self.min_gpus}, max_gpus: {self.max_gpus}')
        self.model_parallel_size = param_dict.get(MODEL_PARLLEL_SIZE, MODEL_PARLLEL_SIZE_DEFAULT)
        if self.model_parallel_size < 1:
            raise ElasticityConfigError(f'Model-Parallel size cannot be less than 1, given model-parallel size: {self.model_parallel_size}')
        self.num_gpus_per_node = param_dict.get(NUM_GPUS_PER_NODE, NUM_GPUS_PER_NODE_DEFAULT)
        if self.num_gpus_per_node < 1:
            raise ElasticityConfigError(f'Number of GPUs per node cannot be less than 1, given number of GPUs per node: {self.num_gpus_per_node}')
        self.min_time = param_dict.get(MIN_TIME, MIN_TIME_DEFAULT)
        if self.min_time < 0:
            raise ElasticityConfigError(f'Elasticity min time needs to be >= 0: given {self.min_time}')
        self.version = param_dict.get(VERSION, VERSION_DEFAULT)
        self.prefer_larger_batch_size = param_dict.get(PREFER_LARGER_BATCH, PREFER_LARGER_BATCH_DEFAULT)
        self.ignore_non_elastic_batch_info = param_dict.get(IGNORE_NON_ELASTIC_BATCH_INFO, IGNORE_NON_ELASTIC_BATCH_INFO_DEFAULT)

    def repr(self):
        return self.__dict__

    def __repr__(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4)


class ElasticityIncompatibleWorldSize(ElasticityError):
    """
    Attempting to run a world size that is incompatible with a given elastic config
    """


MINIMUM_DEEPSPEED_VERSION = '0.3.8'


def _compatible_ds_version_check(target_deepspeed_version: str):
    min_version = pkg_version.parse(MINIMUM_DEEPSPEED_VERSION)
    target_version = pkg_version.parse(target_deepspeed_version)
    err_str = f'Target deepspeed version of {target_deepspeed_version} is not compatible with minimum version {MINIMUM_DEEPSPEED_VERSION} supporting elasticity.'
    if target_version < min_version:
        raise ElasticityError(err_str)
    return True


def get_valid_gpus(batch_size, micro_batches, min_valid_gpus, max_valid_gpus):
    valid_gpus = []
    for micro_batch in micro_batches:
        if batch_size % micro_batch == 0:
            max_gpus = batch_size // micro_batch
            if max_gpus >= min_valid_gpus and max_gpus <= max_valid_gpus:
                valid_gpus.append(max_gpus)
            for i in range(1, max_gpus // 2 + 1):
                if i > max_valid_gpus:
                    break
                if i < min_valid_gpus:
                    continue
                if max_gpus % i == 0:
                    valid_gpus.append(i)
    valid_gpus = set(valid_gpus)
    valid_gpus = sorted(list(valid_gpus))
    return valid_gpus


def get_best_candidates(candidate_batch_sizes, micro_batches, min_gpus, max_gpus, prefer_larger):
    max_valid_gpus = 0
    valid_gpus = None
    final_batch_size = int(min(micro_batches))
    for batch_size in candidate_batch_sizes:
        current_valid_gpus = get_valid_gpus(batch_size, micro_batches, min_gpus, max_gpus)
        if len(current_valid_gpus) > max_valid_gpus or len(current_valid_gpus) == max_valid_gpus and (prefer_larger and batch_size > final_batch_size or not prefer_larger and batch_size < final_batch_size):
            max_valid_gpus = len(current_valid_gpus)
            valid_gpus = current_valid_gpus
            final_batch_size = batch_size
    return final_batch_size, valid_gpus


HCN_LIST = [1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280, 720720]


def get_candidate_batch_sizes(base_list, max_acceptable_batch_size):
    candidate_batch_size = []
    for base in base_list:
        if base >= max_acceptable_batch_size:
            candidate_batch_size.append(base)
        else:
            value = max_acceptable_batch_size // base
            index = np.argmax(np.asarray(HCN_LIST) > value)
            candidate_batch_size.append(HCN_LIST[index - 1] * base)
    candidate_batch_size = list(set(candidate_batch_size))
    logger.info(f'Candidate batch size: {candidate_batch_size}')
    return candidate_batch_size


def _get_compatible_gpus_v01(micro_batches, max_acceptable_batch_size, min_gpus=None, max_gpus=None, prefer_larger=True):
    """We use two heuristics to compute the batch size
        1. We use the Lowest Common Multiple of the micro-batches
    as the base batch size and scale it by a HCN such that the result is
    the largest batch size less than the max_acceptable batch size
        2. We use each of the micro batches as a base and scale it
    by a HCN such that the result is the largest batch size less than the
    max_acceptable batch size.

    We then use brute force to count the number of compatible GPU count for
    each of the aforementioned cases, and return the batch size with the most number of
    compatible GPU counts in the min-max GPU range if provided, other wise
    we return the batch size with the most number of total compatible GPU counts.

    Returns:
        final_batch_size
        valid_gpus
    """
    min_gpus = min_gpus or 1
    max_gpus = max_gpus or max_acceptable_batch_size // min(micro_batches)
    if not all(mb <= max_acceptable_batch_size for mb in micro_batches):
        raise ValueError(f'All micro batches must be less than             or equal to max_acceptable_batch_size: {max_acceptable_batch_size}')
    lcm = np.lcm.reduce(micro_batches)
    base_list = []
    base_list.extend(micro_batches)
    base_list.append(lcm)
    candidate_batch_sizes = get_candidate_batch_sizes(base_list, max_acceptable_batch_size)
    final_batch_size, valid_gpus = get_best_candidates(candidate_batch_sizes, micro_batches, min_gpus, max_gpus, prefer_larger)
    return final_batch_size, valid_gpus


def _get_compatible_gpus_v02(micro_batches, max_acceptable_batch_size, current_num_gpus, min_gpus=None, max_gpus=None, prefer_larger=True, num_gpus_per_node=1, model_parallel_size=1):
    """
    Returns:
        final_batch_size
        valid_gpus
        micro-batch size
    """
    if num_gpus_per_node % model_parallel_size != 0:
        raise ElasticityError(f'In Elasticity v0.2, number of GPUs per node:{num_gpus_per_node} should be divisible by model parallel size {model_parallel_size}')

    def get_microbatch(final_batch_size):
        candidate_microbatch = None
        for micro_batch in micro_batches:
            if final_batch_size // current_num_gpus % micro_batch == 0:
                if candidate_microbatch == None:
                    candidate_microbatch = micro_batch
                if prefer_larger and candidate_microbatch < micro_batch:
                    candidate_microbatch = micro_batch
        return candidate_microbatch
    dp_size_per_node = num_gpus_per_node // model_parallel_size
    final_batch_size, valid_world_size = _get_compatible_gpus_v01(micro_batches, int(max_acceptable_batch_size / dp_size_per_node), int(min_gpus / num_gpus_per_node), int(max_gpus / num_gpus_per_node), prefer_larger=prefer_larger)
    final_batch_size = int(final_batch_size) * dp_size_per_node
    valid_dp_world_size = [(i * dp_size_per_node) for i in valid_world_size]
    if current_num_gpus // model_parallel_size in valid_dp_world_size:
        candidate_microbatch = get_microbatch(final_batch_size)
        return final_batch_size, valid_dp_world_size, candidate_microbatch
    current_dp_size = current_num_gpus / num_gpus_per_node * dp_size_per_node
    candidate_batch_sizes = []
    for micro_batch in micro_batches:
        min_batch_size = micro_batch * current_dp_size
        factor = math.floor(max_acceptable_batch_size / float(min_batch_size))
        candidate_batch_sizes.append(factor * min_batch_size)
    used_microbatch = None
    if prefer_larger:
        candidate_batch_size = max(candidate_batch_sizes)
    else:
        candidate_batch_size = min(candidate_batch_sizes)
    candidate_microbatch = get_microbatch(candidate_batch_size)
    return candidate_batch_size, [int(current_dp_size)], candidate_microbatch


def compute_elastic_config(ds_config: dict, target_deepspeed_version: str, world_size=0, return_microbatch=False):
    """Core deepspeed elasticity API. Given an elastic config (similar to the example below)
    DeepSpeed will compute a total train batch size corresponding valid GPU count list that
    provides a high level of elasticity. Elasticity in this case means we are safe to scale
    the training job up/down across the GPU count list *without* any negative impacts on
    training convergence. This is achievable primarily due to DeepSpeed's gradient accumulation
    feature which allows us to decompose a global training batch size into:
    micro-batch-size * gradient-accumulation-steps * world-size.

    "elasticity": {
        "enabled": true,
        "max_train_batch_size": 2000,
        "micro_batch_sizes": [2,4,6],
        "min_gpus": 1,
        "max_gpus" : 10000
        "min_time": 20
        "version": 0.1
    }

    Intended to be called both by scheduling infrastructure and deepspeed runtime.
    For the same `ds_config` we should return deterministic results.

    Args:
        ds_config (dict): DeepSpeed config dictionary/json
        target_deepspeed_version (str): When called from scheduling
            infrastructure we want to ensure that the target deepspeed version is
            compatible with the elasticity version used in the backend.
        world_size (int, optional): Intended/current DP world size, will do some sanity
            checks to ensure world size is actually valid with the config.
        return_microbatch (bool, optional): whether to return micro batch size or not.

    Raises:
        ElasticityConfigError: Missing required elasticity config or elasticity disabled
        ElasticityError: If target deepspeed version is not compatible with current version

    Returns:
        final_batch_size (int): total batch size used for training
        valid_gpus (list(int)): list of valid GPU counts with this config
        micro_batch_size (int, optional): if world_size is provided will return
            specific micro batch size
    """
    if not isinstance(ds_config, dict):
        raise ValueError(f'Expected ds_config to be a dictionary but received a {type(ds_config)}, containing: {ds_config}')
    if ELASTICITY not in ds_config:
        raise ElasticityConfigError(f"'{ELASTICITY}' is missing from config json, please add it if running an elastic training job.")
    elastic_config_dict = ds_config[ELASTICITY]
    if not elastic_config_dict.get(ENABLED, ENABLED_DEFAULT):
        raise ElasticityConfigError("Elasticity is disabled, please enable it ('enabled':true) if running an elastic training job.")
    elastic_config = ElasticityConfig(elastic_config_dict)
    model_parallel_size = elastic_config.model_parallel_size
    num_gpus_per_node = elastic_config.num_gpus_per_node
    if model_parallel_size > 1 and float(elastic_config.version) != 0.2:
        raise ElasticityConfigError(f'Elasticity V{elastic_config.version} does not support model-parallel training. Given model-parallel size: {model_parallel_size}')
    if float(elastic_config.version) > LATEST_ELASTICITY_VERSION:
        raise ElasticityConfigError(f'Attempting to run elasticity version {elastic_config.version} but runtime only supports up to {LATEST_ELASTICITY_VERSION}')
    if not _compatible_ds_version_check(target_deepspeed_version):
        raise ElasticityError(f'Unable to run elasticity on target deepspeed version of {target_deepspeed_version}, currently {__version__}')
    if float(elastic_config.version) == 0.1:
        final_batch_size, valid_gpus = _get_compatible_gpus_v01(micro_batches=elastic_config.micro_batches, max_acceptable_batch_size=elastic_config.max_acceptable_batch_size, min_gpus=elastic_config.min_gpus, max_gpus=elastic_config.max_gpus, prefer_larger=elastic_config.prefer_larger_batch_size)
        final_batch_size = int(final_batch_size)
    elif float(elastic_config.version) == 0.2:
        if world_size != 0:
            current_num_gpus = world_size
        elif 'WORLD_SIZE' in os.environ and os.getenv('WORLD_SIZE').isnumeric():
            current_num_gpus = int(os.getenv('WORLD_SIZE'))
        else:
            WORLD_SIZE = os.getenv('WORLD_SIZE')
            raise ElasticityConfigError(f'Elasticity V 0.2 needs WORLD_SIZE to compute valid batch size. Either give it as argument to function compute_elastic_config or set it as an environment variable. Value of WORLD_SIZE as environment variable is {WORLD_SIZE}')
        final_batch_size, valid_gpus, candidate_microbatch_size = _get_compatible_gpus_v02(micro_batches=elastic_config.micro_batches, max_acceptable_batch_size=elastic_config.max_acceptable_batch_size, current_num_gpus=current_num_gpus, min_gpus=elastic_config.min_gpus, max_gpus=elastic_config.max_gpus, prefer_larger=elastic_config.prefer_larger_batch_size, num_gpus_per_node=num_gpus_per_node, model_parallel_size=model_parallel_size)
        final_batch_size = int(final_batch_size)
    else:
        raise NotImplementedError(f'Unable to find elastic logic for version: {elastic_config.version}')
    logger.info(f'Valid World Size (GPUs / Model Parallel Size): {valid_gpus}')
    if world_size > 0:
        if world_size not in valid_gpus:
            raise ElasticityIncompatibleWorldSize(f'World size ({world_size}) is not valid with the current list of valid GPU counts: {valid_gpus}')
        micro_batch_size = None
        for mbsz in sorted(list(set(elastic_config.micro_batches)), reverse=True):
            if final_batch_size // world_size % mbsz == 0:
                micro_batch_size = mbsz
                break
        assert micro_batch_size is not None, f'Unable to find divisible micro batch size world_size={world_size}, final_batch_size={final_batch_size}, and  micro_batches={elastic_config.micro_batches}.'
        return final_batch_size, valid_gpus, micro_batch_size
    if return_microbatch:
        if float(elastic_config.version) == 0.2:
            return final_batch_size, valid_gpus, candidate_microbatch_size
        else:
            micro_batch_size = None
            for mbsz in sorted(list(set(elastic_config.micro_batches)), reverse=True):
                if final_batch_size // world_size % mbsz == 0:
                    micro_batch_size = mbsz
                    break
            assert micro_batch_size is not None, f'Unable to find divisible micro batch size world_size={world_size}, final_batch_size={final_batch_size}, and  micro_batches={elastic_config.micro_batches}.'
            return final_batch_size, valid_gpus, micro_batch_size
    return final_batch_size, valid_gpus


def dict_raise_error_on_duplicate_keys(ordered_pairs):
    """Reject duplicate keys."""
    d = dict((k, v) for k, v in ordered_pairs)
    if len(d) != len(ordered_pairs):
        counter = collections.Counter([pair[0] for pair in ordered_pairs])
        keys = [key for key, value in counter.items() if value > 1]
        raise ValueError('Duplicate keys in DeepSpeed config: {}'.format(keys))
    return d


def elasticity_enabled(ds_config: dict):
    if ELASTICITY not in ds_config:
        return False
    return ds_config[ELASTICITY].get(ENABLED, ENABLED_DEFAULT)


DEEPSPEED_ELASTICITY_CONFIG = 'DEEPSPEED_ELASTICITY_CONFIG'


def ensure_immutable_elastic_config(runtime_elastic_config_dict: dict):
    """
    Ensure the resource scheduler saw the same elastic config we are using at runtime
    """
    if DEEPSPEED_ELASTICITY_CONFIG in os.environ:
        scheduler_elastic_config_dict = json.loads(os.environ[DEEPSPEED_ELASTICITY_CONFIG])
        scheduler_elastic_config = ElasticityConfig(scheduler_elastic_config_dict)
        runtime_elastic_config = ElasticityConfig(runtime_elastic_config_dict)
        err_str = "Elastic config '{}={}' seen by resource scheduler does not match config passed to runtime {}={}"
        if runtime_elastic_config.max_acceptable_batch_size != scheduler_elastic_config.max_acceptable_batch_size:
            raise ElasticityConfigError(err_str.format('max_acceptable_batch_size', scheduler_elastic_config.max_acceptable_batch_size, 'max_acceptable_batch_size', runtime_elastic_config.max_acceptable_batch_size))
        if runtime_elastic_config.micro_batches != scheduler_elastic_config.micro_batches:
            raise ElasticityConfigError(err_str.format('micro_batches', scheduler_elastic_config.micro_batches, 'micro_batches', runtime_elastic_config.micro_batches))
        if runtime_elastic_config.version != scheduler_elastic_config.version:
            raise ElasticityConfigError(err_str.format('version', scheduler_elastic_config.version, 'version', runtime_elastic_config.version))
    else:
        logger.warning('Unable to find DEEPSPEED_ELASTICITY_CONFIG environment variable, cannot guarantee resource scheduler will scale this job using compatible GPU counts.')


AIO = 'aio'


AIO_BLOCK_SIZE = 'block_size'


AIO_BLOCK_SIZE_DEFAULT = 1048576


AIO_OVERLAP_EVENTS = 'overlap_events'


AIO_OVERLAP_EVENTS_DEFAULT = True


AIO_QUEUE_DEPTH = 'queue_depth'


AIO_QUEUE_DEPTH_DEFAULT = 8


AIO_SINGLE_SUBMIT = 'single_submit'


AIO_SINGLE_SUBMIT_DEFAULT = False


AIO_THREAD_COUNT = 'thread_count'


AIO_THREAD_COUNT_DEFAULT = 1


AIO_DEFAULT_DICT = {AIO_BLOCK_SIZE: AIO_BLOCK_SIZE_DEFAULT, AIO_QUEUE_DEPTH: AIO_QUEUE_DEPTH_DEFAULT, AIO_THREAD_COUNT: AIO_THREAD_COUNT_DEFAULT, AIO_SINGLE_SUBMIT: AIO_SINGLE_SUBMIT_DEFAULT, AIO_OVERLAP_EVENTS: AIO_OVERLAP_EVENTS_DEFAULT}


def get_aio_config(param_dict):
    if AIO in param_dict.keys() and param_dict[AIO] is not None:
        aio_dict = param_dict[AIO]
        return {AIO_BLOCK_SIZE: get_scalar_param(aio_dict, AIO_BLOCK_SIZE, AIO_BLOCK_SIZE_DEFAULT), AIO_QUEUE_DEPTH: get_scalar_param(aio_dict, AIO_QUEUE_DEPTH, AIO_QUEUE_DEPTH_DEFAULT), AIO_THREAD_COUNT: get_scalar_param(aio_dict, AIO_THREAD_COUNT, AIO_THREAD_COUNT_DEFAULT), AIO_SINGLE_SUBMIT: get_scalar_param(aio_dict, AIO_SINGLE_SUBMIT, AIO_SINGLE_SUBMIT_DEFAULT), AIO_OVERLAP_EVENTS: get_scalar_param(aio_dict, AIO_OVERLAP_EVENTS, AIO_OVERLAP_EVENTS_DEFAULT)}
    return AIO_DEFAULT_DICT


AMP = 'amp'


AMP_ENABLED = 'enabled'


AMP_ENABLED_DEFAULT = False


def get_amp_enabled(param_dict):
    if AMP in param_dict.keys():
        return get_scalar_param(param_dict[AMP], AMP_ENABLED, AMP_ENABLED_DEFAULT)
    else:
        return False


def get_amp_params(param_dict):
    if AMP in param_dict.keys():
        amp_params = copy.copy(param_dict[AMP])
        amp_params.pop(AMP_ENABLED)
        return amp_params
    else:
        return False


BFLOAT16 = 'bf16'


BFLOAT16_ENABLED = 'enabled'


BFLOAT16_ENABLED_DEFAULT = False


BFLOAT16_OLD = 'bfloat16'


def get_bfloat16_enabled(param_dict):
    for key in [BFLOAT16, BFLOAT16_OLD]:
        if key in param_dict.keys():
            return get_scalar_param(param_dict[key], BFLOAT16_ENABLED, BFLOAT16_ENABLED_DEFAULT)
    return False


CHECKPOINT_PARALLEL_WRITE = 'parallel_write'


CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE = 'pipeline_stage'


CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE_DEFAULT = False


class DeepSpeedConfigError(Exception):
    pass


def get_checkpoint_parallel_write_pipeline(checkpoint_params):
    par_write_params = checkpoint_params.get(CHECKPOINT_PARALLEL_WRITE, {})
    par_write_pipeline = par_write_params.get(CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE, CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE_DEFAULT)
    if par_write_pipeline in [True, False]:
        return par_write_pipeline
    else:
        raise DeepSpeedConfigError(f"checkpoint::parallel_write::pipeline_stage value of '{par_write_pipeline}' is invalid, expecting: true or false")


CHECKPOINT = 'checkpoint'


def get_checkpoint_params(param_dict):
    return param_dict.get(CHECKPOINT, {})


CHECKPOINT_TAG_VALIDATION = 'tag_validation'


CHECKPOINT_TAG_VALIDATION_DEFAULT = ValidationMode.WARN


CHECKPOINT_TAG_VALIDATION_MODES = [ValidationMode.WARN, ValidationMode.IGNORE, ValidationMode.FAIL]


def get_checkpoint_tag_validation_mode(checkpoint_params):
    tag_validation_mode = checkpoint_params.get(CHECKPOINT_TAG_VALIDATION, CHECKPOINT_TAG_VALIDATION_DEFAULT)
    tag_validation_mode = tag_validation_mode.upper()
    if tag_validation_mode in CHECKPOINT_TAG_VALIDATION_MODES:
        return tag_validation_mode
    else:
        raise DeepSpeedConfigError(f'Checkpoint config contains invalid tag_validation value of {tag_validation_mode}, expecting one of {CHECKPOINT_TAG_VALIDATION_MODES}')


COMMUNICATION_DATA_TYPE = 'communication_data_type'


COMMUNICATION_DATA_TYPE_DEFAULT = None


def get_communication_data_type(param_dict):
    val = get_scalar_param(param_dict, COMMUNICATION_DATA_TYPE, COMMUNICATION_DATA_TYPE_DEFAULT)
    val = val.lower() if val is not None else val
    if val is None:
        return val
    elif val == 'fp32':
        return torch.float32
    elif val == 'fp16':
        return torch.float16
    elif val == 'bfp16':
        return torch.bfloat16
    raise ValueError(f"Invalid communication_data_type. Supported data types: ['fp16', 'bfp16', 'fp32']. Got: {val}")


ACTIVATION_QUANTIZATION = 'activation_quantization'


CHANNEL_PRUNING = 'channel_pruning'


COMPRESSION_TRAINING = 'compression_training'


HEAD_PRUNING = 'head_pruning'


LAYER_REDUCTION = 'layer_reduction'


ROW_PRUNING = 'row_pruning'


SPARSE_PRUNING = 'sparse_pruning'


WEIGHT_QUANTIZATION = 'weight_quantization'


TECHNIQUE_ENABLED = 'enabled'


ACTIVATION_QUANTIZATION_ENABLED = TECHNIQUE_ENABLED


DIFFERENT_GROUPS = 'different_groups'


SHARED_PARAMETERS = 'shared_parameters'


ACTIVATION_QUANTIZE_BITS = 'bits'


DIFFERENT_GROUPS_MODULE_SCOPE = 'modules'


DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT = '*'


DIFFERENT_GROUPS_PARAMETERS = 'params'


DIFFERENT_GROUPS_RELATED_MODULE_SCOPE = 'related_modules'


DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT = None


def get_activation_quantization_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert ACTIVATION_QUANTIZE_BITS in group_dict.keys(), f'{ACTIVATION_QUANTIZE_BITS} must be specified for activation quantization group {name}'
        return group_dict
    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE, DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)
    return output


ACTIVATION_QUANTIZATION_ENABLED_DEFAULT = False


ACTIVATION_QUANTIZE_ASYMMETRIC = 'asymmetric'


ACTIVATION_QUANTIZE_RANGE = 'range_calibration'


ACTIVATION_QUANTIZE_RANGE_DEFAULT = 'dynamic'


ACTIVATION_QUANTIZE_RANGE_DYNAMIC = 'dynamic'


ACTIVATION_QUANTIZE_RANGE_STATIC = 'static'


TECHNIQUE_SCHEDULE_OFFSET = 'schedule_offset'


ACTIVATION_QUANTIZE_SCHEDULE_OFFSET = TECHNIQUE_SCHEDULE_OFFSET


ACTIVATION_QUANTIZE_SCHEDULE_OFFSET_DEFAULT = 1000


ACTIVATION_QUANTIZE_SYMMETRIC = 'symmetric'


ACTIVATION_QUANTIZE_TYPE = 'quantization_type'


ACTIVATION_QUANTIZE_TYPE_DEFAULT = 'symmetric'


def get_activation_quantization_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[ACTIVATION_QUANTIZATION_ENABLED] = get_scalar_param(sub_param_dict, ACTIVATION_QUANTIZATION_ENABLED, ACTIVATION_QUANTIZATION_ENABLED_DEFAULT)
        output[ACTIVATION_QUANTIZE_TYPE] = get_scalar_param(sub_param_dict, ACTIVATION_QUANTIZE_TYPE, ACTIVATION_QUANTIZE_TYPE_DEFAULT)
        assert output[ACTIVATION_QUANTIZE_TYPE] in [ACTIVATION_QUANTIZE_SYMMETRIC, ACTIVATION_QUANTIZE_ASYMMETRIC], f'Invalid activation quantize type. Supported types: [{ACTIVATION_QUANTIZE_SYMMETRIC}, {ACTIVATION_QUANTIZE_ASYMMETRIC}]'
        output[ACTIVATION_QUANTIZE_RANGE] = get_scalar_param(sub_param_dict, ACTIVATION_QUANTIZE_RANGE, ACTIVATION_QUANTIZE_RANGE_DEFAULT)
        assert output[ACTIVATION_QUANTIZE_RANGE] in [ACTIVATION_QUANTIZE_RANGE_DYNAMIC, ACTIVATION_QUANTIZE_RANGE_STATIC], f'Invalid activation quantize range calibration. Supported types: [{ACTIVATION_QUANTIZE_RANGE_DYNAMIC}, {ACTIVATION_QUANTIZE_RANGE_STATIC}]'
        output[ACTIVATION_QUANTIZE_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, ACTIVATION_QUANTIZE_SCHEDULE_OFFSET, ACTIVATION_QUANTIZE_SCHEDULE_OFFSET_DEFAULT)
    else:
        output[ACTIVATION_QUANTIZATION_ENABLED] = ACTIVATION_QUANTIZATION_ENABLED_DEFAULT
        output[ACTIVATION_QUANTIZE_TYPE] = ACTIVATION_QUANTIZE_TYPE_DEFAULT
        output[ACTIVATION_QUANTIZE_RANGE] = ACTIVATION_QUANTIZE_RANGE_DEFAULT
        output[ACTIVATION_QUANTIZE_SCHEDULE_OFFSET] = ACTIVATION_QUANTIZE_SCHEDULE_OFFSET_DEFAULT
    return output


def get_activation_quantization(param_dict):
    output = {}
    if ACTIVATION_QUANTIZATION not in param_dict.keys():
        param_dict[ACTIVATION_QUANTIZATION] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[ACTIVATION_QUANTIZATION]
    output[SHARED_PARAMETERS] = get_activation_quantization_shared_parameters(sub_param_dict)
    if output[SHARED_PARAMETERS][ACTIVATION_QUANTIZATION_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(), f'Activation Quantization is enabled, {DIFFERENT_GROUPS} must be specified'
    output[DIFFERENT_GROUPS] = get_activation_quantization_different_groups(sub_param_dict)
    return output


CHANNEL_PRUNING_ENABLED = TECHNIQUE_ENABLED


CHANNEL_PRUNING_DENSE_RATIO = 'dense_ratio'


def get_channel_pruning_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert CHANNEL_PRUNING_DENSE_RATIO in group_dict.keys(), f'{CHANNEL_PRUNING_DENSE_RATIO} must be specified for channel pruning group {name}'
        return group_dict
    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE, DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)
    return output


CHANNEL_PRUNING_ENABLED_DEFAULT = False


CHANNEL_PRUNING_METHOD = 'method'


CHANNEL_PRUNING_METHOD_DEFAULT = 'l1'


CHANNEL_PRUNING_METHOD_L1 = 'l1'


CHANNEL_PRUNING_METHOD_TOPK = 'topk'


CHANNEL_PRUNING_SCHEDULE_OFFSET = TECHNIQUE_SCHEDULE_OFFSET


CHANNEL_PRUNING_SCHEDULE_OFFSET_DEFAULT = 1000


def get_channel_pruning_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[CHANNEL_PRUNING_ENABLED] = get_scalar_param(sub_param_dict, CHANNEL_PRUNING_ENABLED, CHANNEL_PRUNING_ENABLED_DEFAULT)
        output[CHANNEL_PRUNING_METHOD] = get_scalar_param(sub_param_dict, CHANNEL_PRUNING_METHOD, CHANNEL_PRUNING_METHOD_DEFAULT)
        assert output[CHANNEL_PRUNING_METHOD] in [CHANNEL_PRUNING_METHOD_L1, CHANNEL_PRUNING_METHOD_TOPK], f'Invalid channel pruning method. Supported types: [{CHANNEL_PRUNING_METHOD_L1}, {CHANNEL_PRUNING_METHOD_TOPK}]'
        output[CHANNEL_PRUNING_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, CHANNEL_PRUNING_SCHEDULE_OFFSET, CHANNEL_PRUNING_SCHEDULE_OFFSET_DEFAULT)
    else:
        output[CHANNEL_PRUNING_ENABLED] = CHANNEL_PRUNING_ENABLED_DEFAULT
        output[CHANNEL_PRUNING_METHOD] = CHANNEL_PRUNING_METHOD_DEFAULT
        output[CHANNEL_PRUNING_SCHEDULE_OFFSET] = CHANNEL_PRUNING_SCHEDULE_OFFSET_DEFAULT
    return output


def get_channel_pruning(param_dict):
    output = {}
    if CHANNEL_PRUNING not in param_dict.keys():
        param_dict[CHANNEL_PRUNING] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[CHANNEL_PRUNING]
    output[SHARED_PARAMETERS] = get_channel_pruning_shared_parameters(sub_param_dict)
    if output[SHARED_PARAMETERS][CHANNEL_PRUNING_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(), f'Sparse Pruning is enabled, {DIFFERENT_GROUPS} must be specified'
    output[DIFFERENT_GROUPS] = get_channel_pruning_different_groups(sub_param_dict)
    return output


HEAD_PRUNING_ENABLED = TECHNIQUE_ENABLED


HEAD_PRUNING_DENSE_RATIO = 'dense_ratio'


def get_head_pruning_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert HEAD_PRUNING_DENSE_RATIO in group_dict.keys(), f'dense_ratio must be specified for head pruning group {name}'
        return group_dict
    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE, DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)
    return output


HEAD_PRUNING_ENABLED_DEFAULT = False


HEAD_PRUNING_METHOD = 'method'


HEAD_PRUNING_METHOD_DEFAULT = 'topk'


HEAD_PRUNING_METHOD_L1 = 'l1'


HEAD_PRUNING_METHOD_TOPK = 'topk'


HEAD_PRUNING_NUM_HEADS = 'num_heads'


HEAD_PRUNING_SCHEDULE_OFFSET = TECHNIQUE_SCHEDULE_OFFSET


HEAD_PRUNING_SCHEDULE_OFFSET_DEFAULT = 1000


def get_head_pruning_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[HEAD_PRUNING_ENABLED] = get_scalar_param(sub_param_dict, HEAD_PRUNING_ENABLED, HEAD_PRUNING_ENABLED_DEFAULT)
        output[HEAD_PRUNING_METHOD] = get_scalar_param(sub_param_dict, HEAD_PRUNING_METHOD, HEAD_PRUNING_METHOD_DEFAULT)
        assert output[HEAD_PRUNING_METHOD] in [HEAD_PRUNING_METHOD_L1, HEAD_PRUNING_METHOD_TOPK], f'Invalid head pruning method. Supported types: [{HEAD_PRUNING_METHOD_L1}, {HEAD_PRUNING_METHOD_TOPK}]'
        output[HEAD_PRUNING_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, HEAD_PRUNING_SCHEDULE_OFFSET, HEAD_PRUNING_SCHEDULE_OFFSET_DEFAULT)
        if output[HEAD_PRUNING_ENABLED]:
            assert HEAD_PRUNING_NUM_HEADS in sub_param_dict.keys(), f'{HEAD_PRUNING_NUM_HEADS} must be specified for head pruning'
            output[HEAD_PRUNING_NUM_HEADS] = sub_param_dict[HEAD_PRUNING_NUM_HEADS]
    else:
        output[HEAD_PRUNING_ENABLED] = HEAD_PRUNING_ENABLED_DEFAULT
        output[HEAD_PRUNING_METHOD] = HEAD_PRUNING_METHOD_DEFAULT
        output[HEAD_PRUNING_SCHEDULE_OFFSET] = HEAD_PRUNING_SCHEDULE_OFFSET_DEFAULT
    return output


def get_head_pruning(param_dict):
    output = {}
    if HEAD_PRUNING not in param_dict.keys():
        param_dict[HEAD_PRUNING] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[HEAD_PRUNING]
    output[SHARED_PARAMETERS] = get_head_pruning_shared_parameters(sub_param_dict)
    if output[SHARED_PARAMETERS][HEAD_PRUNING_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(), f'Head Pruning is enabled, {DIFFERENT_GROUPS} must be specified'
    output[DIFFERENT_GROUPS] = get_head_pruning_different_groups(sub_param_dict)
    return output


LAYER_REDUCTION_ENABLED = 'enabled'


LAYER_REDUCTION_ENABLED_DEFAULT = False


def get_layer_reduction_enabled(param_dict):
    if LAYER_REDUCTION in param_dict.keys():
        return get_scalar_param(param_dict[LAYER_REDUCTION], LAYER_REDUCTION_ENABLED, LAYER_REDUCTION_ENABLED_DEFAULT)
    else:
        return False


def get_layer_reduction_params(param_dict):
    if LAYER_REDUCTION in param_dict.keys():
        layer_reduction_params = copy.copy(param_dict[LAYER_REDUCTION])
        layer_reduction_params.pop(LAYER_REDUCTION_ENABLED)
        return layer_reduction_params
    else:
        return False


def get_layer_reduction(param_dict):
    output = {}
    output[LAYER_REDUCTION_ENABLED] = LAYER_REDUCTION_ENABLED_DEFAULT
    if get_layer_reduction_enabled(param_dict):
        output[LAYER_REDUCTION_ENABLED] = get_layer_reduction_enabled(param_dict)
        for key, val in get_layer_reduction_params(param_dict).items():
            output[key] = val
    return output


ROW_PRUNING_ENABLED = TECHNIQUE_ENABLED


ROW_PRUNING_DENSE_RATIO = 'dense_ratio'


def get_row_pruning_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert ROW_PRUNING_DENSE_RATIO in group_dict.keys(), f'{ROW_PRUNING_DENSE_RATIO} must be specified for row pruning group {name}'
        return group_dict
    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE, DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)
    return output


ROW_PRUNING_ENABLED_DEFAULT = False


ROW_PRUNING_METHOD = 'method'


ROW_PRUNING_METHOD_DEFAULT = 'l1'


ROW_PRUNING_METHOD_L1 = 'l1'


ROW_PRUNING_METHOD_TOPK = 'topk'


ROW_PRUNING_SCHEDULE_OFFSET = TECHNIQUE_SCHEDULE_OFFSET


ROW_PRUNING_SCHEDULE_OFFSET_DEFAULT = 1000


def get_row_pruning_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[ROW_PRUNING_ENABLED] = get_scalar_param(sub_param_dict, ROW_PRUNING_ENABLED, ROW_PRUNING_ENABLED_DEFAULT)
        output[ROW_PRUNING_METHOD] = get_scalar_param(sub_param_dict, ROW_PRUNING_METHOD, ROW_PRUNING_METHOD_DEFAULT)
        assert output[ROW_PRUNING_METHOD] in [ROW_PRUNING_METHOD_L1, ROW_PRUNING_METHOD_TOPK], f'Invalid row pruning method. Supported types: [{ROW_PRUNING_METHOD_L1}, {ROW_PRUNING_METHOD_TOPK}]'
        output[ROW_PRUNING_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, ROW_PRUNING_SCHEDULE_OFFSET, ROW_PRUNING_SCHEDULE_OFFSET_DEFAULT)
    else:
        output[ROW_PRUNING_ENABLED] = ROW_PRUNING_ENABLED_DEFAULT
        output[ROW_PRUNING_METHOD] = ROW_PRUNING_METHOD_DEFAULT
        output[ROW_PRUNING_SCHEDULE_OFFSET] = ROW_PRUNING_SCHEDULE_OFFSET_DEFAULT
    return output


def get_row_pruning(param_dict):
    output = {}
    if ROW_PRUNING not in param_dict.keys():
        param_dict[ROW_PRUNING] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[ROW_PRUNING]
    output[SHARED_PARAMETERS] = get_row_pruning_shared_parameters(sub_param_dict)
    if output[SHARED_PARAMETERS][ROW_PRUNING_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(), f'Row Pruning is enabled, {DIFFERENT_GROUPS} must be specified'
    output[DIFFERENT_GROUPS] = get_row_pruning_different_groups(sub_param_dict)
    return output


SPARSE_PRUNING_ENABLED = TECHNIQUE_ENABLED


SPARSE_PRUNING_DENSE_RATIO = 'dense_ratio'


def get_sparse_pruning_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert SPARSE_PRUNING_DENSE_RATIO in group_dict.keys(), f'{SPARSE_PRUNING_DENSE_RATIO} must be specified for sparse pruning group {name}'
        return group_dict
    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE, DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)
    return output


SPARSE_PRUNING_ENABLED_DEFAULT = False


SPARSE_PRUNING_METHOD = 'method'


SPARSE_PRUNING_METHOD_DEFAULT = 'l1'


SPARSE_PRUNING_METHOD_L1 = 'l1'


SPARSE_PRUNING_METHOD_TOPK = 'topk'


SPARSE_PRUNING_SCHEDULE_OFFSET = TECHNIQUE_SCHEDULE_OFFSET


SPARSE_PRUNING_SCHEDULE_OFFSET_DEFAULT = 1000


def get_sparse_pruning_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[SPARSE_PRUNING_ENABLED] = get_scalar_param(sub_param_dict, SPARSE_PRUNING_ENABLED, SPARSE_PRUNING_ENABLED_DEFAULT)
        output[SPARSE_PRUNING_METHOD] = get_scalar_param(sub_param_dict, SPARSE_PRUNING_METHOD, SPARSE_PRUNING_METHOD_DEFAULT)
        assert output[SPARSE_PRUNING_METHOD] in [SPARSE_PRUNING_METHOD_L1, SPARSE_PRUNING_METHOD_TOPK], f'Invalid sparse pruning method. Supported types: [{SPARSE_PRUNING_METHOD_L1}, {SPARSE_PRUNING_METHOD_TOPK}]'
        output[SPARSE_PRUNING_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, SPARSE_PRUNING_SCHEDULE_OFFSET, SPARSE_PRUNING_SCHEDULE_OFFSET_DEFAULT)
    else:
        output[SPARSE_PRUNING_ENABLED] = SPARSE_PRUNING_ENABLED_DEFAULT
        output[SPARSE_PRUNING_METHOD] = SPARSE_PRUNING_METHOD_DEFAULT
        output[SPARSE_PRUNING_SCHEDULE_OFFSET] = SPARSE_PRUNING_SCHEDULE_OFFSET_DEFAULT
    return output


def get_sparse_pruning(param_dict):
    output = {}
    if SPARSE_PRUNING not in param_dict.keys():
        param_dict[SPARSE_PRUNING] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[SPARSE_PRUNING]
    output[SHARED_PARAMETERS] = get_sparse_pruning_shared_parameters(sub_param_dict)
    if output[SHARED_PARAMETERS][SPARSE_PRUNING_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(), f'Sparse Pruning is enabled, {DIFFERENT_GROUPS} must be specified'
    output[DIFFERENT_GROUPS] = get_sparse_pruning_different_groups(sub_param_dict)
    return output


WEIGHT_QUANTIZE_ENABLED = TECHNIQUE_ENABLED


WEIGHT_QUANTIZATION_PERIOD = 'quantization_period'


WEIGHT_QUANTIZATION_PERIOD_DEFAULT = 1


WEIGHT_QUANTIZE_START_BITS = 'start_bits'


WEIGHT_QUANTIZE_TARGET_BITS = 'target_bits'


def get_weight_quantization_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert WEIGHT_QUANTIZE_START_BITS in group_dict.keys(), f'{WEIGHT_QUANTIZE_START_BITS} must be specified for weight quantization group {name}'
        assert WEIGHT_QUANTIZE_TARGET_BITS in group_dict.keys(), f'{WEIGHT_QUANTIZE_TARGET_BITS} must be specified for weight quantization group {name}'
        group_dict[WEIGHT_QUANTIZATION_PERIOD] = get_scalar_param(group_dict, WEIGHT_QUANTIZATION_PERIOD, WEIGHT_QUANTIZATION_PERIOD_DEFAULT)
        return group_dict
    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE, DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)
    return output


WEIGHT_QUANTIZE_ASYMMETRIC = 'asymmetric'


WEIGHT_QUANTIZE_CHANGE_RATIO = 'quantize_change_ratio'


WEIGHT_QUANTIZE_CHANGE_RATIO_DEFAULT = 0.001


WEIGHT_QUANTIZE_ENABLED_DEFAULT = False


WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE = 'fp16_mixed_quantize'


WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED = 'enabled'


WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED_DEFAULT = False


WEIGHT_QUANTIZE_GROUPS = 'quantize_groups'


WEIGHT_QUANTIZE_GROUPS_DEFAULT = 1


WEIGHT_QUANTIZE_IN_FORWARD_ENABLED = 'quantize_weight_in_forward'


WEIGHT_QUANTIZE_IN_FORWARD_ENABLED_DEFAULT = False


WEIGHT_QUANTIZE_KERNEL = 'quantizer_kernel'


WEIGHT_QUANTIZE_KERNEL_DEFAULT = False


WEIGHT_QUANTIZE_NEAREST_ROUNDING = 'nearest'


WEIGHT_QUANTIZE_ROUNDING = 'rounding'


WEIGHT_QUANTIZE_ROUNDING_DEFAULT = 'nearest'


WEIGHT_QUANTIZE_SCHEDULE_OFFSET = TECHNIQUE_SCHEDULE_OFFSET


WEIGHT_QUANTIZE_SCHEDULE_OFFSET_DEFAULT = 0


WEIGHT_QUANTIZE_STOCHASTIC_ROUNDING = 'stochastic'


WEIGHT_QUANTIZE_SYMMETRIC = 'symmetric'


WEIGHT_QUANTIZE_TYPE = 'quantization_type'


WEIGHT_QUANTIZE_TYPE_DEFAULT = 'symmetric'


WEIGHT_QUANTIZE_VERBOSE = 'quantize_verbose'


WEIGHT_QUANTIZE_VERBOSE_DEFAULT = False


def get_weight_quantization_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[WEIGHT_QUANTIZE_ENABLED] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_ENABLED, WEIGHT_QUANTIZE_ENABLED_DEFAULT)
        output[WEIGHT_QUANTIZE_KERNEL] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_KERNEL, WEIGHT_QUANTIZE_KERNEL_DEFAULT)
        output[WEIGHT_QUANTIZE_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_SCHEDULE_OFFSET, WEIGHT_QUANTIZE_SCHEDULE_OFFSET_DEFAULT)
        output[WEIGHT_QUANTIZE_GROUPS] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_GROUPS, WEIGHT_QUANTIZE_GROUPS_DEFAULT)
        output[WEIGHT_QUANTIZE_VERBOSE] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_VERBOSE, WEIGHT_QUANTIZE_VERBOSE_DEFAULT)
        output[WEIGHT_QUANTIZE_TYPE] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_TYPE, WEIGHT_QUANTIZE_TYPE_DEFAULT)
        output[WEIGHT_QUANTIZE_IN_FORWARD_ENABLED] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_IN_FORWARD_ENABLED, WEIGHT_QUANTIZE_IN_FORWARD_ENABLED_DEFAULT)
        assert output[WEIGHT_QUANTIZE_TYPE] in [WEIGHT_QUANTIZE_SYMMETRIC, WEIGHT_QUANTIZE_ASYMMETRIC], f'Invalid weight quantize type. Supported types: [{WEIGHT_QUANTIZE_SYMMETRIC}, {WEIGHT_QUANTIZE_ASYMMETRIC}]'
        output[WEIGHT_QUANTIZE_ROUNDING] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_ROUNDING, WEIGHT_QUANTIZE_ROUNDING_DEFAULT)
        assert output[WEIGHT_QUANTIZE_ROUNDING] in [WEIGHT_QUANTIZE_NEAREST_ROUNDING, WEIGHT_QUANTIZE_STOCHASTIC_ROUNDING], f'Invalid weight quantize rounding. Supported types: [{WEIGHT_QUANTIZE_NEAREST_ROUNDING}, {WEIGHT_QUANTIZE_STOCHASTIC_ROUNDING}]'
        if WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE in sub_param_dict.keys():
            output[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE] = get_scalar_param(sub_param_dict[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE], WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED, WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED_DEFAULT)
            output[WEIGHT_QUANTIZE_CHANGE_RATIO] = get_scalar_param(sub_param_dict[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE], WEIGHT_QUANTIZE_CHANGE_RATIO, WEIGHT_QUANTIZE_CHANGE_RATIO_DEFAULT)
        else:
            output[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE] = WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED_DEFAULT
            output[WEIGHT_QUANTIZE_CHANGE_RATIO] = WEIGHT_QUANTIZE_CHANGE_RATIO_DEFAULT
    else:
        output[WEIGHT_QUANTIZE_ENABLED] = WEIGHT_QUANTIZE_ENABLED_DEFAULT
        output[WEIGHT_QUANTIZE_KERNEL] = WEIGHT_QUANTIZE_KERNEL_DEFAULT
        output[WEIGHT_QUANTIZE_SCHEDULE_OFFSET] = WEIGHT_QUANTIZE_SCHEDULE_OFFSET_DEFAULT
        output[WEIGHT_QUANTIZE_GROUPS] = WEIGHT_QUANTIZE_GROUPS_DEFAULT
        output[WEIGHT_QUANTIZE_VERBOSE] = WEIGHT_QUANTIZE_VERBOSE_DEFAULT
        output[WEIGHT_QUANTIZE_TYPE] = WEIGHT_QUANTIZE_TYPE_DEFAULT
        output[WEIGHT_QUANTIZE_ROUNDING] = WEIGHT_QUANTIZE_ROUNDING_DEFAULT
        output[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE] = WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED_DEFAULT
        output[WEIGHT_QUANTIZE_CHANGE_RATIO] = WEIGHT_QUANTIZE_CHANGE_RATIO_DEFAULT
    return output


def get_weight_quantization(param_dict):
    output = {}
    if WEIGHT_QUANTIZATION not in param_dict.keys():
        param_dict[WEIGHT_QUANTIZATION] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[WEIGHT_QUANTIZATION]
    output[SHARED_PARAMETERS] = get_weight_quantization_shared_parameters(sub_param_dict)
    if output[SHARED_PARAMETERS][WEIGHT_QUANTIZE_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(), f'Weigh Quantization is enabled, {DIFFERENT_GROUPS} must be specified'
    output[DIFFERENT_GROUPS] = get_weight_quantization_different_groups(sub_param_dict)
    return output


def get_compression_config(param_dict):
    output = {}
    if COMPRESSION_TRAINING not in param_dict.keys():
        param_dict[COMPRESSION_TRAINING] = {}
    sub_param_dict = param_dict[COMPRESSION_TRAINING]
    output[WEIGHT_QUANTIZATION] = get_weight_quantization(sub_param_dict)
    output[ACTIVATION_QUANTIZATION] = get_activation_quantization(sub_param_dict)
    output[SPARSE_PRUNING] = get_sparse_pruning(sub_param_dict)
    output[ROW_PRUNING] = get_row_pruning(sub_param_dict)
    output[HEAD_PRUNING] = get_head_pruning(sub_param_dict)
    output[CHANNEL_PRUNING] = get_channel_pruning(sub_param_dict)
    output[LAYER_REDUCTION] = get_layer_reduction(sub_param_dict)
    return output


CURRICULUM_ENABLED_DEFAULT_LEGACY = False


CURRICULUM_ENABLED_LEGACY = 'enabled'


CURRICULUM_LEARNING_LEGACY = 'curriculum_learning'


def get_curriculum_enabled_legacy(param_dict):
    if CURRICULUM_LEARNING_LEGACY in param_dict.keys():
        return get_scalar_param(param_dict[CURRICULUM_LEARNING_LEGACY], CURRICULUM_ENABLED_LEGACY, CURRICULUM_ENABLED_DEFAULT_LEGACY)
    else:
        return False


def get_curriculum_params_legacy(param_dict):
    if CURRICULUM_LEARNING_LEGACY in param_dict.keys():
        curriculum_params = copy.copy(param_dict[CURRICULUM_LEARNING_LEGACY])
        curriculum_params.pop(CURRICULUM_ENABLED_LEGACY)
        return curriculum_params
    else:
        return False


DATA_EFFICIENCY = 'data_efficiency'


DATA_EFFICIENCY_ENABLED = 'enabled'


DATA_EFFICIENCY_SEED = 'seed'


DATA_ROUTING = 'data_routing'


DATA_SAMPLING = 'data_sampling'


DATA_EFFICIENCY_ENABLED_DEFAULT = False


def get_data_efficiency_enabled(param_dict):
    if DATA_EFFICIENCY in param_dict.keys():
        return get_scalar_param(param_dict[DATA_EFFICIENCY], DATA_EFFICIENCY_ENABLED, DATA_EFFICIENCY_ENABLED_DEFAULT)
    else:
        return False


DATA_EFFICIENCY_SEED_DEFAULT = 1234


def get_data_efficiency_seed(param_dict):
    if DATA_EFFICIENCY in param_dict.keys():
        return get_scalar_param(param_dict[DATA_EFFICIENCY], DATA_EFFICIENCY_SEED, DATA_EFFICIENCY_SEED_DEFAULT)
    else:
        return DATA_EFFICIENCY_SEED_DEFAULT


DATA_ROUTING_ENABLED = 'enabled'


RANDOM_LTD = 'random_ltd'


DATA_ROUTING_ENABLED_DEFAULT = False


def get_data_routing_enabled(param_dict):
    if DATA_ROUTING in param_dict.keys():
        return get_scalar_param(param_dict[DATA_ROUTING], DATA_ROUTING_ENABLED, DATA_ROUTING_ENABLED_DEFAULT)
    else:
        return False


RANDOM_LTD_ENABLED = 'enabled'


RANDOM_LTD_ENABLED_DEFAULT = False


RANDOM_LTD_LAYER_TOKEN_LR_ENABLED = 'enabled'


RANDOM_LTD_LAYER_TOKEN_LR_ENABLED_DEFAULT = False


RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE = 'layer_token_lr_schedule'


def get_random_ltd_enabled(param_dict):
    if RANDOM_LTD in param_dict.keys():
        return get_scalar_param(param_dict[RANDOM_LTD], RANDOM_LTD_ENABLED, RANDOM_LTD_ENABLED_DEFAULT)
    else:
        return False


def get_random_ltd_params(param_dict):
    if RANDOM_LTD in param_dict.keys():
        random_ltd_params = copy.copy(param_dict[RANDOM_LTD])
        random_ltd_params.pop(RANDOM_LTD_ENABLED)
        return random_ltd_params
    else:
        return {}


def get_random_ltd(param_dict):
    output = {}
    output[RANDOM_LTD_ENABLED] = RANDOM_LTD_ENABLED_DEFAULT
    output[RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE] = {}
    output[RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE][RANDOM_LTD_LAYER_TOKEN_LR_ENABLED] = RANDOM_LTD_LAYER_TOKEN_LR_ENABLED_DEFAULT
    if get_random_ltd_enabled(param_dict):
        output[RANDOM_LTD_ENABLED] = get_random_ltd_enabled(param_dict)
        for key, val in get_random_ltd_params(param_dict).items():
            output[key] = val
    return output


def get_data_routing(param_dict):
    output = {}
    output[DATA_ROUTING_ENABLED] = get_data_routing_enabled(param_dict)
    if DATA_ROUTING not in param_dict.keys():
        param_dict[DATA_ROUTING] = {}
    sub_param_dict = param_dict[DATA_ROUTING]
    output[RANDOM_LTD] = get_random_ltd(sub_param_dict)
    return output


CURRICULUM_LEARNING = 'curriculum_learning'


DATA_SAMPLING_ENABLED = 'enabled'


DATA_SAMPLING_NUM_EPOCHS = 'num_epochs'


DATA_SAMPLING_NUM_WORKERS = 'num_workers'


CURRICULUM_LEARNING_ENABLED = 'enabled'


CURRICULUM_LEARNING_METRICS = 'curriculum_metrics'


CURRICULUM_LEARNING_ENABLED_DEFAULT = False


def get_curriculum_learning_enabled(param_dict):
    if CURRICULUM_LEARNING in param_dict.keys():
        return get_scalar_param(param_dict[CURRICULUM_LEARNING], CURRICULUM_LEARNING_ENABLED, CURRICULUM_LEARNING_ENABLED_DEFAULT)
    else:
        return False


def get_curriculum_learning_params(param_dict):
    if CURRICULUM_LEARNING in param_dict.keys():
        curriculum_learning_params = copy.copy(param_dict[CURRICULUM_LEARNING])
        curriculum_learning_params.pop(CURRICULUM_LEARNING_ENABLED)
        return curriculum_learning_params
    else:
        return {}


def get_curriculum_learning(param_dict):
    output = {}
    output[CURRICULUM_LEARNING_ENABLED] = get_curriculum_learning_enabled(param_dict)
    if CURRICULUM_LEARNING not in param_dict.keys():
        param_dict[CURRICULUM_LEARNING] = {}
    sub_param_dict = param_dict[CURRICULUM_LEARNING]
    if output[CURRICULUM_LEARNING_ENABLED]:
        assert CURRICULUM_LEARNING_METRICS in sub_param_dict.keys(), f'Curriculum learning is enabled, {CURRICULUM_LEARNING_METRICS} must be specified'
        for key, val in get_curriculum_learning_params(param_dict).items():
            output[key] = val
    return output


DATA_SAMPLING_ENABLED_DEFAULT = False


def get_data_sampling_enabled(param_dict):
    if DATA_SAMPLING in param_dict.keys():
        return get_scalar_param(param_dict[DATA_SAMPLING], DATA_SAMPLING_ENABLED, DATA_SAMPLING_ENABLED_DEFAULT)
    else:
        return False


DATA_SAMPLING_NUM_EPOCHS_DEFAULT = 1000


def get_data_sampling_num_epochs(param_dict):
    if DATA_SAMPLING in param_dict.keys():
        return get_scalar_param(param_dict[DATA_SAMPLING], DATA_SAMPLING_NUM_EPOCHS, DATA_SAMPLING_NUM_EPOCHS_DEFAULT)
    else:
        return DATA_SAMPLING_NUM_EPOCHS_DEFAULT


DATA_SAMPLING_NUM_WORKERS_DEFAULT = 0


def get_data_sampling_num_workers(param_dict):
    if DATA_SAMPLING in param_dict.keys():
        return get_scalar_param(param_dict[DATA_SAMPLING], DATA_SAMPLING_NUM_WORKERS, DATA_SAMPLING_NUM_WORKERS_DEFAULT)
    else:
        return DATA_SAMPLING_NUM_WORKERS_DEFAULT


def get_data_sampling(param_dict):
    output = {}
    output[DATA_SAMPLING_ENABLED] = get_data_sampling_enabled(param_dict)
    output[DATA_SAMPLING_NUM_EPOCHS] = get_data_sampling_num_epochs(param_dict)
    output[DATA_SAMPLING_NUM_WORKERS] = get_data_sampling_num_workers(param_dict)
    if DATA_SAMPLING not in param_dict.keys():
        param_dict[DATA_SAMPLING] = {}
    sub_param_dict = param_dict[DATA_SAMPLING]
    output[CURRICULUM_LEARNING] = get_curriculum_learning(sub_param_dict)
    return output


def get_data_efficiency_config(param_dict):
    output = {}
    output[DATA_EFFICIENCY_ENABLED] = get_data_efficiency_enabled(param_dict)
    output[DATA_EFFICIENCY_SEED] = get_data_efficiency_seed(param_dict)
    if DATA_EFFICIENCY not in param_dict.keys():
        param_dict[DATA_EFFICIENCY] = {}
    sub_param_dict = param_dict[DATA_EFFICIENCY]
    output[DATA_SAMPLING] = get_data_sampling(sub_param_dict)
    output[DATA_ROUTING] = get_data_routing(sub_param_dict)
    return output


DATA_TYPES = 'data_types'


def get_data_types_params(param_dict):
    return param_dict.get(DATA_TYPES, {})


DATALOADER_DROP_LAST = 'dataloader_drop_last'


DATALOADER_DROP_LAST_DEFAULT = False


def get_dataloader_drop_last(param_dict):
    return get_scalar_param(param_dict, DATALOADER_DROP_LAST, DATALOADER_DROP_LAST_DEFAULT)


DISABLE_ALLGATHER = 'disable_allgather'


DISABLE_ALLGATHER_DEFAULT = False


def get_disable_allgather(param_dict):
    return get_scalar_param(param_dict, DISABLE_ALLGATHER, DISABLE_ALLGATHER_DEFAULT)


DUMP_STATE = 'dump_state'


DUMP_STATE_DEFAULT = False


def get_dump_state(param_dict):
    return get_scalar_param(param_dict, DUMP_STATE, DUMP_STATE_DEFAULT)


DELAYED_SHIFT = 'delayed_shift'


FP16 = 'fp16'


FP16_HYSTERESIS = 'hysteresis'


FP16_HYSTERESIS_DEFAULT = 2


FP16_INITIAL_SCALE_POWER = 'initial_scale_power'


FP16_INITIAL_SCALE_POWER_DEFAULT = 32


FP16_LOSS_SCALE_WINDOW = 'loss_scale_window'


FP16_LOSS_SCALE_WINDOW_DEFAULT = 1000


FP16_MIN_LOSS_SCALE = 'min_loss_scale'


FP16_MIN_LOSS_SCALE_DEFAULT = 1


INITIAL_LOSS_SCALE = 'init_scale'


MIN_LOSS_SCALE = 'min_scale'


SCALE_WINDOW = 'scale_window'


FP16_ENABLED = 'enabled'


FP16_ENABLED_DEFAULT = False


def get_fp16_enabled(param_dict):
    if FP16 in param_dict.keys():
        return get_scalar_param(param_dict[FP16], FP16_ENABLED, FP16_ENABLED_DEFAULT)
    else:
        return False


def get_dynamic_loss_scale_args(param_dict):
    loss_scale_args = None
    if get_fp16_enabled(param_dict):
        fp16_dict = param_dict[FP16]
        dynamic_loss_args = [FP16_INITIAL_SCALE_POWER, FP16_LOSS_SCALE_WINDOW, FP16_MIN_LOSS_SCALE, FP16_HYSTERESIS]
        if any(arg in list(fp16_dict.keys()) for arg in dynamic_loss_args):
            init_scale = get_scalar_param(fp16_dict, FP16_INITIAL_SCALE_POWER, FP16_INITIAL_SCALE_POWER_DEFAULT)
            scale_window = get_scalar_param(fp16_dict, FP16_LOSS_SCALE_WINDOW, FP16_LOSS_SCALE_WINDOW_DEFAULT)
            delayed_shift = get_scalar_param(fp16_dict, FP16_HYSTERESIS, FP16_HYSTERESIS_DEFAULT)
            min_loss_scale = get_scalar_param(fp16_dict, FP16_MIN_LOSS_SCALE, FP16_MIN_LOSS_SCALE_DEFAULT)
            loss_scale_args = {INITIAL_LOSS_SCALE: 2 ** init_scale, SCALE_WINDOW: scale_window, DELAYED_SHIFT: delayed_shift, MIN_LOSS_SCALE: min_loss_scale}
    return loss_scale_args


EIGENVALUE_ENABLED_DEFAULT = False


EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT = 1


EIGENVALUE_LAYER_NAME_DEFAULT = 'bert.encoder.layer'


EIGENVALUE_LAYER_NUM_DEFAULT = 0


EIGENVALUE_MAX_ITER_DEFAULT = 100


EIGENVALUE_STABILITY_DEFAULT = 1e-06


EIGENVALUE_TOL_DEFAULT = 0.01


EIGENVALUE_VERBOSE_DEFAULT = False


EIGENVALUE = 'eigenvalue'


EIGENVALUE_ENABLED = 'enabled'


def get_eigenvalue_enabled(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_ENABLED, EIGENVALUE_ENABLED_DEFAULT)
    else:
        return EIGENVALUE_ENABLED_DEFAULT


EIGENVALUE_GAS_BOUNDARY_RESOLUTION = 'gas_boundary_resolution'


def get_eigenvalue_gas_boundary_resolution(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_GAS_BOUNDARY_RESOLUTION, EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT)
    else:
        return EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT


EIGENVALUE_LAYER_NAME = 'layer_name'


def get_eigenvalue_layer_name(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_LAYER_NAME, EIGENVALUE_LAYER_NAME_DEFAULT)
    else:
        return EIGENVALUE_LAYER_NAME_DEFAULT


EIGENVALUE_LAYER_NUM = 'layer_num'


def get_eigenvalue_layer_num(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_LAYER_NUM, EIGENVALUE_LAYER_NUM_DEFAULT)
    else:
        return EIGENVALUE_LAYER_NUM_DEFAULT


EIGENVALUE_MAX_ITER = 'max_iter'


def get_eigenvalue_max_iter(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_MAX_ITER, EIGENVALUE_MAX_ITER_DEFAULT)
    else:
        return EIGENVALUE_MAX_ITER_DEFAULT


EIGENVALUE_STABILITY = 'stability'


def get_eigenvalue_stability(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_STABILITY, EIGENVALUE_STABILITY_DEFAULT)
    else:
        return EIGENVALUE_STABILITY_DEFAULT


EIGENVALUE_TOL = 'tol'


def get_eigenvalue_tol(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_TOL, EIGENVALUE_TOL_DEFAULT)
    else:
        return EIGENVALUE_TOL_DEFAULT


EIGENVALUE_VERBOSE = 'verbose'


def get_eigenvalue_verbose(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_VERBOSE, EIGENVALUE_VERBOSE_DEFAULT)
    else:
        return EIGENVALUE_VERBOSE_DEFAULT


def get_quantize_enabled(param_dict):
    if COMPRESSION_TRAINING not in param_dict.keys():
        return False
    sub_param_dict = param_dict[COMPRESSION_TRAINING]
    output = get_weight_quantization_shared_parameters(sub_param_dict)
    return output[WEIGHT_QUANTIZE_ENABLED]


def get_eigenvalue_config(param_dict):
    if get_quantize_enabled(param_dict):
        param_dict = param_dict[QUANTIZE_TRAINING]
        assert not get_eigenvalue_enabled(param_dict), 'Eigenvalue based MoQ is temporarily disabled'
        return get_eigenvalue_enabled(param_dict), get_eigenvalue_verbose(param_dict), get_eigenvalue_max_iter(param_dict), get_eigenvalue_tol(param_dict), get_eigenvalue_stability(param_dict), get_eigenvalue_gas_boundary_resolution(param_dict), get_eigenvalue_layer_name(param_dict), get_eigenvalue_layer_num(param_dict)
    else:
        return EIGENVALUE_ENABLED_DEFAULT, EIGENVALUE_VERBOSE_DEFAULT, EIGENVALUE_MAX_ITER_DEFAULT, EIGENVALUE_TOL_DEFAULT, EIGENVALUE_STABILITY_DEFAULT, EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT, EIGENVALUE_LAYER_NAME_DEFAULT, EIGENVALUE_LAYER_NUM_DEFAULT


FP16_AUTO_CAST = 'auto_cast'


FP16_AUTO_CAST_DEFAULT = False


def get_fp16_auto_cast(param_dict):
    if get_fp16_enabled(param_dict):
        return get_scalar_param(param_dict[FP16], FP16_AUTO_CAST, FP16_AUTO_CAST_DEFAULT)


FP16_MASTER_WEIGHTS_AND_GRADS = 'fp16_master_weights_and_grads'


FP16_MASTER_WEIGHTS_AND_GRADS_DEFAULT = False


def get_fp16_master_weights_and_grads_enabled(param_dict):
    if get_fp16_enabled(param_dict):
        return get_scalar_param(param_dict[FP16], FP16_MASTER_WEIGHTS_AND_GRADS, FP16_MASTER_WEIGHTS_AND_GRADS_DEFAULT)
    else:
        return False


GRADIENT_ACCUMULATION_STEPS_DEFAULT = None


def get_gradient_accumulation_steps(param_dict):
    return get_scalar_param(param_dict, GRADIENT_ACCUMULATION_STEPS, GRADIENT_ACCUMULATION_STEPS_DEFAULT)


GRADIENT_CLIPPING = 'gradient_clipping'


GRADIENT_CLIPPING_DEFAULT = 0.0


def get_gradient_clipping(param_dict):
    return get_scalar_param(param_dict, GRADIENT_CLIPPING, GRADIENT_CLIPPING_DEFAULT)


GRADIENT_PREDIVIDE_FACTOR = 'gradient_predivide_factor'


GRADIENT_PREDIVIDE_FACTOR_DEFAULT = 1.0


def get_gradient_predivide_factor(param_dict):
    return get_scalar_param(param_dict, GRADIENT_PREDIVIDE_FACTOR, GRADIENT_PREDIVIDE_FACTOR_DEFAULT)


def get_initial_dynamic_scale(param_dict):
    if get_fp16_enabled(param_dict):
        initial_scale_power = get_scalar_param(param_dict[FP16], FP16_INITIAL_SCALE_POWER, FP16_INITIAL_SCALE_POWER_DEFAULT)
    elif get_bfloat16_enabled(param_dict):
        initial_scale_power = 0
    else:
        initial_scale_power = FP16_INITIAL_SCALE_POWER_DEFAULT
    return 2 ** initial_scale_power


FP16_LOSS_SCALE = 'loss_scale'


FP16_LOSS_SCALE_DEFAULT = 0


def get_loss_scale(param_dict):
    if get_fp16_enabled(param_dict):
        return get_scalar_param(param_dict[FP16], FP16_LOSS_SCALE, FP16_LOSS_SCALE_DEFAULT)
    elif get_bfloat16_enabled(param_dict):
        return 1.0
    else:
        return FP16_LOSS_SCALE_DEFAULT


MEMORY_BREAKDOWN = 'memory_breakdown'


MEMORY_BREAKDOWN_DEFAULT = False


def get_memory_breakdown(param_dict):
    return get_scalar_param(param_dict, MEMORY_BREAKDOWN, MEMORY_BREAKDOWN_DEFAULT)


LEGACY_FUSION = 'legacy_fusion'


LEGACY_FUSION_DEFAULT = False


OPTIMIZER = 'optimizer'


def get_optimizer_legacy_fusion(param_dict):
    if OPTIMIZER in param_dict.keys() and LEGACY_FUSION in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][LEGACY_FUSION]
    else:
        return LEGACY_FUSION_DEFAULT


OPTIMIZER_TYPE_DEFAULT = None


TYPE = 'type'


def get_optimizer_name(param_dict):
    if OPTIMIZER in param_dict.keys() and TYPE in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][TYPE]
    else:
        return OPTIMIZER_TYPE_DEFAULT


OPTIMIZER_PARAMS = 'params'


def get_optimizer_params(param_dict):
    if get_optimizer_name(param_dict) is not None and OPTIMIZER_PARAMS in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][OPTIMIZER_PARAMS]
    else:
        return None


def get_pipeline_config(param_dict):
    """Parses pipeline engine configuration. """
    default_pipeline = {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0}
    config = default_pipeline
    for key, val in param_dict.get('pipeline', {}).items():
        config[key] = val
    return config


PLD_ENABLED = 'enabled'


PLD_ENABLED_DEFAULT = False


PROGRESSIVE_LAYER_DROP = 'progressive_layer_drop'


def get_pld_enabled(param_dict):
    if PROGRESSIVE_LAYER_DROP in param_dict.keys():
        return get_scalar_param(param_dict[PROGRESSIVE_LAYER_DROP], PLD_ENABLED, PLD_ENABLED_DEFAULT)
    else:
        return False


def get_pld_params(param_dict):
    if PROGRESSIVE_LAYER_DROP in param_dict.keys():
        pld_params = copy.copy(param_dict[PROGRESSIVE_LAYER_DROP])
        pld_params.pop(PLD_ENABLED)
        return pld_params
    else:
        return False


PRESCALE_GRADIENTS = 'prescale_gradients'


PRESCALE_GRADIENTS_DEFAULT = False


def get_prescale_gradients(param_dict):
    return get_scalar_param(param_dict, PRESCALE_GRADIENTS, PRESCALE_GRADIENTS_DEFAULT)


SCHEDULER = 'scheduler'


SCHEDULER_TYPE_DEFAULT = None


def get_scheduler_name(param_dict):
    if SCHEDULER in param_dict.keys() and TYPE in param_dict[SCHEDULER].keys():
        return param_dict[SCHEDULER][TYPE]
    else:
        return SCHEDULER_TYPE_DEFAULT


SCHEDULER_PARAMS = 'params'


def get_scheduler_params(param_dict):
    if get_scheduler_name(param_dict) is not None and SCHEDULER_PARAMS in param_dict[SCHEDULER].keys():
        return param_dict[SCHEDULER][SCHEDULER_PARAMS]
    else:
        return None


SPARSE_ATTENTION = 'sparse_attention'


SPARSE_BIGBIRD_MODE = 'bigbird'


SPARSE_BSLONGFORMER_MODE = 'bslongformer'


SPARSE_DENSE_MODE = 'dense'


SPARSE_FIXED_MODE = 'fixed'


SPARSE_VARIABLE_MODE = 'variable'


SPARSE_MODE = 'mode'


SPARSE_MODE_DEFAULT = SPARSE_FIXED_MODE


def get_sparse_attention_mode(param_dict):
    if SPARSE_MODE in param_dict.keys():
        return param_dict[SPARSE_MODE]
    else:
        return SPARSE_MODE_DEFAULT


SPARSE_BLOCK = 'block'


SPARSE_BLOCK_DEFAULT = 16


SPARSE_DIFFERENT_LAYOUT_PER_HEAD = 'different_layout_per_head'


SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT = False


SPARSE_NUM_GLOBAL_BLOCKS = 'num_global_blocks'


SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT = 1


SPARSE_NUM_RANDOM_BLOCKS = 'num_random_blocks'


SPARSE_NUM_RANDOM_BLOCKS_DEFAULT = 0


SPARSE_NUM_SLIDING_WINDOW_BLOCKS = 'num_sliding_window_blocks'


SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT = 3


def get_sparse_bigbird_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(sparsity, SPARSE_DIFFERENT_LAYOUT_PER_HEAD, SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT)
    num_random_blocks = get_scalar_param(sparsity, SPARSE_NUM_RANDOM_BLOCKS, SPARSE_NUM_RANDOM_BLOCKS_DEFAULT)
    num_sliding_window_blocks = get_scalar_param(sparsity, SPARSE_NUM_SLIDING_WINDOW_BLOCKS, SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT)
    num_global_blocks = get_scalar_param(sparsity, SPARSE_NUM_GLOBAL_BLOCKS, SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT)
    return {SPARSE_MODE: SPARSE_BIGBIRD_MODE, SPARSE_BLOCK: block, SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head, SPARSE_NUM_RANDOM_BLOCKS: num_random_blocks, SPARSE_NUM_SLIDING_WINDOW_BLOCKS: num_sliding_window_blocks, SPARSE_NUM_GLOBAL_BLOCKS: num_global_blocks}


SPARSE_GLOBAL_BLOCK_END_INDICES = 'global_block_end_indices'


SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT = None


SPARSE_GLOBAL_BLOCK_INDICES = 'global_block_indices'


SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT = [0]


def get_sparse_bslongformer_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(sparsity, SPARSE_DIFFERENT_LAYOUT_PER_HEAD, SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT)
    num_sliding_window_blocks = get_scalar_param(sparsity, SPARSE_NUM_SLIDING_WINDOW_BLOCKS, SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT)
    global_block_indices = get_scalar_param(sparsity, SPARSE_GLOBAL_BLOCK_INDICES, SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT)
    global_block_end_indices = get_scalar_param(sparsity, SPARSE_GLOBAL_BLOCK_END_INDICES, SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT)
    return {SPARSE_MODE: SPARSE_BSLONGFORMER_MODE, SPARSE_BLOCK: block, SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head, SPARSE_NUM_SLIDING_WINDOW_BLOCKS: num_sliding_window_blocks, SPARSE_GLOBAL_BLOCK_INDICES: global_block_indices, SPARSE_GLOBAL_BLOCK_END_INDICES: global_block_end_indices}


def get_sparse_dense_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    return {SPARSE_MODE: SPARSE_DENSE_MODE, SPARSE_BLOCK: block}


SPARSE_ATTENTION_TYPE = 'attention'


SPARSE_ATTENTION_TYPE_DEFAULT = 'bidirectional'


SPARSE_HORIZONTAL_GLOBAL_ATTENTION = 'horizontal_global_attention'


SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT = False


SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS = 'num_different_global_patterns'


SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS_DEFAULT = 1


SPARSE_NUM_LOCAL_BLOCKS = 'num_local_blocks'


SPARSE_NUM_LOCAL_BLOCKS_DEFAULT = 4


def get_sparse_fixed_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(sparsity, SPARSE_DIFFERENT_LAYOUT_PER_HEAD, SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT)
    num_local_blocks = get_scalar_param(sparsity, SPARSE_NUM_LOCAL_BLOCKS, SPARSE_NUM_LOCAL_BLOCKS_DEFAULT)
    num_global_blocks = get_scalar_param(sparsity, SPARSE_NUM_GLOBAL_BLOCKS, SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT)
    attention = get_scalar_param(sparsity, SPARSE_ATTENTION_TYPE, SPARSE_ATTENTION_TYPE_DEFAULT)
    horizontal_global_attention = get_scalar_param(sparsity, SPARSE_HORIZONTAL_GLOBAL_ATTENTION, SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT)
    num_different_global_patterns = get_scalar_param(sparsity, SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS, SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS_DEFAULT)
    return {SPARSE_MODE: SPARSE_FIXED_MODE, SPARSE_BLOCK: block, SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head, SPARSE_NUM_LOCAL_BLOCKS: num_local_blocks, SPARSE_NUM_GLOBAL_BLOCKS: num_global_blocks, SPARSE_ATTENTION_TYPE: attention, SPARSE_HORIZONTAL_GLOBAL_ATTENTION: horizontal_global_attention, SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS: num_different_global_patterns}


SPARSE_LOCAL_WINDOW_BLOCKS = 'local_window_blocks'


SPARSE_LOCAL_WINDOW_BLOCKS_DEFAULT = [4]


def get_sparse_variable_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(sparsity, SPARSE_DIFFERENT_LAYOUT_PER_HEAD, SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT)
    num_random_blocks = get_scalar_param(sparsity, SPARSE_NUM_RANDOM_BLOCKS, SPARSE_NUM_RANDOM_BLOCKS_DEFAULT)
    local_window_blocks = get_scalar_param(sparsity, SPARSE_LOCAL_WINDOW_BLOCKS, SPARSE_LOCAL_WINDOW_BLOCKS_DEFAULT)
    global_block_indices = get_scalar_param(sparsity, SPARSE_GLOBAL_BLOCK_INDICES, SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT)
    global_block_end_indices = get_scalar_param(sparsity, SPARSE_GLOBAL_BLOCK_END_INDICES, SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT)
    attention = get_scalar_param(sparsity, SPARSE_ATTENTION_TYPE, SPARSE_ATTENTION_TYPE_DEFAULT)
    horizontal_global_attention = get_scalar_param(sparsity, SPARSE_HORIZONTAL_GLOBAL_ATTENTION, SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT)
    return {SPARSE_MODE: SPARSE_VARIABLE_MODE, SPARSE_BLOCK: block, SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head, SPARSE_NUM_RANDOM_BLOCKS: num_random_blocks, SPARSE_LOCAL_WINDOW_BLOCKS: local_window_blocks, SPARSE_GLOBAL_BLOCK_INDICES: global_block_indices, SPARSE_GLOBAL_BLOCK_END_INDICES: global_block_end_indices, SPARSE_ATTENTION_TYPE: attention, SPARSE_HORIZONTAL_GLOBAL_ATTENTION: horizontal_global_attention}


def get_sparse_attention(param_dict):
    if SPARSE_ATTENTION in param_dict.keys():
        sparsity = param_dict[SPARSE_ATTENTION]
        mode = get_sparse_attention_mode(sparsity)
        if mode == SPARSE_DENSE_MODE:
            return get_sparse_dense_config(sparsity)
        elif mode == SPARSE_FIXED_MODE:
            return get_sparse_fixed_config(sparsity)
        elif mode == SPARSE_VARIABLE_MODE:
            return get_sparse_variable_config(sparsity)
        elif mode == SPARSE_BIGBIRD_MODE:
            return get_sparse_bigbird_config(sparsity)
        elif mode == SPARSE_BSLONGFORMER_MODE:
            return get_sparse_bslongformer_config(sparsity)
        else:
            raise NotImplementedError(f'Given sparsity mode, {mode}, has not been implemented yet!')
    else:
        return None


SPARSE_GRADIENTS = 'sparse_gradients'


SPARSE_GRADIENTS_DEFAULT = False


def get_sparse_gradients_enabled(param_dict):
    return get_scalar_param(param_dict, SPARSE_GRADIENTS, SPARSE_GRADIENTS_DEFAULT)


STEPS_PER_PRINT = 'steps_per_print'


STEPS_PER_PRINT_DEFAULT = 10


def get_steps_per_print(param_dict):
    return get_scalar_param(param_dict, STEPS_PER_PRINT, STEPS_PER_PRINT_DEFAULT)


TRAIN_BATCH_SIZE_DEFAULT = None


def get_train_batch_size(param_dict):
    return get_scalar_param(param_dict, TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE_DEFAULT)


TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT = None


def get_train_micro_batch_size_per_gpu(param_dict):
    return get_scalar_param(param_dict, TRAIN_MICRO_BATCH_SIZE_PER_GPU, TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT)


WALL_CLOCK_BREAKDOWN = 'wall_clock_breakdown'


WALL_CLOCK_BREAKDOWN_DEFAULT = False


def get_wall_clock_breakdown(param_dict):
    return get_scalar_param(param_dict, WALL_CLOCK_BREAKDOWN, WALL_CLOCK_BREAKDOWN_DEFAULT)


ZERO_ALLOW_UNTESTED_OPTIMIZER = 'zero_allow_untested_optimizer'


ZERO_ALLOW_UNTESTED_OPTIMIZER_DEFAULT = False


def get_zero_allow_untested_optimizer(param_dict):
    return get_scalar_param(param_dict, ZERO_ALLOW_UNTESTED_OPTIMIZER, ZERO_ALLOW_UNTESTED_OPTIMIZER_DEFAULT)


def get_caller_func(frame=3):
    return sys._getframe(frame).f_code.co_name


cdb = None


def print_rank_0(message, debug=False, force=False):
    rank = dist.get_rank()
    if rank == 0 and (debug or force):
        None


def calc_bw_log(comm_op, size, duration):
    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == 'all_to_all_single':
        tput = size / duration
        busbw = size / duration * ((n - 1) / n)
    elif comm_op == 'all_gather' or comm_op == 'all_gather_base' or comm_op == 'reduce_scatter' or comm_op == 'reduce_scatter_base':
        size *= n
        tput = size / duration
        busbw = size / duration * ((n - 1) / n)
    elif comm_op == 'all_reduce':
        tput = size * 2 / duration
        busbw = size / duration * (2 * (n - 1) / n)
    elif comm_op == 'send' or comm_op == 'recv' or comm_op == 'isend' or comm_op == 'irecv' or comm_op == 'broadcast' or comm_op == 'reduce' or comm_op == 'gather' or comm_op == 'scatter' or comm_op == 'barrier':
        tput = size / duration
        busbw = tput
    else:
        print_rank_0('wrong comm_op specified')
        exit(0)
    tput *= 8
    busbw *= 8
    tput /= 1000000.0
    busbw /= 1000000.0
    return tput, busbw


def convert_size(size_bytes):
    if size_bytes == 0:
        return '0B'
    size_name = 'B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return '%s %s' % (s, size_name[i])


class CommsLogger:

    def __init__(self):
        self.comms_dict = {}
        self.verbose = COMMS_LOGGER_VERBOSE_DEFAULT
        self.debug = COMMS_LOGGER_DEBUG_DEFAULT
        self.prof_ops = COMMS_LOGGER_PROF_OPS_DEFAULT
        self.prof_all = COMMS_LOGGER_PROF_ALL_DEFAULT
        self.enabled = COMMS_LOGGER_ENABLED_DEFAULT

    def configure(self, comms_config):
        self.enabled = comms_config.comms_logger_enabled
        if self.enabled:
            self.verbose = comms_config.comms_logger.verbose
            self.debug = comms_config.comms_logger.debug
            self.prof_ops = comms_config.comms_logger.prof_ops
            self.prof_all = comms_config.comms_logger.prof_all

    def start_profiling_comms(self):
        self.prof_all = True

    def stop_profiling_comms(self):
        self.prof_all = True

    def start_profiling_op(self, op_name_list):
        self.prof_ops = list(set(self.prof_ops) | set(op_name_list))

    def stop_profiling_op(self, op_name_list):
        self.prof_ops = [op for op in self.prof_ops if op not in op_name_list]

    def append(self, raw_name, record_name, latency, msg_size):
        algbw, busbw = calc_bw_log(raw_name, msg_size, latency)
        if record_name in self.comms_dict.keys():
            if msg_size in self.comms_dict[record_name].keys():
                self.comms_dict[record_name][msg_size][0] += 1
                self.comms_dict[record_name][msg_size][1].append(latency)
                self.comms_dict[record_name][msg_size][2].append(algbw)
                self.comms_dict[record_name][msg_size][3].append(busbw)
            else:
                self.comms_dict[record_name][msg_size] = [1, [latency], [algbw], [busbw]]
        else:
            self.comms_dict[record_name] = {msg_size: [1, [latency], [algbw], [busbw]]}
        if self.verbose:
            n = dist.get_world_size()
            log_str = f'rank={dist.get_rank()} | comm op: ' + record_name + ' | time (ms): {:.2f}'.format(latency)
            log_str += ' | msg size: ' + convert_size(msg_size)
            log_str += ' | algbw (Gbps): {:.2f} '.format(algbw)
            log_str += ' | busbw (Gbps): {:.2f} '.format(busbw)
            log_dist(log_str, [0])

    def log_all(self):
        None
        for record_name in self.comms_dict.keys():
            None
            for msg_size, vals in sorted(self.comms_dict[record_name].items()):
                count = vals[0]
                total_lat = sum(vals[1])
                avg_lat = trim_mean(vals[1], 0.1)
                avg_algbw = trim_mean(vals[2], 0.1)
                avg_busbw = trim_mean(vals[3], 0.1)
                None


comms_logger = CommsLogger()


def get_debug_log_name(func_args, debug):
    if debug:
        return func_args['log_name'] + ' | [Caller Func: ' + get_caller_func() + ']'
    else:
        return func_args['log_name']


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_tensor_kwarg(func, kwargs):
    func_args = get_default_args(func)
    func_args.update(kwargs)
    arg = None
    if 'tensor' in func_args:
        arg = func_args['tensor']
    elif 'input_list' in func_args:
        arg = func_args['input_list']
    elif 'input_tensor_list' in func_args:
        arg = func_args['input_tensor_list']
    return arg


def get_tensor_position(func):
    sig_params = inspect.signature(func).parameters
    arg = None
    if 'tensor' in sig_params:
        arg = 'tensor'
    elif 'input_list' in sig_params:
        arg = 'input_list'
    elif 'input_tensor_list' in sig_params:
        arg = 'input_tensor_list'
    if arg is None:
        return -1
    else:
        return list(sig_params).index(arg)


def get_msg_size_from_args(func, *args, **kwargs):
    tensor_arg_position = -1
    tensor_arg = None
    if len(args) > 0:
        tensor_arg_position = get_tensor_position(func)
        if tensor_arg_position > -1:
            tensor_arg = args[get_tensor_position(func)]
    if tensor_arg is None and len(kwargs) > 0:
        tensor_arg = get_tensor_kwarg(func, kwargs)
    if tensor_arg is None:
        return 0
    elif type(tensor_arg) is list:
        return sum(x.element_size() * x.nelement() for x in tensor_arg)
    else:
        return tensor_arg.element_size() * tensor_arg.nelement()


timers = SynchronizedWallClockTimer()


def timed_op(func):

    def log_wrapper(*args, **kwargs):
        if comms_logger.enabled:
            if 'prof' in kwargs and kwargs['prof'] or comms_logger.prof_all or 'log_name' in kwargs and kwargs['log_name'] in comms_logger.prof_ops:
                func_args = get_default_args(func)
                func_args.update(kwargs)
                msg_size = get_msg_size_from_args(func, *args, **kwargs)
                log_name = get_debug_log_name(func_args, comms_logger.debug)
                timers(log_name).start()
        try:
            return func(*args, **kwargs)
        finally:
            if comms_logger.enabled:
                torch.cuda.synchronize()
                if cdb.using_mpi:
                    cdb.barrier()
                if 'prof' in kwargs and kwargs['prof'] or comms_logger.prof_all or 'log_name' in kwargs and kwargs['log_name'] in comms_logger.prof_ops:
                    log_name = get_debug_log_name(func_args, comms_logger.debug)
                    raw_name = func.__name__
                    timers(log_name).stop()
                    time_elapsed = timers(log_name).elapsed(reset=False)
                    comms_logger.append(raw_name, log_name, time_elapsed, msg_size)
    return log_wrapper


@timed_op
def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False, prof=False, log_name='reduce', debug=get_caller_func()):
    global cdb
    return cdb.reduce(tensor=tensor, dst=dst, op=op, group=group, async_op=async_op)


class OffloadDeviceEnum(str, Enum):
    """ Enum for valid offload devices """
    none = 'none'
    cpu = 'cpu'
    nvme = 'nvme'


class pp_int(int):
    """
    A wrapper for integers that will return a custom string or comma-formatted
    string of the integer. For example, print(pp_int(1e5)) will return
    "10,000". This is useful mainly for auto-generated documentation purposes.
    """

    def __new__(cls, val, custom_print_str=None):
        inst = super().__new__(cls, val)
        inst.custom_print_str = custom_print_str
        return inst

    def __repr__(self):
        if self.custom_print_str:
            return self.custom_print_str
        return f'{self.real:,}'


ZERO_OPTIMIZATION = 'zero_optimization'


ZERO_FORMAT = """
ZeRO optimization should be enabled as:
"session_params": {
  "zero_optimization": {
    "stage": [0|1|2],
    "stage3_max_live_parameters" : 1000000000,
    "stage3_max_reuse_distance" : 1000000000,
    "allgather_partitions": [true|false],
    "allgather_bucket_size": 500000000,
    "reduce_scatter": [true|false],
    "contiguous_gradients" : [true|false]
    "overlap_comm": [true|false],
    "reduce_bucket_size": 500000000,
    "load_from_fp32_weights": [true|false],
    "cpu_offload": [true|false] (deprecated),
    "cpu_offload_params" : [true|false] (deprecated),
    "cpu_offload_use_pin_memory": [true|false] (deprecated),
    "sub_group_size" : 1000000000000,
    "offload_param": {...},
    "offload_optimizer": {...},
    "ignore_unused_parameters": [true|false],
    "round_robin_gradients": [true|false]
    }
}
"""


def read_zero_config_deprecated(param_dict):
    zero_config_dict = {}
    zero_config_dict['stage'] = 1 if param_dict[ZERO_OPTIMIZATION] else 0
    if zero_config_dict['stage'] > 0:
        zero_config_dict['allgather_bucket_size'] = get_scalar_param(param_dict, 'allgather_size', 500000000.0)
    logger.warning('DeepSpeedConfig: this format of ZeRO optimization setup is deprecated. Please use the following format: {}'.format(ZERO_FORMAT))
    return zero_config_dict


def get_zero_config(param_dict):
    if ZERO_OPTIMIZATION in param_dict:
        zero_config_dict = param_dict[ZERO_OPTIMIZATION]
        if isinstance(zero_config_dict, bool):
            zero_config_dict = read_zero_config_deprecated(param_dict)
    else:
        zero_config_dict = {}
    return DeepSpeedZeroConfig(**zero_config_dict)


class DeepSpeedConfig(object):

    def __init__(self, config: Union[str, dict], mpu=None):
        super(DeepSpeedConfig, self).__init__()
        if isinstance(config, dict):
            self._param_dict = config
        elif os.path.exists(config):
            self._param_dict = json.load(open(config, 'r'), object_pairs_hook=dict_raise_error_on_duplicate_keys)
        else:
            try:
                config_decoded = base64.urlsafe_b64decode(config).decode('utf-8')
                self._param_dict = json.loads(config_decoded)
            except (UnicodeDecodeError, AttributeError):
                raise ValueError(f'Expected a string path to an existing deepspeed config, or a dictionary or a valid base64. Received: {config}')
        try:
            self.global_rank = dist.get_rank()
            if mpu is None:
                self.world_size = dist.get_world_size()
            else:
                self.world_size = mpu.get_data_parallel_world_size()
        except:
            self.global_rank = 0
            self.world_size = 1
        self.elasticity_enabled = elasticity_enabled(self._param_dict)
        if self.elasticity_enabled:
            logger.info('DeepSpeed elasticity support enabled')
            final_batch_size, valid_gpus, micro_batch_size = compute_elastic_config(ds_config=self._param_dict, target_deepspeed_version=__version__, world_size=self.world_size)
            elastic_dict = self._param_dict[ELASTICITY]
            ensure_immutable_elastic_config(runtime_elastic_config_dict=elastic_dict)
            self.elastic_model_parallel_size = elastic_dict.get(MODEL_PARLLEL_SIZE, MODEL_PARLLEL_SIZE_DEFAULT)
            if self.elastic_model_parallel_size < 1:
                raise ElasticityConfigError(f'Model-Parallel size cannot be less than 1, given model-parallel size: {self.elastic_model_parallel_size}')
            self.num_gpus_per_node = elastic_dict.get(NUM_GPUS_PER_NODE, NUM_GPUS_PER_NODE_DEFAULT)
            if self.num_gpus_per_node < 1:
                raise ElasticityConfigError(f'NUmber of GPUs per node cannot be less than 1, given number of GPUs per node: {self.num_gpus_per_node}')
            ignore_non_elastic_batch_info = elastic_dict.get(IGNORE_NON_ELASTIC_BATCH_INFO, IGNORE_NON_ELASTIC_BATCH_INFO_DEFAULT)
            if not ignore_non_elastic_batch_info:
                batch_params = [TRAIN_BATCH_SIZE, TRAIN_MICRO_BATCH_SIZE_PER_GPU, GRADIENT_ACCUMULATION_STEPS]
                if any(map(lambda t: t in self._param_dict, batch_params)):
                    raise ElasticityConfigError(f"One or more batch related parameters were found in your ds_config ({TRAIN_BATCH_SIZE}, {TRAIN_MICRO_BATCH_SIZE_PER_GPU}, and/or {GRADIENT_ACCUMULATION_STEPS}). These parameters *will not be used* since elastic training is enabled, which takes control of these parameters. If you want to suppress this error (the parameters will be silently ignored) please set {IGNORE_NON_ELASTIC_BATCH_INFO}':true in your elasticity config.")
            gradient_accu_steps = final_batch_size // (micro_batch_size * self.world_size)
            if TRAIN_BATCH_SIZE in self._param_dict:
                logger.warning(f'[Elasticity] overriding training_batch_size: {self._param_dict[TRAIN_BATCH_SIZE]} -> {final_batch_size}')
            if TRAIN_MICRO_BATCH_SIZE_PER_GPU in self._param_dict:
                logger.warning(f'[Elasticity] overriding train_micro_batch_size_per_gpu: {self._param_dict[TRAIN_MICRO_BATCH_SIZE_PER_GPU]} -> {micro_batch_size}')
            if GRADIENT_ACCUMULATION_STEPS in self._param_dict:
                logger.warning(f'[Elasticity] overriding gradient_accumulation_steps: {self._param_dict[GRADIENT_ACCUMULATION_STEPS]} -> {gradient_accu_steps}')
            logger.info(f'[Elasticity] valid GPU counts: {valid_gpus}')
            self._param_dict[TRAIN_BATCH_SIZE] = final_batch_size
            self._param_dict[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = micro_batch_size
            self._param_dict[GRADIENT_ACCUMULATION_STEPS] = gradient_accu_steps
        self._initialize_params(copy.copy(self._param_dict))
        self._configure_train_batch_size()
        self._do_sanity_check()

    def _initialize_params(self, param_dict):
        self.train_batch_size = get_train_batch_size(param_dict)
        self.train_micro_batch_size_per_gpu = get_train_micro_batch_size_per_gpu(param_dict)
        self.gradient_accumulation_steps = get_gradient_accumulation_steps(param_dict)
        self.steps_per_print = get_steps_per_print(param_dict)
        self.dump_state = get_dump_state(param_dict)
        self.disable_allgather = get_disable_allgather(param_dict)
        self.communication_data_type = get_communication_data_type(param_dict)
        self.prescale_gradients = get_prescale_gradients(param_dict)
        self.gradient_predivide_factor = get_gradient_predivide_factor(param_dict)
        self.sparse_gradients_enabled = get_sparse_gradients_enabled(param_dict)
        self.zero_config = get_zero_config(param_dict)
        self.zero_optimization_stage = self.zero_config.stage
        self.zero_enabled = self.zero_optimization_stage > 0
        self.activation_checkpointing_config = DeepSpeedActivationCheckpointingConfig(param_dict)
        self.comms_config = DeepSpeedCommsConfig(param_dict)
        self.monitor_config = DeepSpeedMonitorConfig(param_dict)
        self.gradient_clipping = get_gradient_clipping(param_dict)
        self.fp16_enabled = get_fp16_enabled(param_dict)
        self.fp16_auto_cast = get_fp16_auto_cast(param_dict)
        self.bfloat16_enabled = get_bfloat16_enabled(param_dict)
        assert not (self.fp16_enabled and self.bfloat16_enabled), 'bfloat16 and fp16 modes cannot be simultaneously enabled'
        self.fp16_master_weights_and_gradients = get_fp16_master_weights_and_grads_enabled(param_dict)
        self.amp_enabled = get_amp_enabled(param_dict)
        self.amp_params = get_amp_params(param_dict)
        self.loss_scale = get_loss_scale(param_dict)
        self.initial_dynamic_scale = get_initial_dynamic_scale(param_dict)
        self.dynamic_loss_scale_args = get_dynamic_loss_scale_args(param_dict)
        self.compression_config = get_compression_config(param_dict)
        self.optimizer_name = get_optimizer_name(param_dict)
        if self.optimizer_name is not None and self.optimizer_name.lower() in DEEPSPEED_OPTIMIZERS:
            self.optimizer_name = self.optimizer_name.lower()
        self.optimizer_params = get_optimizer_params(param_dict)
        self.optimizer_legacy_fusion = get_optimizer_legacy_fusion(param_dict)
        self.zero_allow_untested_optimizer = get_zero_allow_untested_optimizer(param_dict)
        self.scheduler_name = get_scheduler_name(param_dict)
        self.scheduler_params = get_scheduler_params(param_dict)
        self.flops_profiler_config = DeepSpeedFlopsProfilerConfig(param_dict)
        self.wall_clock_breakdown = get_wall_clock_breakdown(param_dict) | self.flops_profiler_config.enabled
        self.memory_breakdown = get_memory_breakdown(param_dict)
        self.autotuning_config = DeepSpeedAutotuningConfig(param_dict)
        self.eigenvalue_enabled, self.eigenvalue_verbose, self.eigenvalue_max_iter, self.eigenvalue_tol, self.eigenvalue_stability, self.eigenvalue_gas_boundary_resolution, self.eigenvalue_layer_name, self.eigenvalue_layer_num = get_eigenvalue_config(param_dict)
        self.sparse_attention = get_sparse_attention(param_dict)
        self.pipeline = get_pipeline_config(param_dict)
        self.pld_enabled = get_pld_enabled(param_dict)
        self.pld_params = get_pld_params(param_dict)
        self.curriculum_enabled_legacy = get_curriculum_enabled_legacy(param_dict)
        self.curriculum_params_legacy = get_curriculum_params_legacy(param_dict)
        self.data_efficiency_enabled = get_data_efficiency_enabled(param_dict)
        self.data_efficiency_config = get_data_efficiency_config(param_dict)
        checkpoint_params = get_checkpoint_params(param_dict)
        validation_mode = get_checkpoint_tag_validation_mode(checkpoint_params)
        self.checkpoint_tag_validation_enabled = validation_mode != ValidationMode.IGNORE
        self.checkpoint_tag_validation_fail = validation_mode == ValidationMode.FAIL
        self.load_universal_checkpoint = checkpoint_params.get(LOAD_UNIVERSAL_CHECKPOINT, LOAD_UNIVERSAL_CHECKPOINT_DEFAULT)
        self.use_node_local_storage = checkpoint_params.get(USE_NODE_LOCAL_STORAGE_CHECKPOINT, USE_NODE_LOCAL_STORAGE_CHECKPOINT_DEFAULT)
        data_types_params = get_data_types_params(param_dict)
        self.grad_accum_dtype = data_types_params.get(GRAD_ACCUM_DTYPE, GRAD_ACCUM_DTYPE_DEFAULT)
        par_write_pipe = get_checkpoint_parallel_write_pipeline(checkpoint_params)
        self.checkpoint_parallel_write_pipeline = par_write_pipe
        self.aio_config = get_aio_config(param_dict)
        self.dataloader_drop_last = get_dataloader_drop_last(param_dict)
        self.nebula_config = DeepSpeedNebulaConfig(param_dict)

    def _batch_assertion(self):
        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps
        assert train_batch > 0, f'Train batch size: {train_batch} has to be greater than 0'
        assert micro_batch > 0, f'Micro batch size per gpu: {micro_batch} has to be greater than 0'
        assert grad_acc > 0, f'Gradient accumulation steps: {grad_acc} has to be greater than 0'
        assert train_batch == micro_batch * grad_acc * self.world_size, f'Check batch related parameters. train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size {train_batch} != {micro_batch} * {grad_acc} * {self.world_size}'

    def _set_batch_related_parameters(self):
        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps
        if train_batch is not None and micro_batch is not None and grad_acc is not None:
            return
        elif train_batch is not None and micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= self.world_size
            self.gradient_accumulation_steps = grad_acc
        elif train_batch is not None and grad_acc is not None:
            micro_batch = train_batch // self.world_size
            micro_batch //= grad_acc
            self.train_micro_batch_size_per_gpu = micro_batch
        elif micro_batch is not None and grad_acc is not None:
            train_batch_size = micro_batch * grad_acc
            train_batch_size *= self.world_size
            self.train_batch_size = train_batch_size
        elif train_batch is not None:
            self.gradient_accumulation_steps = 1
            self.train_micro_batch_size_per_gpu = train_batch // self.world_size
        elif micro_batch is not None:
            self.train_batch_size = micro_batch * self.world_size
            self.gradient_accumulation_steps = 1
        else:
            assert False, 'Either train_batch_size or train_micro_batch_size_per_gpu needs to be provided'

    def _configure_train_batch_size(self):
        self._set_batch_related_parameters()
        self._batch_assertion()

    def _do_sanity_check(self):
        self._do_error_check()
        self._do_warning_check()

    def print_user_config(self):
        logger.info('  json = {}'.format(json.dumps(self._param_dict, sort_keys=True, indent=4, cls=ScientificNotationEncoder, separators=(',', ':'))))

    def print(self, name):
        logger.info('{}:'.format(name))
        for arg in sorted(vars(self)):
            if arg != '_param_dict':
                dots = '.' * (29 - len(arg))
                logger.info('  {} {} {}'.format(arg, dots, getattr(self, arg)))
        self.print_user_config()

    def _do_error_check(self):
        assert self.train_micro_batch_size_per_gpu, 'DeepSpeedConfig: {} is not defined'.format(TRAIN_MICRO_BATCH_SIZE_PER_GPU)
        assert self.gradient_accumulation_steps, 'DeepSpeedConfig: {} is not defined'.format(GRADIENT_ACCUMULATION_STEPS)
        if self.zero_enabled:
            assert self.zero_optimization_stage <= ZeroStageEnum.max_stage, 'DeepSpeedConfig: Maximum supported ZeRO stage is {}'.format(ZeroStageEnum.max_stage)
        if self.fp16_master_weights_and_gradients:
            assert self.zero_enabled and self.zero_optimization_stage == ZeroStageEnum.gradients, 'Fp16_master_weights_and_grads is only supported with ZeRO Stage 2 for now.'

    def _do_warning_check(self):
        fp16_enabled = self.fp16_enabled
        vocabulary_size = self._param_dict.get(VOCABULARY_SIZE, VOCABULARY_SIZE_DEFAULT)
        if vocabulary_size and vocabulary_size % TENSOR_CORE_ALIGN_SIZE != 0:
            logger.warning('DeepSpeedConfig: vocabulary size {} is not aligned to {}, may import tensor core utilization.'.format(vocabulary_size, TENSOR_CORE_ALIGN_SIZE))
        if self.optimizer_params is not None and MAX_GRAD_NORM in self.optimizer_params.keys() and self.optimizer_params[MAX_GRAD_NORM] > 0:
            if fp16_enabled:
                if self.global_rank == 0:
                    logger.warning('DeepSpeedConfig: In FP16 mode, DeepSpeed will pass {}:{} to FP16 wrapper'.format(MAX_GRAD_NORM, self.optimizer_params[MAX_GRAD_NORM]))
            else:
                if self.global_rank == 0:
                    logger.warning('DeepSpeedConfig: In FP32 mode, DeepSpeed does not permit MAX_GRAD_NORM ({}) > 0, setting to zero'.format(self.optimizer_params[MAX_GRAD_NORM]))
                self.optimizer_params[MAX_GRAD_NORM] = 0.0


def _configure_using_config_file(config, mpu=None):
    global num_layers, PARTITION_ACTIVATIONS, CONTIGUOUS_CHECKPOINTING, CPU_CHECKPOINT, SYNCHRONIZE, PROFILE_TIME
    config = DeepSpeedConfig(config, mpu=mpu).activation_checkpointing_config
    if dist.get_rank() == 0:
        logger.info(config.repr())
    PARTITION_ACTIVATIONS = config.partition_activations
    CONTIGUOUS_CHECKPOINTING = config.contiguous_memory_optimization
    num_layers = config.number_checkpoints
    CPU_CHECKPOINT = config.cpu_checkpointing
    SYNCHRONIZE = config.synchronize_checkpoint_boundary
    PROFILE_TIME = config.profile


def configure(mpu_, deepspeed_config=None, partition_activations=None, contiguous_checkpointing=None, num_checkpoints=None, checkpoint_in_cpu=None, synchronize=None, profile=None):
    """Configure DeepSpeed Activation Checkpointing.

    Arguments:
        mpu_: Optional: An object that implements the following methods
            get_model_parallel_rank/group/world_size, and get_data_parallel_rank/group/world_size

        deepspeed_config: Optional: DeepSpeed Config json file when provided will be used to
            configure DeepSpeed Activation Checkpointing

        partition_activations: Optional: Partitions activation checkpoint across model parallel
            GPUs when enabled. By default False. Will overwrite deepspeed_config if provided

        contiguous_checkpointing: Optional: Copies activation checkpoints to a contiguous memory
            buffer. Works only with homogeneous checkpoints when partition_activations is enabled.
            Must provide num_checkpoints. By default False. Will overwrite deepspeed_config if
            provided

        num_checkpoints: Optional: Number of activation checkpoints stored during the forward
            propagation of the model. Used to calculate the buffer size for contiguous_checkpointing
            Will overwrite deepspeed_config if provided

        checkpoint_in_cpu: Optional: Moves the activation checkpoint to CPU. Only works with
            partition_activation. Default is false. Will overwrite deepspeed_config if provided

        synchronize: Optional: Performs torch.cuda.synchronize() at the beginning and end of
            each call to deepspeed.checkpointing.checkpoint for both forward and backward pass.
            By default false. Will overwrite deepspeed_config if provided

        profile: Optional: Logs the forward and backward time for each
            deepspeed.checkpointing.checkpoint invocation. Will overwrite deepspeed_config
            if provided

    Returns:
        None
    """
    global mpu, num_layers, deepspeed_checkpointing_enabled
    global PARTITION_ACTIVATIONS, CONTIGUOUS_CHECKPOINTING, CPU_CHECKPOINT, SYNCHRONIZE, PROFILE_TIME
    _configure_defaults()
    if mpu_ is not None:
        mpu = mpu_
    if deepspeed_config is not None:
        _configure_using_config_file(deepspeed_config, mpu=mpu)
    if partition_activations is not None:
        PARTITION_ACTIVATIONS = partition_activations
    if contiguous_checkpointing is not None:
        CONTIGUOUS_CHECKPOINTING = contiguous_checkpointing
    if num_checkpoints is not None:
        num_layers = num_checkpoints
    if checkpoint_in_cpu is not None:
        CPU_CHECKPOINT = checkpoint_in_cpu
    if synchronize is not None:
        SYNCHRONIZE = synchronize
    if profile is not None:
        PROFILE_TIME = profile
    if CONTIGUOUS_CHECKPOINTING:
        assert PARTITION_ACTIVATIONS, 'Contiguous Checkpointing is only available with partitioned activations. Set partitioned activations to true in deepspeed config'
    if CONTIGUOUS_CHECKPOINTING:
        assert num_layers is not None, 'Must specify the number of layers with contiguous memory checkpointing'


def in_aml():
    return 'AZUREML_EXPERIMENT_ID' in os.environ


def in_aws_sm():
    return 'SM_TRAINING_ENV' in os.environ


def in_dlts():
    return 'DLTS_JOB_ID' in os.environ


def mpi_discovery(distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT, verbose=True):
    """
    Discovery MPI environment via mpi4py and map to relevant dist state
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    master_addr = None
    if rank == 0:
        hostname_cmd = ['hostname -I']
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_addr = result.decode('utf-8').split()[0]
    master_addr = comm.bcast(master_addr, root=0)
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([(i == proc_name) for i in all_procs[:rank]])
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(distributed_port)
    if verbose:
        utils.logger.info('Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}'.format(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))
    if cdb is not None and cdb.is_initialized():
        assert cdb.get_rank() == rank, 'MPI rank {} does not match torch rank {}'.format(rank, cdb.get_rank())
        assert cdb.get_world_size() == world_size, 'MPI world size {} does not match torch world size {}'.format(world_size, cdb.get_world_size())


DEFAULT_AML_MASTER_PORT = '54965'


DEFAULT_AML_NCCL_SOCKET_IFNAME = '^docker0,lo'


def patch_aml_env_for_torch_nccl_backend(master_port=6105, verbose=True):
    """Helper routine to get and set environment variables.
    This is adapted from Azure ML's documentation available from:
    https://azure.github.io/azureml-web/docs/cheatsheet/distributed-training/#environment-variables-from-openmpi
    """
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    single_node = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE']) == int(os.environ['WORLD_SIZE'])
    if not single_node:
        master_node_params = os.environ['AZ_BATCH_MASTER_NODE'].split(':')
        os.environ['MASTER_ADDR'] = master_node_params[0]
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = str(master_port)
    else:
        os.environ['MASTER_ADDR'] = os.environ['AZ_BATCHAI_MPI_MASTER_NODE']
        os.environ['MASTER_PORT'] = DEFAULT_AML_MASTER_PORT
    if verbose:
        utils.logger.info('NCCL_SOCKET_IFNAME original value = {}'.format(os.environ['NCCL_SOCKET_IFNAME']))
    os.environ['NCCL_SOCKET_IFNAME'] = DEFAULT_AML_NCCL_SOCKET_IFNAME
    os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    if verbose:
        utils.logger.info('Discovered AzureML settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}'.format(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))


def patch_aws_sm_env_for_torch_nccl_backend(verbose=True):
    """Helper routine to get and set environment variables when running inside an AWS SageMaker environment.
    """
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    if verbose:
        utils.logger.info('Discovered AWS SageMaker settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}'.format(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))


class GroupQuantizer:

    def __init__(self, q_int8=True, group_size=1, num_bits=8):
        self.group_size = group_size
        self.num_bits = num_bits
        self.q_int8 = q_int8
        self.num_groups = 32

    def quantize(self, inputs, qkv=True, count=1, parallel_dim=0):
        if not self.q_int8 or not qkv:
            inputs = torch.nn.Parameter(inputs, requires_grad=False)
            inputs.scale = torch.empty(1)
            return inputs
        q_range = 2 ** self.num_bits
        num_groups = inputs.shape[0] // self.group_size
        inputs = inputs
        input_flat = inputs.reshape(num_groups, -1).contiguous()
        input_min = torch.min(input_flat, dim=1, keepdim=True)[0].float()
        input_max = torch.max(input_flat, dim=1, keepdim=True)[0].float()
        scale = torch.max(input_min.abs(), input_max.abs()) * 2.0 / q_range
        input_flat = (input_flat / scale).round().clamp(-q_range // 2, q_range // 2 - 1)
        inputs_q = input_flat.reshape(inputs.shape).contiguous()
        out = torch.nn.Parameter(inputs_q, requires_grad=False)
        inputs_split = inputs.split(inputs.shape[parallel_dim] // 2, dim=parallel_dim)
        input_flat = [inputs_split[i].reshape(num_groups, -1).contiguous() for i in range(2)]
        input_min = [torch.min(input_flat[i], dim=1, keepdim=True)[0].float() for i in range(2)]
        input_max = [torch.max(input_flat[i], dim=1, keepdim=True)[0].float() for i in range(2)]
        scale1 = [(torch.max(input_min[i].abs(), input_max[i].abs()) * 2.0 / q_range).squeeze().unsqueeze(0) for i in range(2)]
        out.scale = torch.cat([scale.squeeze().unsqueeze(0), scale1[0], scale1[1]], dim=0).reshape(num_groups, -1).contiguous()
        return out


def get_transformer_name(replaced_module):
    from torch.nn import ModuleList
    transformer_name = ''
    for n, c in replaced_module.named_children():
        if c.__class__ in supported_models:
            transformer_name += n + '.'
            for name, child in c.named_children():
                if child.__class__ is ModuleList:
                    transformer_name += name
                    break
            break
    return transformer_name


class EmbeddingLayer(nn.Module):

    def __init__(self, weight_shape, dtype=torch.half):
        super(EmbeddingLayer, self).__init__()
        self.weight = Parameter(torch.empty(weight_shape[0], weight_shape[1], dtype=dtype, device=torch.cuda.current_device()))

    def forward(self, input):
        return F.embedding(input, self.weight)


class OPTEmbedding(EmbeddingLayer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, weight_shape):
        self.offset = 2
        super().__init__(weight_shape)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        positions = positions[:, past_key_values_length:]
        return super().forward(positions + self.offset)


def load_model_with_checkpoint(r_module, sd, mp_replace, ckpt_type, weight_quantizer=None, rank=0, param_names=None, transformer_config=None, megatron_v2=False):
    error_msgs = []

    def transpose(data):
        with torch.no_grad():
            data = data.contiguous()
            data1 = data.transpose(-1, -2).reshape(-1)
            data.reshape(-1).copy_(data1)
            data1 = None
        return data.reshape(data.shape[-1], data.shape[-2])

    def load(module, prefix):
        args = sd[0], prefix, {}, True, [], [], error_msgs
        if hasattr(module, 'weight'):
            module.weight = mp_replace.copy(module.weight.data, sd[0][prefix + 'weight'])
        if prefix + 'bias' in sd[0].keys():
            module.bias = mp_replace.copy(module.bias.data, sd[0][prefix + 'bias'])
        args = None
        gc.collect()

    def load_transformer_layer(module, prefix):
        if ckpt_type == 'tp':

            def load_parameters(module, prefix):
                for n, p in module.named_parameters():
                    if prefix + n in sd[0] and len(n.split('.')) == 1:
                        if type(sd[0][prefix + n]) is list:
                            tmp_data, scale = sd[0][prefix + n]
                            tmp_data = tmp_data
                            scale = scale
                        else:
                            tmp_data = sd[0][prefix + n]
                            scale = None
                        src_shape = tmp_data.shape
                        dst_shape = p.shape
                        inner_dim = 1 if tmp_data.dtype == torch.int8 else 0
                        outer_dim = 0 if tmp_data.dtype == torch.int8 else 1
                        if len(src_shape) == 2 and len(dst_shape) == 2:
                            if src_shape[inner_dim] == dst_shape[0] and src_shape[outer_dim] == dst_shape[1]:
                                if tmp_data.dtype != torch.int8:
                                    p = weight_quantizer.quantize(transpose(tmp_data) if weight_quantizer.q_int8 else tmp_data)
                                else:
                                    p = torch.nn.parameter.Parameter(tmp_data, requires_grad=False)
                                    p.scale = scale
                                setattr(module, n, p)
                            else:
                                dim = inner_dim if src_shape[inner_dim] != dst_shape[0] else outer_dim
                                dim1 = 0 if src_shape[inner_dim] != dst_shape[0] else 1
                                if src_shape[dim] > dst_shape[dim1]:
                                    weight_partition = torch.split(tmp_data, dst_shape[dim1], dim=dim)[rank]
                                    assert tmp_data.dtype != torch.int8 or scale.numel() > weight_quantizer.num_groups * (rank + 1), 'ERROR: We require the quantization scales for larger TP-size when loading INT8 checkpoint!                                           Please use the FP16 checkpoint to generate INT8 checkpoint with the sharding parameters!'
                                    scale = scale.view(-1)[weight_quantizer.num_groups * (rank + 1):].reshape(weight_quantizer.num_groups, -1).contiguous()
                                else:
                                    assert tmp_data.dtype != torch.int8, 'Merging of the checkpoints are not supported when using INT8 checkpoint!                                           Please use a as many GPUs as TP-size for the checkpoint'
                                    all_data = [(sd[j][prefix + n] if type(sd[j][prefix + n]) is list else sd[j][prefix + n]) for j in range(len(sd))]
                                    weight_partition = torch.cat([(ad[0] if type(ad) is list else ad) for ad in all_data], dim=dim)
                                    if tmp_data.dtype == torch.int8:
                                        scale = torch.cat([ad[1] for ad in all_data], dim=dim)
                                if tmp_data.dtype != torch.int8:
                                    weight_partition = weight_quantizer.quantize(transpose(weight_partition), parallel_dim=0 if dim == 1 else 1) if weight_quantizer.q_int8 else weight_quantizer.quantize(weight_partition)
                                else:
                                    weight_partition = torch.nn.parameter.Parameter(weight_partition, requires_grad=False)
                                    weight_partition.scale = scale
                                setattr(module, n, weight_partition)
                        elif src_shape[0] == dst_shape[0]:
                            p.data.copy_(tmp_data)
                        elif src_shape[0] > dst_shape[0]:
                            bias_split = torch.split(tmp_data, dst_shape[-1])[rank].contiguous()
                            p.data.copy_(bias_split)
                        else:
                            p.data.copy_(torch.cat([sd[j][prefix + n] for j in range(len(sd))], dim=0).contiguous())
            load_parameters(module, prefix)
            for n, child in module.named_children():
                load_parameters(child, prefix + n + '.')
        else:

            def _transpose(x):
                heads = transformer_config.heads // mp_replace.mp_size
                attention_head_size = x.shape[-1] // heads
                new_x_shape = x.size()[:-1] + (heads, attention_head_size)
                x_1 = x.view(*new_x_shape)
                q, k, v = torch.split(x_1, x_1.shape[-1] // 3, dim=x_1.dim() - 1)
                if len(q.shape) > 2:
                    return torch.cat((q.reshape(q.shape[0], -1), k.reshape(q.shape[0], -1), v.reshape(q.shape[0], -1)), dim=-1).reshape(x.shape)
                else:
                    return torch.cat((q.reshape(-1), k.reshape(-1), v.reshape(-1)), dim=-1).reshape(x.shape)

            def maybe_copy(module, dst_name, src_name, qkv=False, megatron_v2=False, split_qkv=False):
                if src_name in sd[0]:
                    dst = getattr(module, dst_name)
                    tmp = sd[0][src_name]
                    if len(dst.shape) == 1:
                        if split_qkv:
                            dst = mp_replace.qkv_copy(dst, tmp)
                        else:
                            dst = mp_replace.copy(dst, tmp)
                        if qkv and megatron_v2:
                            dst = torch.nn.parameter.Parameter(_transpose(dst).contiguous())
                    else:
                        if split_qkv:
                            dst = weight_quantizer.quantize(mp_replace.qkv_copy(dst, tmp if weight_quantizer.q_int8 else transpose(tmp).contiguous()))
                        else:
                            dst = weight_quantizer.quantize(mp_replace.copy(dst, tmp if weight_quantizer.q_int8 else transpose(tmp)))
                        if qkv and megatron_v2:
                            scale1 = dst.scale
                            dst = torch.nn.parameter.Parameter(_transpose(dst).contiguous())
                            dst.scale = scale1
                    setattr(module, dst_name, dst)

            def maybe_copy_qkv(module, dst_name, src_names, split_qkv=False):
                if src_names[0] in sd[0]:
                    q = sd[0][src_names[0]]
                    k = sd[0][src_names[1]]
                    v = sd[0][src_names[2]]
                    qkv_data = torch.cat((q, k, v), dim=0)
                    dst = getattr(module, dst_name)
                    if len(dst.shape) == 1:
                        if split_qkv:
                            dst = mp_replace.qkv_copy(dst, qkv_data.contiguous())
                        else:
                            dst = mp_replace.copy(dst, qkv_data)
                    elif split_qkv:
                        dst = weight_quantizer.quantize(mp_replace.qkv_copy(dst, qkv_data if weight_quantizer.q_int8 else transpose(qkv_data).contiguous()))
                    else:
                        dst = weight_quantizer.quantize(mp_replace.copy(dst, qkv_data if weight_quantizer.q_int8 else transpose(qkv_data)))
                    setattr(module, dst_name, dst)
            if len(param_names) == 14:
                qkv_w, qkv_b, attn_ow, attn_ob, mlp_intw, mlp_intb, mlp_ow, mlp_ob, inp_normw, inp_normb, attn_nw, attn_nb, _, split_qkv = param_names
            elif len(param_names) < 14:
                q_w, k_w, v_w, attn_ow, mlp_intw, mlp_intb, mlp_ow, mlp_ob, inp_normw, inp_normb, _, split_qkv = param_names
            else:
                q_w, q_b, k_w, k_b, v_w, v_b, attn_ow, attn_ob, mlp_intw, mlp_intb, mlp_ow, mlp_ob, inp_normw, inp_normb, attn_nw, attn_nb, _, split_qkv = param_names
            maybe_copy(module, 'norm_w', prefix + inp_normw)
            maybe_copy(module, 'norm_b', prefix + inp_normb)
            if len(param_names) == 14:
                maybe_copy(module.attention, 'attn_qkvw', prefix + qkv_w, qkv=True, megatron_v2=megatron_v2, split_qkv=split_qkv)
                maybe_copy(module.attention, 'attn_qkvb', prefix + qkv_b, qkv=True, megatron_v2=megatron_v2, split_qkv=split_qkv)
            elif len(param_names) < 14:
                maybe_copy_qkv(module.attention, 'attn_qkvw', [prefix + q_w, prefix + k_w, prefix + v_w], split_qkv=split_qkv)
            else:
                maybe_copy_qkv(module.attention, 'attn_qkvw', [prefix + q_w, prefix + k_w, prefix + v_w], split_qkv=split_qkv)
                maybe_copy_qkv(module.attention, 'attn_qkvb', [prefix + q_b, prefix + k_b, prefix + v_b], split_qkv=split_qkv)
            maybe_copy(module.attention, 'attn_ow', prefix + attn_ow)
            if len(param_names) >= 14:
                maybe_copy(module.attention, 'attn_ob', prefix + attn_ob)
                maybe_copy(module.mlp, 'attn_nw', prefix + attn_nw)
                maybe_copy(module.mlp, 'attn_nb', prefix + attn_nb)
            maybe_copy(module.mlp, 'inter_w', prefix + mlp_intw)
            maybe_copy(module.mlp, 'inter_b', prefix + mlp_intb)
            maybe_copy(module.mlp, 'output_w', prefix + mlp_ow)
            maybe_copy(module.mlp, 'output_b', prefix + mlp_ob)
    try:
        OPTLearnedPositionalEmbedding = transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding
    except:
        OPTLearnedPositionalEmbedding = None
    layer_policies = {nn.Linear: load, nn.Embedding: load, nn.LayerNorm: load, EmbeddingLayer: load, LinearLayer: load, Normalize: load, transformer_inference.DeepSpeedTransformerInference: load_transformer_layer, OPTLearnedPositionalEmbedding: load, OPTEmbedding: load}
    all_ds_ids = {}

    def load_module_recursive(module, prefix='', level=0):
        for name, child in module.named_children():
            if child.__class__ in layer_policies:
                checking_key = prefix + name + '.'
                if not any(checking_key in item for item in sd[0].keys()):
                    if hasattr(child, 'weight') and (hasattr(child.weight, 'ds_id') and child.weight.ds_id in all_ds_ids):
                        prefix1 = all_ds_ids[child.weight.ds_id]
                        if child.__class__ is nn.Linear:
                            child = LinearLayer(weight=all_ds_ids[child.weight.ds_id])
                            setattr(module, name, child)
                    continue
                child_params = list(child.parameters())
                if len(child_params) > 0 and (child_params[0].numel() == 0 or child_params[0].is_meta):
                    if child.weight.is_meta:
                        ds_shape = child.weight.shape
                    else:
                        ds_shape = child.weight.ds_shape
                    if child.__class__ is nn.LayerNorm:
                        child = Normalize(dim=ds_shape[-1], dtype=child.weight.dtype, eps=child.eps)
                        setattr(module, name, child)
                    elif child.__class__ is nn.Linear:
                        child = LinearLayer(weight_shape=child.weight.shape, bias=child.bias)
                        setattr(module, name, child)
                    elif child.__class__ is OPTLearnedPositionalEmbedding:
                        child = OPTEmbedding(weight_shape=ds_shape)
                        setattr(module, name, child)
                    else:
                        ds_id = None
                        if hasattr(child.weight, 'ds_id'):
                            ds_id = child.weight.ds_id
                        child = EmbeddingLayer(weight_shape=ds_shape, dtype=child.weight.dtype)
                        if ds_id is not None:
                            all_ds_ids[ds_id] = child.weight
                        setattr(module, name, child)
                layer_policies[child.__class__](child, prefix + name + '.')
            else:
                load_module_recursive(child, prefix if (level == 0 and ckpt_type == 'pp') and param_names[-2] else prefix + name + '.', level + 1)
    load_module_recursive(r_module)
    embedding_weight = None
    for n, p in r_module.named_parameters():
        if 'word_embeddings.' in n or 'embed_tokens.' in n:
            embedding_weight = p
    if embedding_weight is not None:
        r_module.lm_head.weight = embedding_weight
    for sd_ in sd:
        del sd_
    sd = None
    gc.collect()


def _replace_module(model, policies, layer_id=0):
    """ Traverse model's children recursively and apply any transformations in ``policies``.
    Arguments:
        model (torch.nn.Module): model to augment
        policies (dict): Mapping of source class to replacement function.
    Returns:
        Modified ``model``.
    """
    for name, child in model.named_children():
        if child.__class__ in policies:
            replaced_module = policies[child.__class__][0](child, policies[child.__class__][-1], layer_id)
            setattr(model, name, replaced_module)
            if isinstance(model, PipelineModule):
                assert hasattr(model, 'forward_funcs'), 'we require pipe-module to have the list of fwd_functions'
                model.forward_funcs[model.fwd_map[name]] = replaced_module
            layer_id += 1
        else:
            _, layer_id = _replace_module(child, policies, layer_id=layer_id)
    return model, layer_id


def replace_module(model, orig_class, replace_fn, _replace_policy):
    """ Scan the model for instances of ``orig_clas:`` to replace using ``replace_fn``.
    Arguments:
        model (torch.nn.Module): the model to augment
        orig_class (torch.nn.Module): the module to search for
        replace_fn (method): a method to convert instances of ``orig_class`` to the
                             desired type and return a new instance.
    Returns:
        A modified ``model``.
    """
    policy = {}
    if orig_class is not None:
        policy.update({orig_class: (replace_fn, _replace_policy)})
    else:
        for plcy in replace_policies:
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                for orig_layer_class in plcy._orig_layer_class:
                    policy.update({orig_layer_class: (replace_fn, plcy)})
            elif plcy._orig_layer_class is not None:
                policy.update({plcy._orig_layer_class: (replace_fn, plcy)})
    assert len(policy.items()) > 0, 'No default policy found! Please specify your policy injection_policy (like {BertLayer:HFBEertLayerPolicy}).' + 'You can find some samples here: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py'
    replaced_module, _ = _replace_module(model, policy)
    return replaced_module


def replace_transformer_layer(orig_layer_impl, model, checkpoint_dict, config, model_config):
    """ Replace bert-style transformer layers with DeepSpeed's transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation to look for,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        checkpoint_dict: Dictionary for checkpoint passed from the Inference Engine
        config: top-level DS Inference config defined in inference/config.py
        model_config: HuggingFace model config passed from the inference/engine.py
    Returns:
        Updated nn.module with replaced transformer layers
    """
    fp16 = config.dtype == torch.float16 or config.dtype == torch.int8
    quantize = config.dtype == torch.int8
    linear_layer_setting = None
    """
        linear_layer_setting (tuple of modules) [Optional]: shows which two classes are used for linear layers and embedding layers
    """
    micro_batch_size = -1
    seed = -1
    local_rank = -1
    mp_replace = ReplaceWithTensorSlicing(mp_group=config.tensor_parallel.tp_group, mp_size=config.tensor_parallel.tp_size)

    def replace_with_policy(child, policy_cls, triangular_masking, inference=False, layer_id=0):
        policy = policy_cls(child, inference=inference)
        global selected_policy_g
        if selected_policy_g is None:
            selected_policy_g = policy
        if not policy.cuda_graph_supported:
            assert not config.enable_cuda_graph, 'cuda graph is not supported with this model, please disable'
        if inference:
            hidden_size, num_attention_heads = policy.get_hidden_heads()
            assert num_attention_heads % config.tensor_parallel.tp_size == 0, 'To run the model parallel across the GPUs, the attention_heads require to be divisible by the world_size!' + 'This is because the attention computation is partitioned evenly among the parallel GPUs.'
        moe = False
        if hasattr(child, 'mlp') and isinstance(child.mlp, MoE):
            num_experts = child.mlp.num_experts
            moe = True
        attn_linear_layer, qkvw, qkvb, dense_w, dense_b, scale_attention, megatron_v2 = policy.attention()
        global megatron_v2_g
        megatron_v2_g = megatron_v2
        if not moe or config.moe.type == 'standard':
            mlp_linear_layer, _h4h_w, _h4h_b, _4hh_w, _4hh_b = policy.mlp()
        else:
            mlp_linear_layer, _h4h_w, _h4h_b, _4hh_w, _4hh_b, _res_h4h_w, _res_h4h_b, _res_4hh_w, _res_4hh_b, _res_coef = policy.mlp(config.moe.type)
        attn_nw, attn_nb, input_nw, input_nb = policy.layerNorm()
        if False:
            if policy_cls is not HFBertLayerPolicy:
                qkvw = qkvw
            dense_w = dense_w
            _h4h_w = [moe_w1 for moe_w1 in _h4h_w] if moe else _h4h_w
            _4hh_w = [moe_w1 for moe_w1 in _4hh_w] if moe else _4hh_w
        elif fp16:
            qkvw = qkvw.half()
            dense_w = dense_w.half()
            _h4h_w = [moe_w1.half() for moe_w1 in _h4h_w] if moe else _h4h_w.half()
            _4hh_w = [moe_w1.half() for moe_w1 in _4hh_w] if moe else _4hh_w.half()
        if quantize or fp16:
            qkvb = qkvb if qkvb is None else qkvb.half()
            dense_b = dense_b if dense_b is None else dense_b.half()
            _h4h_b = [moe_b1.half() for moe_b1 in _h4h_b] if moe else _h4h_b.half()
            _4hh_b = [moe_b1.half() for moe_b1 in _4hh_b] if moe else _4hh_b.half()
            attn_nw = attn_nw if attn_nw is None else attn_nw.half()
            attn_nb = attn_nb if attn_nb is None else attn_nb.half()
            input_nw = input_nw.half()
            input_nb = input_nb.half()
        if config.moe.enabled and config.moe.type == 'residual' and fp16:
            _res_h4h_b = _res_h4h_b.half()
            _res_4hh_b = _res_4hh_b.half()
            _res_h4h_w = _res_h4h_w.half()
            _res_4hh_w = _res_4hh_w.half()
            _res_coef = _res_coef.half()
        quantizer = GroupQuantizer(q_int8=quantize)
        if inference:
            scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx if hasattr(config, 'scale_attn_by_inverse_layer_idx') else False
            if moe:
                ep_world_size = dist.get_world_size()
                local_ep_size = 1 if num_experts < ep_world_size else num_experts // ep_world_size
                bigscience_bloom = policy_cls is BLOOMLayerPolicy
                transformer_config = transformer_inference.DeepSpeedMoEInferenceConfig(hidden_size=hidden_size, heads=num_attention_heads, layer_norm_eps=config.layer_norm_eps if hasattr(config, 'layer_norm_eps') else 1e-12, fp16=fp16, pre_layer_norm=policy.pre_attn_norm, mp_size=config.tensor_parallel.tp_size, q_int8=quantize, moe_experts=local_ep_size, global_experts=num_experts, mlp_type=config.moe.type, scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx)
            else:
                rotary_dim = model_config.rotary_dim if hasattr(model_config, 'rotary_dim') else child.attention.rotary_ndims if hasattr(child, 'attention') and hasattr(child.attention, 'rotary_ndims') else -1
                bigscience_bloom = policy_cls is BLOOMLayerPolicy
                transformer_config = transformer_inference.DeepSpeedInferenceConfig(hidden_size=hidden_size, heads=num_attention_heads, layer_norm_eps=model_config.layer_norm_eps if hasattr(model_config, 'layer_norm_eps') else model_config.layer_norm_epsilon if hasattr(model_config, 'layer_norm_epsilon') else model_config.layernorm_epsilon if hasattr(model_config, 'layernorm_epsilon') else 1e-12, fp16=fp16, pre_layer_norm=policy.pre_attn_norm, mp_size=config.tensor_parallel.tp_size, q_int8=quantize, return_tuple=config.return_tuple or policy_cls is HFBertLayerPolicy, triangular_masking=policy_cls is not HFBertLayerPolicy, local_attention=model_config.attention_layers[layer_id] == 'local' if hasattr(model_config, 'attention_layers') else False, window_size=model_config.window_size if hasattr(model_config, 'window_size') else 1, rotary_dim=rotary_dim, mlp_after_attn=rotary_dim is None or rotary_dim < 0, mlp_act_func_type=policy.mlp_act_func_type, training_mp_size=config.training_mp_size, bigscience_bloom=bigscience_bloom, max_out_tokens=config.max_out_tokens, scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx)
                global transformer_config_g
                transformer_config_g = transformer_config
            if moe:
                new_module = transformer_inference.DeepSpeedMoEInference(transformer_config, mp_group=config.tensor_parallel.tp_group, ep_group=None if config.moe.ep_group is None else config.moe.ep_group[num_experts], expert_mp_group=None if config.moe.ep_mp_group is None else config.moe.ep_mp_group[num_experts])
            else:
                new_module = transformer_inference.DeepSpeedTransformerInference(transformer_config, mp_group=config.tensor_parallel.tp_group)
            new_module.config.scale_attention = scale_attention

            def transpose(data):
                data = data.contiguous()
                data.reshape(-1).copy_(data.transpose(-1, -2).contiguous().reshape(-1))
                data = data.reshape(data.shape[-1], data.shape[-2])
                data
                return data
            attn_block = new_module.attention
            mpl_block = new_module.mlp
            if attn_linear_layer:
                if qkvw.is_meta:
                    pass
                else:
                    qkvw.data = transpose(qkvw.data)
                    dense_w.data = transpose(dense_w.data)

            def _transpose(x):
                attention_head_size = x.shape[-1] // transformer_config.heads
                new_x_shape = x.size()[:-1] + (transformer_config.heads, attention_head_size)
                x_1 = x.view(*new_x_shape)
                q, k, v = torch.split(x_1, x_1.shape[-1] // 3, dim=x_1.dim() - 1)
                if len(q.shape) > 2:
                    return torch.cat((q.reshape(q.shape[0], -1), k.reshape(q.shape[0], -1), v.reshape(q.shape[0], -1)), dim=-1).reshape(x.shape)
                else:
                    return torch.cat((q.reshape(-1), k.reshape(-1), v.reshape(-1)), dim=-1).reshape(x.shape)
            if megatron_v2:
                new_module.config.rotate_half = True
                new_module.config.rotate_every_two = False
                qkvw = torch.nn.parameter.Parameter(_transpose(qkvw).contiguous())
                qkvb = torch.nn.parameter.Parameter(_transpose(qkvb).contiguous())
            if mlp_linear_layer:
                if not moe and _4hh_w.is_meta:
                    pass
                else:
                    _h4h_w = [transpose(moe_w1.data) for moe_w1 in _h4h_w] if moe else transpose(_h4h_w.data)
                    _4hh_w = [transpose(moe_w1.data) for moe_w1 in _4hh_w] if moe else transpose(_4hh_w.data)
            if moe and config.moe.type == 'residual':
                _res_h4h_w.data = transpose(_res_h4h_w.data)
                _res_4hh_w.data = transpose(_res_4hh_w.data)
                _res_coef.data = transpose(_res_coef.data)
            if qkvw.is_meta:
                if qkvb is None:
                    attn_block.attn_qkvb = None
                if dense_b is None:
                    attn_block.attn_ob = None
                pass
            else:
                attn_block.attn_qkvw = quantizer.quantize(mp_replace.copy(attn_block.attn_qkvw, qkvw) if bigscience_bloom else mp_replace.qkv_copy(attn_block.attn_qkvw, qkvw))
                attn_block.attn_qkvb = mp_replace.copy(attn_block.attn_qkvb, qkvb) if bigscience_bloom else mp_replace.qkv_copy(attn_block.attn_qkvb, qkvb)
                attn_block.attn_ow = quantizer.quantize(mp_replace.copy(attn_block.attn_ow, dense_w))
                attn_block.attn_ob = mp_replace.copy(attn_block.attn_ob, dense_b)
            if moe:
                gpu_index = dist.get_rank()
                gpu_index = 0
                for ep_index in range(local_ep_size):
                    mpl_block[ep_index].inter_w.data = _h4h_w[gpu_index * local_ep_size + ep_index]
                    mpl_block[ep_index].inter_b.data = _h4h_b[gpu_index * local_ep_size + ep_index]
                    mpl_block[ep_index].output_w.data = _4hh_w[gpu_index * local_ep_size + ep_index]
                    mpl_block[ep_index].output_b.data = _4hh_b[gpu_index * local_ep_size + ep_index]
                new_module.attn_nw.data = attn_nw
                new_module.attn_nb.data = attn_nb
                if config.moe.type == 'residual':
                    new_module.res_mlp.inter_w.data = _res_h4h_w
                    new_module.res_mlp.inter_b.data = _res_h4h_b
                    new_module.res_mlp.output_w.data = _res_4hh_w
                    new_module.res_mlp.output_b.data = _res_4hh_b
                    new_module.res_coef.data = _res_coef
            else:
                if _4hh_w.is_meta:
                    pass
                else:
                    mpl_block.inter_w = quantizer.quantize(mp_replace.copy(mpl_block.inter_w, _h4h_w))
                    mpl_block.inter_b = mp_replace.copy(mpl_block.inter_b, _h4h_b)
                    mpl_block.output_w = quantizer.quantize(mp_replace.copy(mpl_block.output_w, _4hh_w))
                    mpl_block.output_b = mp_replace.copy(mpl_block.output_b, _4hh_b)
                if attn_nw is None:
                    new_module.mlp.attn_nw = attn_nw
                    new_module.mlp.attn_nb = attn_nb
                elif attn_nw.is_meta:
                    pass
                else:
                    new_module.mlp.attn_nw.data.copy_(attn_nw)
                    new_module.mlp.attn_nb.data.copy_(attn_nb)
            if input_nw.is_meta:
                pass
            else:
                new_module.norm_w.data.copy_(input_nw)
                new_module.norm_b.data.copy_(input_nb)
        else:
            transformer_config = deepspeed.DeepSpeedTransformerConfig(batch_size=micro_batch_size if micro_batch_size > 0 else 1, hidden_size=config.hidden_size, heads=config.num_attention_heads, attn_dropout_ratio=config.attention_probs_dropout_prob, hidden_dropout_ratio=config.hidden_dropout_prob, num_hidden_layers=config.num_hidden_layers, initializer_range=config.initializer_range, layer_norm_eps=config.layer_norm_eps if hasattr(config, 'layer_norm_eps') else 1e-12, seed=seed, fp16=fp16, pre_layer_norm=policy.pre_attn_norm, return_tuple=config.return_tuple, local_rank=local_rank, stochastic_mode=True, normalize_invertible=True, training=True)
            new_module = deepspeed.DeepSpeedTransformerLayer(transformer_config)
            new_module.attn_qkvw.data = qkvw
            new_module.attn_qkvb.data = qkvb
            new_module.attn_ow.data = dense_w
            new_module.attn_ob.data = dense_b
            new_module.attn_nw.data = attn_nw
            new_module.attn_nb.data = attn_nb
            new_module.norm_w.data = input_nw
            new_module.norm_b.data = input_nb
            new_module.inter_w.data = _h4h_w
            new_module.inter_b.data = _h4h_b
            new_module.output_w.data = _4hh_w
            new_module.output_b.data = _4hh_b
        return new_module

    def replace_wo_policy(module, all_reduce_linears):
        mp_size = config.tensor_parallel.tp_size
        mp_group = config.tensor_parallel.tp_group

        def _replace(child, name, conv_linear_layer):
            mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group)
            weight_shape = child.weight.shape
            if name in all_reduce_linears:
                new_weight = torch.empty((weight_shape[1] if conv_linear_layer else weight_shape[0], (weight_shape[0] if conv_linear_layer else weight_shape[1]) // mp_size), device=child.weight.device, dtype=child.weight.dtype)
                if conv_linear_layer:
                    child.weight.data = child.weight.data.transpose(-1, -2).contiguous()
                data = mp_replace.copy(new_weight, child.weight.data)
                new_bias = torch.empty(weight_shape[0], device=child.weight.device, dtype=child.weight.dtype)
                if child.bias is not None:
                    new_bias.data.copy_(child.bias.data)
                return LinearAllreduce(data, child.bias if child.bias is None else torch.nn.parameter.Parameter(new_bias), mp_group)
            else:
                new_weight = torch.empty(((weight_shape[1] if conv_linear_layer else weight_shape[0]) // mp_size, weight_shape[0] // mp_size if conv_linear_layer else weight_shape[1]), device=child.weight.device, dtype=child.weight.dtype)
                if conv_linear_layer:
                    child.weight.data = child.weight.data.transpose(-1, -2).contiguous()
                data = mp_replace.copy(new_weight, child.weight.data)
                new_bias = torch.empty(weight_shape[0] // mp_size, device=child.weight.device, dtype=child.weight.dtype)
                bias_data = None if child.bias is None else mp_replace.copy(new_bias, child.bias.data)
                return LinearLayer(weight=data, bias=bias_data)

        def _slice_embedding(child, name, conv_linear_layer):
            mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group)
            new_weight = torch.empty((child.weight.shape[0], child.weight.shape[1] // mp_size), device=child.weight.device, dtype=child.weight.dtype)
            data = mp_replace.copy(new_weight, child.weight.ds_tensor.data if hasattr(child.weight, 'ds_tensor') else child.weight.data)
            new_embedding = nn.Embedding(child.weight.shape[0], child.weight.shape[1] // mp_size)
            new_embedding.weight.data.copy_(data)
            return new_embedding

        def update_mp_params(child):
            if hasattr(child, 'n_heads'):
                child.n_heads = child.n_heads // mp_size
            if hasattr(child, 'inner_dim'):
                child.inner_dim = child.inner_dim // mp_size
            if hasattr(child, 'num_heads'):
                child.num_heads = child.num_heads // mp_size
            if hasattr(child, 'num_attention_heads'):
                child.num_attention_heads = child.num_attention_heads // mp_size
            if hasattr(child, 'all_head_size'):
                child.all_head_size = child.all_head_size // mp_size
            if hasattr(child, 'embed_dim'):
                child.embed_dim = child.embed_dim // mp_size
            if hasattr(child, 'hidden_size'):
                child.hidden_size = child.hidden_size // mp_size
        conv_linear_layer = False
        if linear_layer_setting is not None:
            linear_policies = {linear_layer_setting[0]: _replace}
            if len(linear_layer_setting) == 2:
                linear_policies.update({linear_layer_setting[1]: _slice_embedding})
        elif orig_layer_impl is HFGPT2LayerPolicy._orig_layer_class:
            try:
                conv_linear_layer = True
                linear_policies = {transformers.model_utils.Conv1D: _replace}
            except ImportError:
                linear_policies = {nn.Linear: _replace}
        else:
            linear_policies = {nn.Linear: _replace, nn.Embedding: _slice_embedding}

        def _replace_module(r_module, prev_name=''):
            for name, child in r_module.named_children():
                if child.__class__ in linear_policies:
                    setattr(r_module, name, linear_policies[child.__class__](child, prev_name + '.' + name, conv_linear_layer))
                else:
                    update_mp_params(child)
                    _replace_module(child, name)
            return r_module
        return _replace_module(module)

    def replace_fn(child, _policy, layer_id=0):
        training = False
        if training:
            new_module = replace_with_policy(child, _policy, config.triangular_masking)
        elif config.replace_with_kernel_inject:
            new_module = replace_with_policy(child, _policy, config.triangular_masking, inference=True, layer_id=layer_id)
        else:
            new_module = replace_wo_policy(child, _policy)
        return new_module
    replaced_module = replace_module(model=model, orig_class=orig_layer_impl, replace_fn=replace_fn, _replace_policy=config.injection_policy_tuple)
    quantizer = GroupQuantizer(q_int8=quantize)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if checkpoint_dict is not None:
        start_time = time.time()
        checkpoint = checkpoint_dict['checkpoints']
        ckpt_list = checkpoint['tp'] if type(checkpoint) is dict else checkpoint
        ckpt_type = checkpoint_dict.get('parallelization', 'pp')
        ckpt_mp_size = checkpoint_dict.get('tp_size', len(ckpt_list))
        ckpt_mp_size = checkpoint_dict.get('mp_size', ckpt_mp_size)
        base_dir1 = checkpoint_dict.get('base_dir', config.base_dir)
        if ckpt_type == 'pp' and type(checkpoint) is list:
            pbar = tqdm.tqdm(total=len(checkpoint), desc=f'Loading {len(checkpoint)} checkpoint shards')
            for i in range(len(checkpoint)):
                sd = [torch.load(os.path.join(base_dir1, checkpoint[i]), map_location='cpu')]
                load_model_with_checkpoint(replaced_module, sd, mp_replace, ckpt_type, quantizer, param_names=selected_policy_g.get_param_names(), transformer_config=transformer_config_g, megatron_v2=megatron_v2_g)
                pbar.update(1)
        else:
            num_checkpoints = len(ckpt_list) // ckpt_mp_size
            tp_split_size = world_size / ckpt_mp_size
            sd_offset = int(rank / tp_split_size)
            sd_count = int((rank + max(1, tp_split_size)) / tp_split_size) - sd_offset
            pbar = tqdm.tqdm(total=num_checkpoints, desc=f'Loading {num_checkpoints} checkpoint shards')
            for i in range(num_checkpoints):
                pbar.update(1)
                ckpt_index = i * ckpt_mp_size + sd_offset
                ckpt_files = [(os.path.join(base_dir1, ckpt_list[ckpt_index + j]) if base_dir1 else ckpt_list[ckpt_index + j]) for j in range(sd_count)]
                sds = [torch.load(ckpt_file, map_location='cpu') for ckpt_file in ckpt_files]
                load_model_with_checkpoint(replaced_module, sds, mp_replace, ckpt_type, quantizer, int(rank % tp_split_size), param_names=selected_policy_g.get_param_names(), transformer_config=transformer_config_g, megatron_v2=megatron_v2_g)
                sds = [None for _ in sds]
                gc.collect()
            if 'non_tp' in checkpoint:
                pbar = tqdm.tqdm(total=len(checkpoint['non_tp']), desc=f"Loading {len(checkpoint['non_tp'])} checkpoint shards")
                for i in range(len(checkpoint['non_tp'])):
                    pbar.update(1)
                    ckpt_file = os.path.join(base_dir1, checkpoint['non_tp'][i]) if base_dir1 else checkpoint['non_tp'][i]
                    sds = [torch.load(ckpt_file, map_location='cpu')]
                    load_model_with_checkpoint(replaced_module, sds, mp_replace, ckpt_type, quantizer, int(rank % tp_split_size), param_names=selected_policy_g.get_param_names(), transformer_config=transformer_config_g, megatron_v2=megatron_v2_g)
                    sds = [None for _ in sds]
                    gc.collect()
        None
    if config.save_mp_checkpoint_path is not None:
        from collections import OrderedDict
        num_partitions = 8
        if checkpoint_dict is None:
            ckpt_name = 'ds_model'
            try:
                if isinstance(model, BloomForCausalLM):
                    ckpt_name = 'bloom'
            except ImportError:
                ckpt_name = 'ds_model'
        else:
            ckpt_name = checkpoint_dict['type']
        if dist.is_initialized():
            dist.barrier()
        transformer_name = get_transformer_name(replaced_module)
        non_tp_ckpt_name = f'non-tp.pt'
        ckpt_files = [non_tp_ckpt_name]
        os.makedirs(config.save_mp_checkpoint_path, exist_ok=True)
        if not dist.is_initialized() or dist.get_rank() == 0:
            None
            torch.save(OrderedDict({k: v for k, v in dict(replaced_module.state_dict()).items() if transformer_name not in k}), f'{config.save_mp_checkpoint_path}/{non_tp_ckpt_name}')
            ckpt_config = json.dumps({'type': ckpt_name, 'base_dir': f'{config.save_mp_checkpoint_path}', 'checkpoints': {'non_tp': ckpt_files, 'tp': [f'tp_{r:0>2d}_{m:0>2d}.pt' for m in range(num_partitions) for r in range(world_size)]}, 'version': 1.0, 'parallelization': 'tp', 'tp_size': world_size, 'dtype': 'int8' if quantize else 'float16' if fp16 else 'float32'})
            with open(f'{config.save_mp_checkpoint_path}/ds_inference_config.json', 'w') as cfg:
                cfg.write(ckpt_config)
        rep_sd = replaced_module.state_dict()
        for n, p in replaced_module.named_parameters():
            if hasattr(p, 'scale'):
                rep_sd[n] = [p, p.scale]
        keys = list(rep_sd.keys())
        partition_size = len(keys) // num_partitions + 1
        for m in range(num_partitions):
            torch.save(OrderedDict({k: ([rep_sd[k], rep_sd[k].scale] if hasattr(rep_sd[k], 'scale') else rep_sd[k]) for k in keys[m * partition_size:(m + 1) * partition_size] if transformer_name in k}), f'{config.save_mp_checkpoint_path}/tp_{rank:0>2d}_{m:0>2d}.pt')
    return replaced_module


class InferenceEngine(Module):
    inference_mp_group = None
    inference_ep_group = None
    expert_mp_group = None

    def __init__(self, model, config):
        """
        Args:
            model: torch.nn.Module
            config: DeepSpeedInferenceConfig
        """
        global DS_INFERENCE_ENABLED
        DS_INFERENCE_ENABLED = True
        super().__init__()
        self.module = model
        self._config = config
        self._get_model_config_generate(config)
        if hasattr(self.module, 'generate'):
            self.generate = self._generate
        if hasattr(self.module, 'config'):
            DSPolicy.hf_model_config = self.module.config
        self.injection_dict = config.injection_policy
        self.mp_group = config.tensor_parallel.tp_group
        self.mpu = config.tensor_parallel.mpu
        self.quantize_merge_count = 1
        self.quantization_scales = None
        self.ep_group = None
        self.expert_mp_group = None
        self.cuda_graph_created = False
        self.checkpoint_engine = TorchCheckpointEngine()
        quantization_setting = None
        self._init_quantization_setting(quantization_setting)
        self.model_profile_enabled = False
        self._model_times = []
        self.remove_mask_prepare_for_bloom()
        if config.enable_cuda_graph:
            assert pkg_version.parse(torch.__version__) >= pkg_version.parse('1.10'), 'If you want to use cuda graph, please upgrade torch to at least v1.10'
        if config.checkpoint and not config.replace_with_kernel_inject:
            self._load_checkpoint(config.checkpoint)
        if config.dtype:
            self._convert_to_dtype(config)
        if self.mpu:
            config.tensor_parallel.tp_size = dist.get_world_size(group=self.mpu.get_model_parallel_group())
            self.mp_group = self.mpu.get_model_parallel_group()
        elif config.tensor_parallel.tp_size > 1:
            self._create_model_parallel_group(config)
            config.tensor_parallel.tp_group = self.mp_group
        if isinstance(self.module, torch.nn.Module):
            moe, _ = has_moe_layers(self.module)
        else:
            moe = False
        if moe and dist.get_world_size() > 1:
            self._create_ep_parallel_group(config.moe.moe_experts)
        if not config.replace_with_kernel_inject:
            config.checkpoint = None
        if self.injection_dict:
            for client_module, injection_policy in self.injection_dict.items():
                if isinstance(injection_policy, str):
                    config.injection_policy_tuple = injection_policy,
                else:
                    config.injection_policy_tuple = injection_policy
                self._apply_injection_policy(config, client_module)
        elif config.replace_method == 'auto':
            self._apply_injection_policy(config)
        device = torch.cuda.current_device()
        self.module
        if config.tensor_parallel.tp_size > 1:
            _rng_state = torch.cuda.get_rng_state()
            dist.broadcast(_rng_state, 0)
            torch.set_rng_state(_rng_state.cpu())
        if config.tensor_parallel.tp_size > 1:
            assert not config.enable_cuda_graph, 'Cuda graph is not supported for model parallelism'

    def profile_model_time(self, use_cuda_events=True):
        if not self.model_profile_enabled and not self._config.enable_cuda_graph:
            self.module.register_forward_pre_hook(self._pre_forward_hook)
            self.module.register_forward_hook(self._post_forward_hook)
        self.model_profile_enabled = True
        self.use_cuda_events = use_cuda_events
        if self.use_cuda_events:
            self.timers = SynchronizedWallClockTimer()

    def _get_model_config_generate(self, config):
        self.config = getattr(self.module, 'config', None) if config.config is None else config.config

    def remove_mask_prepare_for_bloom(self):
        if hasattr(self.module, 'transformer'):
            if hasattr(self.module.transformer, '_prepare_attn_mask'):
                self.module.transformer._prepare_attn_mask = lambda attention_mask, *args, **kwargs: attention_mask

    def _pre_forward_hook(self, module, *inputs, **kwargs):
        if self.use_cuda_events:
            self.timers(INFERENCE_MODEL_TIMER).start()
        else:
            torch.cuda.synchronize()
            self._start = time.time()

    def _post_forward_hook(self, module, input, output):
        if self.use_cuda_events:
            self.timers(INFERENCE_MODEL_TIMER).stop()
            elapsed_time = self.timers(INFERENCE_MODEL_TIMER).elapsed(reset=True)
        else:
            torch.cuda.synchronize()
            self._end = time.time()
            elapsed_time = self._end - self._start
        self._model_times.append(elapsed_time)

    def _create_model_parallel_group(self, config):
        if InferenceEngine.inference_mp_group is None:
            init_distributed()
            local_rank = int(os.getenv('LOCAL_RANK', '0'))
            torch.cuda.set_device(local_rank)
            ranks = [i for i in range(config.tensor_parallel.tp_size)]
            self.mp_group = dist.new_group(ranks)
            InferenceEngine.inference_mp_group = self.mp_group
        else:
            self.mp_group = InferenceEngine.inference_mp_group

    def _create_ep_parallel_group(self, moe_experts):
        self.ep_group = {}
        self.expert_mp_group = {}
        moe_experts = moe_experts if type(moe_experts) is list else [moe_experts]
        for e in moe_experts:
            self.ep_group.update({e: None})
            self.expert_mp_group.update({e: None})
        for moe_ep_size in self.ep_group.keys():
            num_ep_groups = dist.get_world_size() // moe_ep_size
            for i in range(num_ep_groups):
                ep_cnt = i * moe_ep_size
                size = dist.get_world_size() if moe_ep_size > dist.get_world_size() else moe_ep_size
                ranks = list(range(ep_cnt, ep_cnt + size))
                _ep_group = dist.new_group(ranks)
                if dist.get_rank() in ranks:
                    self.ep_group.update({moe_ep_size: _ep_group})
            if dist.get_world_size() > moe_ep_size:
                num_expert_mp_groups = dist.get_world_size() // num_ep_groups
                expert_mp_size = dist.get_world_size() // moe_ep_size
                for i in range(num_expert_mp_groups):
                    expert_mp_comm_ranks = [(i + nr * moe_ep_size) for nr in range(expert_mp_size)]
                    _expert_mp_group = dist.new_group(expert_mp_comm_ranks)
                    if dist.get_rank() in expert_mp_comm_ranks:
                        self.expert_mp_group.update({moe_ep_size: _expert_mp_group})

    def _init_quantization_setting(self, quantization_setting):
        self.quantize_bits = 8
        self.mlp_extra_grouping = False
        self.quantize_groups = 1
        if type(quantization_setting) is tuple:
            self.mlp_extra_grouping, self.quantize_groups = quantization_setting
        elif quantization_setting is not None:
            self.quantize_groups = quantization_setting
        log_dist(f'quantize_bits = {self.quantize_bits} mlp_extra_grouping = {self.mlp_extra_grouping}, quantize_groups = {self.quantize_groups}', [0])

    def _validate_args(self, mpu, replace_with_kernel_inject):
        if replace_with_kernel_inject and not isinstance(self.module, Module):
            raise ValueError(f'model must be a torch.nn.Module, got {type(self.module)}')
        if not isinstance(self._config.tensor_parallel.tp_size, int) or self._config.tensor_parallel.tp_size < 1:
            raise ValueError(f'mp_size must be an int >= 1, got {self._config.tensor_parallel.tp_size}')
        if mpu:
            methods = ['get_model_parallel_group', 'get_data_parallel_group']
            for method in methods:
                if not hasattr(mpu, method):
                    raise ValueError(f'mpu is missing {method}')
        if self._config.checkpoint is not None and not isinstance(self._config.checkpoint, (str, dict)):
            raise ValueError(f'checkpoint must be None, str or dict, got {type(self._config.checkpoint)}')
        supported_dtypes = [None, torch.half, torch.int8, torch.float]
        if self._config.dtype not in supported_dtypes:
            raise ValueError(f'{self._config.dtype} not supported, valid dtype: {supported_dtypes}')
        if self.injection_dict is not None and not isinstance(self.injection_dict, dict):
            raise ValueError(f'injection_dict must be None or a dict, got: {self.injection_dict}')

    def load_model_with_checkpoint(self, r_module):
        self.mp_replace = ReplaceWithTensorSlicing(mp_group=self.mp_group, mp_size=self._config.tensor_parallel.tp_size)
        error_msgs = []

        def load(module, state_dict, prefix):
            args = state_dict, prefix, {}, True, [], [], error_msgs
            if hasattr(module, 'weight'):
                if 'query_key_value' in prefix:
                    module.weight = self.mp_replace.qkv_copy(module.weight.data, state_dict[prefix + 'weight'])
                else:
                    module.weight = self.mp_replace.copy(module.weight.data, state_dict[prefix + 'weight'])
            else:
                module.norm.weight = self.mp_replace.copy(module.norm.weight.data, state_dict[prefix + 'weight'])
            if prefix + 'bias' in self.key_list:
                if hasattr(module, 'norm'):
                    module.norm.bias = self.mp_replace.copy(module.norm.bias, state_dict[prefix + 'bias'])
                else:
                    data = state_dict[prefix + 'bias']
                    data = data
                    module.bias = self.mp_replace.copy(module.bias, data)
        layer_policies = {nn.Linear: load, nn.Embedding: load, nn.LayerNorm: load, LinearLayer: load, LinearAllreduce: load}

        def load_module_recursive(module, prefix='', level=0):
            for name, child in module.named_children():
                if child.__class__ in layer_policies:
                    checking_key = prefix + name + '.'
                    if not any(checking_key in item for item in self.key_list):
                        continue
                    if len(list(child.parameters())) > 0 and list(child.parameters())[0].numel() == 0:
                        if len(child.weight.ds_shape) == 1:
                            child = Normalize(dim=child.weight.ds_shape[-1], dtype=child.weight.dtype, eps=child.eps)
                            setattr(module, name, child)
                    load(child, self.sd, prefix + name + '.')
                else:
                    load_module_recursive(child, prefix if level == 0 else prefix + name + '.', level + 1)
        load_module_recursive(r_module)

    def _apply_injection_policy(self, config, client_module=None):
        checkpoint_dir = config.checkpoint
        checkpoint = SDLoaderFactory.get_sd_loader_json(checkpoint_dir, self.checkpoint_engine) if checkpoint_dir is not None else None
        generic_injection(self.module, fp16=config.dtype == torch.half or config.dtype == torch.int8, enable_cuda_graph=config.enable_cuda_graph)
        if isinstance(self.module, torch.nn.Module):
            replace_transformer_layer(client_module, self.module, checkpoint, config, self.config)

    def _get_all_ckpt_names(self, checkpoints_path, tag):
        ckpt_file_pattern = self._get_ckpt_name(checkpoints_path, tag, mp_placeholder='*')
        ckpt_files = glob.glob(ckpt_file_pattern)
        ckpt_files.sort()
        return ckpt_files

    def _get_ckpt_name(self, checkpoints_path, tag, mp_placeholder=None):
        if mp_placeholder is not None:
            mp_rank_str = mp_placeholder
        else:
            mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
            mp_rank_str = '{:02d}'.format(mp_rank)
        ckpt_name = os.path.join(checkpoints_path, 'mp_rank_' + mp_rank_str + '_model_states.pt')
        return ckpt_name

    def _load_checkpoint(self, load_dir, load_module_strict=True, tag=None):
        is_pipe_parallel = isinstance(self.module, PipelineModule)
        if is_pipe_parallel:
            raise RuntimeError('pipeline parallelism is currently not supported in inference.')
        if not isinstance(load_dir, dict) and os.path.isdir(load_dir):
            if tag is None:
                latest_path = os.path.join(load_dir, 'latest')
                if os.path.isfile(latest_path):
                    with open(latest_path, 'r') as fd:
                        tag = fd.read().strip()
            ckpt_list = self._get_all_ckpt_names(load_dir, tag)
            sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list, self.checkpoint_engine)
        else:
            sd_loader = SDLoaderFactory.get_sd_loader_json(load_dir, self.checkpoint_engine)
        if type(sd_loader) is list:
            self.sd = torch.load(sd_loader[0], map_location='cpu')
            self.key_list = list(self.sd.keys())
            self.load_model_with_checkpoint(self.module)
            for i in range(1, len(sd_loader)):
                if not dist.is_initialized() or dist.get_rank() == 0:
                    None
                self.sd = torch.load(sd_loader[i], map_location='cuda')
                self.key_list = list(self.sd.keys())
                self.load_model_with_checkpoint(self.module)
        else:
            mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
            load_path, checkpoint, quantize_config = sd_loader.load(self._config.tensor_parallel.tp_size, mp_rank, is_pipe_parallel=is_pipe_parallel, quantize=self._config.dtype is torch.int8, quantize_groups=self.quantize_groups, mlp_extra_grouping=self.mlp_extra_grouping)
            self.quantization_scales, self.quantize_merge_count = quantize_config
            moe, _ = has_moe_layers(self.module)
            if moe:
                old_moe_load = False
                if not isinstance(checkpoint['num_experts'], list):
                    old_moe_load = True
                DeepSpeedEngine.load_moe_state_dict(load_dir, tag, state_dict=checkpoint[self._choose_module_key(checkpoint)], old_moe_load=old_moe_load, model=self.module, mpu=self.mpu, checkpoint_engine=self.checkpoint_engine)
            self.module.load_state_dict(state_dict=checkpoint[self._choose_module_key(checkpoint)], strict=load_module_strict)

    def _choose_module_key(self, sd):
        assert not ('module' in sd and 'model' in sd), "checkpoint has both 'model' and 'module' keys, not sure how to proceed"
        assert 'module' in sd or 'model' in sd, "checkpoint contains neither 'model' or 'module' keys, not sure how to proceed"
        if 'module' in sd:
            return 'module'
        elif 'model' in sd:
            return 'model'

    def _convert_to_dtype(self, config):
        if not isinstance(self.module, torch.nn.Module):
            return
        if False:
            quantizer = WeightQuantization(mlp_extra_grouping=self.mlp_extra_grouping)
            model, self.quantization_scales = quantizer.model_quantize(self.module, self.injection_dict, self.quantize_bits, self.quantize_groups)
        elif config.dtype == torch.half:
            self.module.half()
        elif config.dtype == torch.bfloat16:
            self.module.bfloat16()
        elif config.dtype == torch.float:
            self.module.float()

    def _create_cuda_graph(self, *inputs, **kwargs):
        cuda_stream = torch.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self.module(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)
        self._cuda_graphs = torch.cuda.CUDAGraph()
        self.static_inputs = inputs
        self.static_kwargs = kwargs
        with torch.cuda.graph(self._cuda_graphs):
            self.static_output = self.module(*self.static_inputs, **self.static_kwargs)
        self.cuda_graph_created = True

    def _graph_replay(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[k].copy_(kwargs[k])
        self._cuda_graphs.replay()
        return self.static_output

    def model_times(self):
        assert self.model_profile_enabled, 'model profiling is not enabled'
        model_times = self._model_times
        if self._config.enable_cuda_graph and len(self._model_times) == 0:
            raise ValueError(f'Model times are empty and cuda graph is enabled. If this is a GPT-style model this combo is not supported. If this is a BERT-style model this is a bug, please report it. Model type is: {type(self.module)}')
        self._model_times = []
        return model_times

    def forward(self, *inputs, **kwargs):
        """Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        start = None
        if self.model_profile_enabled and self._config.enable_cuda_graph:
            torch.cuda.synchronize()
            start = time.time()
        if self._config.enable_cuda_graph:
            if self.cuda_graph_created:
                outputs = self._graph_replay(*inputs, **kwargs)
            else:
                self._create_cuda_graph(*inputs, **kwargs)
                outputs = self._graph_replay(*inputs, **kwargs)
        else:
            outputs = self.module(*inputs, **kwargs)
        if self.model_profile_enabled and self._config.enable_cuda_graph:
            torch.cuda.synchronize()
            duration = time.time() - start
            self._model_times.append(duration)
        return outputs

    def _generate(self, *inputs, **kwargs):
        num_beams = 1
        if 'generation_config' in kwargs:
            gen_config = kwargs['generation_config']
            num_beams = getattr(gen_config, 'num_beams', 1)
        if 'num_beams' in kwargs:
            num_beams = kwargs['num_beams']
        if num_beams > 1:
            raise NotImplementedError('DeepSpeed does not support `num_beams` > 1, if this is important to you please add your request to: https://github.com/microsoft/DeepSpeed/issues/2506')
        return self.module.generate(*inputs, **kwargs)


class DSUNet(torch.nn.Module):

    def __init__(self, unet, enable_cuda_graph=True):
        super().__init__()
        self.unet = unet
        self.in_channels = unet.in_channels
        self.device = self.unet.device
        self.dtype = self.unet.dtype
        self.config = self.unet.config
        self.fwd_count = 0
        self.unet.requires_grad_(requires_grad=False)
        self.unet
        self.cuda_graph_created = False
        self.enable_cuda_graph = enable_cuda_graph

    def _graph_replay(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[k].copy_(kwargs[k])
        self._cuda_graphs.replay()
        return self.static_output

    def forward(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self.cuda_graph_created:
                outputs = self._graph_replay(*inputs, **kwargs)
            else:
                self._create_cuda_graph(*inputs, **kwargs)
                outputs = self._graph_replay(*inputs, **kwargs)
            return outputs
        else:
            return self._forward(*inputs, **kwargs)

    def _create_cuda_graph(self, *inputs, **kwargs):
        cuda_stream = torch.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._forward(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)
        self._cuda_graphs = torch.cuda.CUDAGraph()
        self.static_inputs = inputs
        self.static_kwargs = kwargs
        with torch.cuda.graph(self._cuda_graphs):
            self.static_output = self._forward(*self.static_inputs, **self.static_kwargs)
        self.cuda_graph_created = True

    def _forward(self, sample, timestamp, encoder_hidden_states, return_dict=True):
        return self.unet(sample, timestamp, encoder_hidden_states, return_dict)


class DSVAE(torch.nn.Module):

    def __init__(self, vae, enable_cuda_graph=True):
        super().__init__()
        self.vae = vae
        self.device = self.vae.device
        self.dtype = self.vae.dtype
        self.vae.requires_grad_(requires_grad=False)
        self.decoder_cuda_graph_created = False
        self.encoder_cuda_graph_created = False
        self.all_cuda_graph_created = False
        self.enable_cuda_graph = enable_cuda_graph

    def _graph_replay_decoder(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_decoder_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_decoder_kwargs[k].copy_(kwargs[k])
        self._decoder_cuda_graph.replay()
        return self.static_decoder_output

    def _decode(self, x, return_dict=True):
        return self.vae.decode(x, return_dict=return_dict)

    def _create_cuda_graph_decoder(self, *inputs, **kwargs):
        cuda_stream = torch.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._decode(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)
        self._decoder_cuda_graph = torch.cuda.CUDAGraph()
        self.static_decoder_inputs = inputs
        self.static_decoder_kwargs = kwargs
        with torch.cuda.graph(self._decoder_cuda_graph):
            self.static_decoder_output = self._decode(*self.static_decoder_inputs, **self.static_decoder_kwargs)
        self.decoder_cuda_graph_created = True

    def decode(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self.decoder_cuda_graph_created:
                outputs = self._graph_replay_decoder(*inputs, **kwargs)
            else:
                self._create_cuda_graph_decoder(*inputs, **kwargs)
                outputs = self._graph_replay_decoder(*inputs, **kwargs)
            return outputs
        else:
            return self._decode(*inputs, **kwargs)

    def _graph_replay_encoder(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_encoder_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_encoder_kwargs[k].copy_(kwargs[k])
        self._encoder_cuda_graph.replay()
        return self.static_encoder_output

    def _encode(self, x, return_dict=True):
        return self.vae.encode(x, return_dict=return_dict)

    def _create_cuda_graph_encoder(self, *inputs, **kwargs):
        cuda_stream = torch.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._encode(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)
        self._encoder_cuda_graph = torch.cuda.CUDAGraph()
        self.static_encoder_inputs = inputs
        self.static_encoder_kwargs = kwargs
        with torch.cuda.graph(self._encoder_cuda_graph):
            self.static_encoder_output = self._encode(*self.static_encoder_inputs, **self.static_encoder_kwargs)
        self.encoder_cuda_graph_created = True

    def encode(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self.encoder_cuda_graph_created:
                outputs = self._graph_replay_encoder(*inputs, **kwargs)
            else:
                self._create_cuda_graph_encoder(*inputs, **kwargs)
                outputs = self._graph_replay_encoder(*inputs, **kwargs)
            return outputs
        else:
            return self._encode(*inputs, **kwargs)

    def _graph_replay_all(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[k].copy_(kwargs[k])
        self._all_cuda_graph.replay()
        return self.static_output

    def forward(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self.cuda_graph_created:
                outputs = self._graph_replay_all(*inputs, **kwargs)
            else:
                self._create_cuda_graph(*inputs, **kwargs)
                outputs = self._graph_replay_all(*inputs, **kwargs)
            return outputs
        else:
            return self._forward(*inputs, **kwargs)

    def _create_cuda_graph(self, *inputs, **kwargs):
        cuda_stream = torch.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._forward(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)
        self._all_cuda_graph = torch.cuda.CUDAGraph()
        self.static_inputs = inputs
        self.static_kwargs = kwargs
        with torch.cuda.graph(self._all_cuda_graph):
            self.static_output = self._forward(*self.static_inputs, **self.static_kwargs)
        self.all_cuda_graph_created = True

    def _forward(self, sample, timestamp, encoder_hidden_states, return_dict=True):
        return self.vae(sample, timestamp, encoder_hidden_states, return_dict)


class DSClipEncoder(torch.nn.Module):

    def __init__(self, enc, enable_cuda_graph=False):
        super().__init__()
        enc.text_model._build_causal_attention_mask = self._build_causal_attention_mask
        self.enc = enc
        self.device = self.enc.device
        self.dtype = self.enc.dtype
        self.cuda_graph_created = [False, False]
        self.static_inputs = [None, None]
        self.static_kwargs = [None, None]
        self.static_output = [None, None]
        self._cuda_graphs = [None, None]
        self.iter = 0
        self.enable_cuda_graph = enable_cuda_graph
        self.config = self.enc.config

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype, device=torch.cuda.current_device())
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        return mask

    def _graph_replay(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[self.iter][i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[self.iter][k].copy_(kwargs[k])
        self._cuda_graphs[self.iter].replay()
        return self.static_output[self.iter]

    def forward(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self.cuda_graph_created[self.iter]:
                outputs = self._graph_replay(*inputs, **kwargs)
            else:
                self._create_cuda_graph(*inputs, **kwargs)
                outputs = self._graph_replay(*inputs, **kwargs)
            self.iter = (self.iter + 1) % 2
            return outputs
        else:
            return self.enc(*inputs, **kwargs)

    def _create_cuda_graph(self, *inputs, **kwargs):
        cuda_stream = torch.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._forward(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)
        self._cuda_graphs[self.iter] = torch.cuda.CUDAGraph()
        self.static_inputs[self.iter] = inputs
        self.static_kwargs[self.iter] = kwargs
        with torch.cuda.graph(self._cuda_graphs[self.iter]):
            self.static_output[self.iter] = self._forward(*self.static_inputs[self.iter], **self.static_kwargs[self.iter])
        self.cuda_graph_created[self.iter] = True

    def _forward(self, *inputs, **kwargs):
        return self.enc(*inputs, **kwargs)


class MoETypeEnum(str, Enum):
    residual = 'residual'
    standard = 'standard'


class DtypeEnum(Enum):
    fp16 = torch.float16, 'torch.float16', 'fp16', 'float16', 'half'
    bf16 = torch.bfloat16, 'torch.bfloat16', 'bf16', 'bfloat16'
    fp32 = torch.float32, 'torch.float32', 'fp32', 'float32', 'float'
    int8 = torch.int8, 'torch.int8', 'int8'

    def __new__(cls, *values):
        obj = object.__new__(cls)
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self):
        return '<%s.%s: %s>' % (self.__class__.__name__, self._name_, ', '.join([repr(v) for v in self._all_values]))


class QuantTypeEnum(str, Enum):
    asym = 'asymmetric'
    sym = 'symmetric'


minus_inf = -10000.0


class DeepSpeedSelfAttention(nn.Module):
    num_layers = 0

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1, qkv_merging=False):
        super(DeepSpeedSelfAttention, self).__init__()
        self.config = config
        data_type = torch.int8 if config.q_int8 else torch.half if config.fp16 else torch.float
        data_type_fp = torch.half if config.fp16 else torch.float
        self.config.layer_id = DeepSpeedSelfAttention.num_layers
        DeepSpeedSelfAttention.num_layers = DeepSpeedSelfAttention.num_layers + 1
        device = torch.cuda.current_device()
        qkv_size_per_partition = self.config.hidden_size // self.config.mp_size * 3
        self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size, qkv_size_per_partition, dtype=data_type, device=device), requires_grad=False)
        self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition, dtype=data_type_fp, device=device), requires_grad=False)
        out_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition, self.config.hidden_size, dtype=data_type, device=device), requires_grad=False)
        self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device), requires_grad=False)
        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads
        self.mp_group = mp_group
        self.q_scales = q_scales
        self.q_groups = q_groups
        self.merge_count = int(math.log2(merge_count))
        self.norm_factor = math.sqrt(math.sqrt(self.config.hidden_size // self.config.heads))
        self.qkv_merging = qkv_merging
        if self.config.scale_attn_by_inverse_layer_idx is True:
            self.norm_factor *= math.sqrt(self.config.layer_id + 1)
        self.qkv_func = QKVGemmOp(config)
        self.score_context_func = SoftmaxContextOp(config)
        self.linear_func = LinearOp(config)
        self.vector_matmul_func = VectorMatMulOp(config)

    def compute_attention(self, qkv_out, input_mask, layer_past, alibi):
        if isinstance(qkv_out, list):
            qkv_out = qkv_out[0]
        no_masking = input_mask is None
        if no_masking:
            input_mask = torch.empty(1)
        if alibi is not None:
            batch_heads = qkv_out.shape[0] * self.num_attention_heads_per_partition
            offset = dist.get_rank() * batch_heads if dist.is_initialized() else 0
            sliced_alibi = alibi[offset:batch_heads + offset, :, :]
        else:
            sliced_alibi = torch.empty(1)
        attn_key_value = self.score_context_func(query_key_value=qkv_out, attn_mask=(1 - input_mask) * minus_inf if input_mask.dtype == torch.int64 else input_mask, heads=self.num_attention_heads_per_partition, norm_factor=1 / self.norm_factor if self.config.scale_attention else 1.0, no_masking=no_masking, layer_id=self.config.layer_id, num_layers=DeepSpeedSelfAttention.num_layers, alibi=sliced_alibi)
        context_layer, key_layer, value_layer = attn_key_value
        return context_layer, key_layer, value_layer

    def forward(self, input, input_mask, head_mask=None, layer_past=None, get_present=False, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False, norm_w=None, norm_b=None, alibi=None):
        if not self.config.pre_layer_norm:
            qkv_out = self.linear_func(input=input, weight=self.attn_qkvw, bias=self.attn_qkvb, add_bias=self.attn_qkvb is not None, do_flash_attn=False, num_heads=self.num_attention_heads_per_partition)
        else:
            qkv_out = self.qkv_func(input=input, weight=self.attn_qkvw, bias=self.attn_qkvb if self.attn_qkvb is not None else norm_b, gamma=norm_w, beta=norm_b, add_bias=self.attn_qkvb is not None, num_layers=DeepSpeedSelfAttention.num_layers)
        context_layer, key_layer, value_layer = self.compute_attention(qkv_out=qkv_out, input_mask=input_mask, layer_past=layer_past, alibi=alibi)
        output = self.vector_matmul_func(input=context_layer, weight=self.attn_ow)
        inp_norm = qkv_out[-1]
        if self.config.mlp_after_attn and self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
            dist.all_reduce(output, group=self.mp_group)
        return output, key_layer, value_layer, context_layer, inp_norm


class BloomSelfAttention(DeepSpeedSelfAttention):

    def __init__(self, *args, **kwargs):
        super(BloomSelfAttention, self).__init__(*args, **kwargs)
        self.softmax_func = SoftmaxOp(self.config)

    def _transpose_for_context(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_layer_shape = x.size()[:-2] + (self.hidden_size_per_partition,)
        return x.view(*new_x_layer_shape).contiguous()

    def _split_tensor_along_last_dim(self, tensor, num_partitions, contiguous_split_chunks=True):
        """Split a tensor along its last dimension.

        Args:
            tensor: ([`torch.tensor`], *required*):
                input tensor to split
            num_partitions ([`int`], *required*):
                number of partitions to split the tensor
            contiguous_split_chunks ([`bool`], *optional*, default=`False`)::
                If True, make each chunk contiguous in memory.
        """
        last_dim = tensor.dim() - 1
        numerator, denominator = tensor.size()[last_dim], num_partitions
        if not numerator % denominator == 0:
            raise ValueError(f'{numerator} is not divisible by {denominator}')
        last_dim_size = numerator // denominator
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)
        return tensor_list

    def compute_attention(self, qkv_out, input_mask, layer_past, alibi):
        if isinstance(qkv_out, list):
            qkv_out = qkv_out[0]
        no_masking = input_mask is None
        if no_masking:
            input_mask = torch.empty(1)
        mixed_x_layer = qkv_out
        alibi = alibi
        head_dim = self.hidden_size_per_partition // self.num_attention_heads_per_partition
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * head_dim)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        query_layer, key_layer, value_layer = self._split_tensor_along_last_dim(mixed_x_layer, 3)
        output_size = query_layer.size(0), query_layer.size(2), query_layer.size(1), key_layer.size(1)
        query_layer = query_layer.transpose(1, 2).reshape(output_size[0] * output_size[1], output_size[2], -1)
        key_layer = key_layer.transpose(1, 2).reshape(output_size[0] * output_size[1], output_size[3], -1).transpose(-1, -2)
        value_layer = value_layer.transpose(1, 2).reshape(output_size[0] * output_size[1], output_size[3], -1)
        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=-1)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=-2)
        presents = key_layer, value_layer
        matmul_result = torch.matmul(query_layer, key_layer)
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], -1)
        offset = dist.get_rank() * self.num_attention_heads_per_partition if dist.is_initialized() else 0
        attention_probs = self.softmax_func(attn_scores=attention_scores, attn_mask=(1 - input_mask).half() * minus_inf if input_mask.dtype == torch.int64 else input_mask, alibi=alibi, triangular=self.config.triangular_masking and attention_scores.shape[-2] > 1, recompute=False, local_attention=False, window_size=1, async_op=False, layer_scale=1 / (self.norm_factor * self.norm_factor), head_offset=offset)
        attention_probs_reshaped = attention_probs.view(*matmul_result.shape)
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)
        context_layer = context_layer.view(context_layer.size(0) // self.num_attention_heads_per_partition, self.num_attention_heads_per_partition, context_layer.size(1), context_layer.shape[-1])
        context_layer = self._transpose_for_context(context_layer)
        key_layer = presents[0]
        value_layer = presents[1]
        return context_layer, key_layer, value_layer


class DeepSpeedMLPFunction(Function):

    @staticmethod
    def forward(ctx, input, inter_w, inter_b, config, output_b, output_w, q_scales, q_groups, merge_count, mp_group, async_op):
        if config.q_int8:
            intermediate = inference_cuda_module.fused_gemm_gelu_int8(input, inter_w, inter_b, config.epsilon, q_scales[2], q_groups * 2 ** merge_count, config.pre_layer_norm)
            output = inference_cuda_module.vector_matmul_int8(intermediate, output_w, q_scales[3], q_groups, merge_count)
        else:
            mlp_gemm_func = inference_cuda_module.fused_gemm_gelu_fp16 if config.fp16 else inference_cuda_module.fused_gemm_gelu_fp32
            output = mlp_gemm_func(input, inter_w, inter_b, output_w, config.epsilon, config.pre_layer_norm, async_op)
        if mp_group is not None and dist.get_world_size(group=mp_group) > 1:
            dist.all_reduce(output, group=mp_group, async_op=async_op)
        return output + output_b

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('You are running with DeepSpeed Inference mode.                             Please switch to Training mode for running backward!')


class DeepSpeedMLP(nn.Module):

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1, mlp_extra_grouping=False):
        super(DeepSpeedMLP, self).__init__()
        self.config = config
        data_type = torch.int8 if config.q_int8 else torch.half if config.fp16 else torch.float
        data_type_fp = torch.half if config.fp16 else torch.float
        device = torch.cuda.current_device()
        self.attn_nw = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device), requires_grad=False)
        self.attn_nb = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device), requires_grad=False)
        intm_size_per_partition = self.config.intermediate_size // self.config.mp_size
        self.inter_w = nn.Parameter(torch.empty(self.config.hidden_size, intm_size_per_partition, dtype=data_type, device=device), requires_grad=False)
        self.inter_b = nn.Parameter(torch.empty(intm_size_per_partition, dtype=data_type_fp, device=device), requires_grad=False)
        self.output_w = nn.Parameter(torch.empty(intm_size_per_partition, self.config.hidden_size, dtype=data_type, device=device), requires_grad=False)
        self.output_b = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device), requires_grad=False)
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))
        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()
        self.mp_group = mp_group
        self.mlp_gemm_func = inference_cuda_module.mlp_gemm_fp16 if config.fp16 else inference_cuda_module.mlp_gemm_fp32
        self.vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if config.fp16 else inference_cuda_module.vector_matmul_fp32
        self.fused_gemm_gelu = inference_cuda_module.fused_gemm_gelu_fp16 if config.fp16 else inference_cuda_module.fused_gemm_gelu_fp32
        self.bias_residual_func = inference_cuda_module.bias_residual_fp16 if config.fp16 or config.q_int8 else inference_cuda_module.bias_residual_fp32
        self.residual_add_func = inference_cuda_module.residual_add_bias_fp16 if config.fp16 or config.q_int8 else inference_cuda_module.residual_add_bias_fp32

    def forward(self, input, residual, residual_norm, bias):
        return DeepSpeedMLPFunction.apply(input, residual, residual_norm, bias, self.inter_w, self.inter_b, self.attn_nw, self.attn_nb, self.config, self.mp_group, self.output_b, self.output_w, self.q_scales, self.q_groups, self.merge_count, self.mlp_gemm_func, self.fused_gemm_gelu, self.vector_matmul_func, self.bias_residual_func, self.residual_add_func)


class DeepSpeedTransformerInference(nn.Module):
    """Initialize the DeepSpeed Transformer Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepSpeedInferenceConfig
            mp_group: Model parallelism group initialized on the modeling side.
            quantize_scales: This argument groups all the layers' scales used for quantization
            quantize_groups: Number of groups used for quantizing the model
            merge_count: Shows the number of model-parallel checkpoints merged before running inference.
                We use this argument to control the quantization scale for the model parameters if a bigger
                quantize-grouping than 1 is used.
            mlp_extra_grouping: This flag is used to show a 2x higher number of groups used for the MLP part
                of a Transformer layer. We use this feature for quantization to reduce the convergence impact
                for specific downstream tasks.
    """
    layer_id = 0

    def __init__(self, config, mp_group=None, quantize_scales=None, quantize_groups=1, merge_count=1, mlp_extra_grouping=False, qkv_merging=False):
        super(DeepSpeedTransformerInference, self).__init__()
        self.config = config
        self.config.layer_id = DeepSpeedTransformerInference.layer_id
        DeepSpeedTransformerInference.layer_id += 1
        data_type = torch.half if config.fp16 else torch.float
        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()
        if DeepSpeedTransformerInference.layer_id == 1:
            log_dist(f'DeepSpeed-Inference config: {self.config.__dict__}', [0])
        if self.config.bigscience_bloom:
            self.attention = BloomSelfAttention(self.config, mp_group, quantize_scales, quantize_groups, merge_count, qkv_merging)
        else:
            self.attention = DeepSpeedSelfAttention(self.config, mp_group, quantize_scales, quantize_groups, merge_count, qkv_merging)
        self.mlp = DeepSpeedMLP(self.config, mp_group, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping)
        device = torch.cuda.current_device()
        self.norm_w = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type, device=device), requires_grad=False)
        self.norm_b = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type, device=device), requires_grad=False)
        self.layer_past = None
        self.allocate_workspace = inference_cuda_module.allocate_workspace_fp32 if not config.fp16 else inference_cuda_module.allocate_workspace_fp16

    def forward(self, input, input_mask=None, attention_mask=None, head_mask=None, layer_past=None, get_key_value=False, get_present=False, encoder_output=None, enc_dec_attn_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, alibi=None, output_attentions=False, layer_head_mask=None, past_key_value=None):
        if self.config.layer_id == 0:
            self.allocate_workspace(self.config.hidden_size, self.config.heads, input.size()[1], input.size()[0], DeepSpeedTransformerInference.layer_id, self.config.mp_size, self.config.bigscience_bloom, dist.get_rank() if dist.is_initialized() else 0, self.config.max_out_tokens)
        get_present = get_present or get_key_value or use_cache
        input_mask = input_mask if attention_mask is None else attention_mask
        if input.shape[1] > 1:
            self.layer_past = None
        layer_past = layer_past if layer_past is not None else self.layer_past
        head_mask = layer_head_mask if layer_head_mask is not None else head_mask
        attn_mask = None
        if isinstance(input, tuple):
            attn_mask = input[1]
            input = input[0]
        input_type = input.dtype
        if (self.config.fp16 or self.config.q_int8) and input.dtype == torch.float:
            input = input.half()
        with torch.no_grad():
            attention_output, key, value, context_outputtn_ctx, inp_norm = self.attention(input, input_mask, head_mask, layer_past, get_present, encoder_hidden_states, encoder_attention_mask, output_attentions, self.norm_w, self.norm_b, alibi)
            presents = key, value
            self.layer_past = presents if layer_past is None else None
            output = self.mlp(attention_output, input, inp_norm, self.attention.attn_ob)
            if not self.config.pre_layer_norm:
                output = inference_cuda_module.layer_norm(output, self.norm_w, self.norm_b, self.config.epsilon)
            output = output
        if get_present:
            output = output, presents
        if self.config.return_tuple:
            return output if type(output) is tuple else (output, attn_mask)
        else:
            return output


class SparsityConfig:
    """Abstract Configuration class to store `sparsity configuration of a self attention layer`.
    It contains shared property of different block-sparse sparsity patterns. However, each class needs to extend it based on required property and functionality.
    """

    def __init__(self, num_heads, block=16, different_layout_per_head=False):
        """Initialize the Sparsity Pattern Config.

        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability.
        """
        self.num_heads = num_heads
        self.block = block
        self.different_layout_per_head = different_layout_per_head
        self.num_layout_heads = num_heads if different_layout_per_head else 1

    def setup_layout(self, seq_len):
        """Create layout tensor for the given sequence length

        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) for sparsity layout of all head; initialized with zero
        """
        if seq_len % self.block != 0:
            raise ValueError(f'Sequence Length, {seq_len}, needs to be dividable by Block size {self.block}!')
        num_blocks = seq_len // self.block
        layout = torch.zeros((self.num_heads, num_blocks, num_blocks), dtype=torch.int64)
        return layout

    def check_and_propagate_first_head_layout(self, layout):
        """If all heads require same sparsity layout, it propagate first head layout to all heads

        Arguments:
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head; may not be completely set at this step

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head
        """
        if not self.different_layout_per_head:
            layout[1:self.num_heads, :, :] = layout[0, :, :]
        return layout


class FixedSparsityConfig(SparsityConfig):
    """Configuration class to store `Fixed` sparsity configuration.
    For more details about this sparsity config, please see `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509; this has been customized.
    This class extends parent class of `SparsityConfig` and customizes it for `Fixed` sparsity.
    """

    def __init__(self, num_heads, block=16, different_layout_per_head=False, num_local_blocks=4, num_global_blocks=1, attention='bidirectional', horizontal_global_attention=False, num_different_global_patterns=1):
        """Initialize `Fixed` Sparsity Pattern Config.

        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability.
             num_local_blocks: optional: an integer determining the number of blocks in local attention window.
             num_global_blocks: optional: an integer determining how many consecutive blocks in a local window is used as the representative of the window for global attention.
             attention: optional: a string determining attention type. Attention can be `unidirectional`, such as autoregressive models, in which tokens attend only to tokens appear before them in the context. Considering that, the upper triangular of attention matrix is empty as above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to any other tokens before or after them. Then, the upper triangular part of the attention matrix is mirror of the lower triangular in the above figure.
             horizontal_global_attention: optional: a boolean determining if blocks that are global representative of a local window, also attend to all other blocks. This is valid only if attention type is `bidirectional`. Looking at the attention matrix, that means global attention not only includes the vertical blocks, but also horizontal blocks.
             num_different_global_patterns: optional: an integer determining number of different global attentions layouts. While global attention can be fixed by which block/s are representative of any local window, since there are multi-heads, each head can use a different global representative. For example, with 4 blocks local window and global attention size of 1 block, we can have 4 different versions in which the first, Second, third, or forth block of each local window can be global representative of that window. This parameter determines how many of such patterns we want. Of course, there is a limitation based on num_local_blocks and num_global_blocks.
        """
        super().__init__(num_heads, block, different_layout_per_head)
        self.num_local_blocks = num_local_blocks
        if num_local_blocks % num_global_blocks != 0:
            raise ValueError(f'Number of blocks in a local window, {num_local_blocks}, must be dividable by number of global blocks, {num_global_blocks}!')
        self.num_global_blocks = num_global_blocks
        if attention != 'unidirectional' and attention != 'bidirectional':
            raise NotImplementedError('only "uni/bi-directional" attentions are supported for now!')
        self.attention = attention
        if attention != 'bidirectional' and horizontal_global_attention:
            raise ValueError('only "bi-directional" attentions can support horizontal global attention!')
        self.horizontal_global_attention = horizontal_global_attention
        if num_different_global_patterns > 1 and not different_layout_per_head:
            raise ValueError(f'Number of different layouts cannot be more than one when you have set a single layout for all heads! Set different_layout_per_head to True.')
        if num_different_global_patterns > num_local_blocks // num_global_blocks:
            raise ValueError(f'Number of layout versions (num_different_global_patterns), {num_different_global_patterns}, cannot be larger than number of local window blocks divided by number of global blocks, {num_local_blocks} / {num_global_blocks} = {num_local_blocks // num_global_blocks}!')
        self.num_different_global_patterns = num_different_global_patterns

    def set_local_layout(self, h, layout):
        """Sets local attention layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head; may not be completely set at this step

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head in which local layout is set
        """
        num_blocks = layout.shape[1]
        for i in range(0, num_blocks, self.num_local_blocks):
            end = min(i + self.num_local_blocks, num_blocks)
            for row in range(i, end):
                for col in range(i, row + 1 if self.attention == 'unidirectional' else end):
                    layout[h, row, col] = 1
        return layout

    def set_global_layout(self, h, layout):
        """Sets global attention layout used by the given head in the sparse attention.

        Currently we set global blocks starting from the last block of a local window to the first one. That means if a local window consists of 4 blocks and global attention size is one block, we use block #4 in each local window as global. If we have different layout per head, then other heads will get #3, #2, and #1. And if we have more heads (and different layout has set) than num of global attentions, multiple head may have same global attentions.
        Note) if horizontal_global_attention is set, global blocks will be set both horizontally and vertically.

        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head; may not be completely set at this step

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head in which global layout is set
        """
        num_blocks = layout.shape[1]
        first_global_block_idx = self.num_local_blocks - (1 + h % self.num_different_global_patterns) * self.num_global_blocks
        end = num_blocks - num_blocks % self.num_local_blocks
        for i in range(first_global_block_idx, end, self.num_local_blocks):
            first_row = 0 if self.attention == 'bidirectional' else i
            layout[h, first_row:, i:i + self.num_global_blocks] = 1
            if self.horizontal_global_attention:
                layout[h, i:i + self.num_global_blocks, :] = 1
        if end < num_blocks:
            start = min(end + first_global_block_idx, num_blocks - self.num_global_blocks)
            end = start + self.num_global_blocks
            first_row = 0 if self.attention == 'bidirectional' else start
            layout[h, first_row:, start:end] = 1
            if self.horizontal_global_attention:
                layout[h, start:end, :] = 1
        return layout

    def make_layout(self, seq_len):
        """Generates `Fixed` sparsity layout used by each head in the sparse attention.

        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing `Fixed` sparsity layout of all head
        """
        layout = self.setup_layout(seq_len)
        for h in range(0, self.num_layout_heads):
            layout = self.set_local_layout(h, layout)
            layout = self.set_global_layout(h, layout)
        layout = self.check_and_propagate_first_head_layout(layout)
        return layout


class SparseSelfAttention(nn.Module):
    """Implements an efficient Sparse Self Attention of Transformer layer based on `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509

    For more information please see, TODO DeepSpeed Sparse Transformer.

    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial.
    """

    def __init__(self, sparsity_config=SparsityConfig(num_heads=4), key_padding_mask_mode='add', attn_mask_mode='mul', max_seq_length=2048):
        """Initialize the sparse self attention layer.
        Arguments:
            sparsity_config: optional: this parameter determines sparsity pattern configuration; it is based on SparsityConfig class.
            key_padding_mask_mode: optional: a string determining if key padding mask needs to be added, `add`, or be multiplied, `mul`.
            attn_mask_mode: optional: a string determining if attention mask needs to be added, `add`, or be multiplied, `mul`.
            max_seq_length: optional: the maximum sequence length this sparse attention module will be applied to; it controls the size of the master_layout.
        """
        super().__init__()
        self.sparsity_config = sparsity_config
        master_layout = self.sparsity_config.make_layout(max_seq_length)
        self.register_buffer('master_layout', master_layout)
        self._need_layout_synchronization = True
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode
    ops = dict()

    def get_layout(self, L):
        if self._need_layout_synchronization and dist.is_initialized():
            dist.broadcast(self.master_layout, src=0)
            self._need_layout_synchronization = False
        if L % self.sparsity_config.block != 0:
            raise ValueError(f'Sequence Length, {L}, needs to be dividable by Block size {self.sparsity_config.block}!')
        num_blocks = L // self.sparsity_config.block
        return self.master_layout[..., :num_blocks, :num_blocks].cpu()

    def get_ops(self, H, L):
        if L not in SparseSelfAttention.ops:
            sparsity_layout = self.get_layout(L)
            sparse_dot_sdd_nt = MatMul(sparsity_layout, self.sparsity_config.block, 'sdd', trans_a=False, trans_b=True)
            sparse_dot_dsd_nn = MatMul(sparsity_layout, self.sparsity_config.block, 'dsd', trans_a=False, trans_b=False)
            sparse_softmax = Softmax(sparsity_layout, self.sparsity_config.block)
            SparseSelfAttention.ops[L] = sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax
        return SparseSelfAttention.ops[L]

    def transpose_key_for_scores(self, x, L):
        bsz, num_heads, seq_len, head_dim = x.size()
        if seq_len != L:
            return x.permute(0, 1, 3, 2)
        return x

    def transpose_mask_for_sparse(self, qtype, x, is_key_padding_mask=False):
        x = x.type(qtype)
        if is_key_padding_mask:
            xdim = x.dim()
            for d in range(xdim - 1, 0, -1):
                x = x.squeeze(dim=d)
            return x
        return x.squeeze()

    def forward(self, query, key, value, rpe=None, key_padding_mask=None, attn_mask=None):
        """Applies forward phase of sparse self attention

        Arguments:
            query: required: query tensor
            key: required: key tensor
            value: required: value tensor
            rpe: optional: a tensor same dimension as x that is used as relative position embedding
            key_padding_mask: optional: a mask tensor of size (BatchSize X SequenceLength)
            attn_mask: optional: a mask tensor of size (SequenceLength X SequenceLength); currently only 2D is supported
            key_padding_mask_mode: optional: a boolean determining if key_padding_mask needs to be added or multiplied
            attn_mask_mode: optional: a boolean determining if attn_mask needs to be added or multiplied

        Return:
             attn_output: a dense tensor containing attention context
        """
        assert query.dtype == torch.half, 'sparse attention only supports training in fp16 currently, please file a github issue if you need fp32 support'
        bsz, num_heads, tgt_len, head_dim = query.size()
        key = self.transpose_key_for_scores(key, tgt_len)
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')
        if key_padding_mask is not None:
            key_padding_mask = self.transpose_mask_for_sparse(query.dtype, key_padding_mask, is_key_padding_mask=True)
        if attn_mask is not None:
            attn_mask = self.transpose_mask_for_sparse(query.dtype, attn_mask)
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(num_heads, tgt_len)
        scaling = float(head_dim) ** -0.5
        attn_output_weights = sparse_dot_sdd_nt(query, key)
        attn_output_weights = sparse_softmax(attn_output_weights, scale=scaling, rpe=rpe, key_padding_mask=key_padding_mask, attn_mask=attn_mask, key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)
        attn_output = sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output


class BertSparseSelfAttention(nn.Module):
    """Implements Sparse Self Attention layer of Bert model based on https://github.com/microsoft/DeepSpeedExamples/blob/master/bing_bert/nvidia/modelingpreln.py#L373

    For more information please see, TODO DeepSpeed Sparse Transformer.

    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial.
    """

    def __init__(self, config, sparsity_config=FixedSparsityConfig(num_heads=4)):
        """Initialize the bert sparse self attention layer.

        Note) you can use any of the provided sparsity configs or simply add yours!

        Arguments:
            config: required: Bert model config
            sparsity_config: optional: this parameter determines sparsity pattern configuration; it is based on FixedSparsityConfig class.
        """
        super(BertSparseSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.sparse_self_attention = SparseSelfAttention(sparsity_config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """Applies forward phase of bert sparse self attention

        Arguments:
            hidden_states: required: hidden_states tensor of the bert model
            attn_mask: required: a mask tensor of size (SequenceLength X SequenceLength); currently only 2D is supported

        Return:
             context_layer: a dense tensor containing attention context
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        context_layer = self.sparse_self_attention(query_layer, key_layer, value_layer, key_padding_mask=attention_mask)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class DeepSpeedMoEMLP(nn.Module):

    def __init__(self, config, q_scales=None, q_groups=1, merge_count=1, mlp_extra_grouping=False, mp_group=None):
        super(DeepSpeedMoEMLP, self).__init__()
        self.config = config
        self.attn_nw = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.attn_nb = nn.Parameter(torch.Tensor(self.config.hidden_size))
        interm_size = self.config.intermediate_size // (1 if mp_group is None else dist.get_world_size(group=mp_group))
        self.inter_w = nn.Parameter(torch.Tensor(self.config.hidden_size, interm_size))
        self.inter_b = nn.Parameter(torch.Tensor(interm_size))
        self.output_w = nn.Parameter(torch.Tensor(interm_size, self.config.hidden_size))
        self.output_b = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))
        self.mp_group = mp_group

    def forward(self, input, async_op=False):
        return DeepSpeedMLPFunction.apply(input, self.inter_w, self.inter_b, self.config, self.output_b, self.output_w, self.q_scales, self.q_groups, self.merge_count, self.mp_group, async_op)


class DeepSpeedMoEInference(nn.Module):
    """Initialize the DeepSpeed MoE Transformer Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepSpeedInferenceConfig
            mp_group: Model parallelism group initialized on the modeling side.
            quantize_scales: This argument groups all the layers' scales used for quantization
            quantize_groups: Number of groups used for quantizing the model
            merge_count: Shows the number of model-parallel checkpoints merged before running inference.
                We use this argument to control the quantization scale for the model parameters if a bigger
                quantize-grouping than 1 is used.
            mlp_extra_grouping: This flag is used to show a 2x higher number of groups used for the MLP part
                of a Transformer layer. We use this feature for quantization to reduce the convergence impact
                for specific downstream tasks.
    """
    layer_id = 0

    def __init__(self, config, mp_group=None, ep_group=None, expert_mp_group=None, quantize_scales=None, quantize_groups=1, merge_count=1, mlp_extra_grouping=False, qkv_merging=False):
        super(DeepSpeedMoEInference, self).__init__()
        self.config = config
        self.config.layer_id = DeepSpeedMoEInference.layer_id
        global inference_cuda_module
        global specialized_mode
        if inference_cuda_module is None:
            specialized_mode = False
            if hasattr(op_builder, 'InferenceSpecializedBuilder'):
                builder = op_builder.InferenceSpecializedBuilder()
                if builder.is_compatible():
                    inference_cuda_module = builder.load()
                    specialized_mode = True
                else:
                    inference_cuda_module = op_builder.InferenceBuilder().load()
            else:
                inference_cuda_module = op_builder.InferenceBuilder().load()
        self.config.specialized_mode = specialized_mode
        DeepSpeedMoEInference.layer_id += 1
        self.attention = DeepSpeedSelfAttention(self.config, mp_group, quantize_scales, quantize_groups, merge_count, qkv_merging)
        self.attn_nw = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.attn_nb = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.norm_w = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.norm_b = nn.Parameter(torch.Tensor(self.config.hidden_size))
        if config.mlp_type == 'residual':
            self.res_mlp = DeepSpeedMoEMLP(config, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping, mp_group)
            self.res_coef = nn.Parameter(torch.Tensor(self.config.hidden_size, 2))
            self.coef_func = inference_cuda_module.softmax_fp16 if self.config.fp16 or self.config.q_int8 else inference_cuda_module.softmax_fp32
            self.vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if config.fp16 else inference_cuda_module.vector_matmul_fp32
        config.mp_size = 1
        self.mlp = nn.ModuleList(DeepSpeedMoEMLP(config, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping, expert_mp_group) for i in range(self.config.moe_experts))
        self.moe_gate = TopKGate(self.config.hidden_size, self.config.global_experts, self.config.k, self.config.capacity_factor, self.config.eval_capacity_factor, self.config.min_capacity, self.config.noisy_gate_policy, self.config.drop_tokens, self.config.use_rts)
        self.ep_group = ep_group
        self.mp_group = mp_group
        self.expert_mp_group = expert_mp_group
        None
        self.bias_residual_func = inference_cuda_module.bias_residual_fp16 if config.fp16 or config.q_int8 else inference_cuda_module.bias_residual_fp32
        self.ds_layernorm = inference_cuda_module.layer_norm_fp16 if self.config.fp16 or self.config.q_int8 else inference_cuda_module.layer_norm_fp32
        self.einsum_sec_sm_ecm = inference_cuda_module.einsum_sec_sm_ecm_fp16 if self.config.fp16 or self.config.q_int8 else inference_cuda_module.einsum_sec_sm_ecm_fp32

    def res_coef_func(self, inp, async_op):
        inp = self.vector_matmul_func(inp, self.res_coef, async_op)
        return self.coef_func(inp, torch.empty(1), False, False, False, 256, async_op)

    def moe_gate_einsum(self, attention_output):
        _, combined_weights, dispatch_mask, _ = self.moe_gate(attention_output.view(-1, self.config.hidden_size), None)
        dispatched_attention = self.einsum_sec_sm_ecm(dispatch_mask.type_as(attention_output), attention_output.view(-1, self.config.hidden_size))
        return dispatched_attention, combined_weights

    def expert_exec(self, dispatched_input):
        dispatched_input = dispatched_input.reshape(self.config.global_experts // self.config.moe_experts, self.config.moe_experts, -1, self.config.hidden_size)
        chunks = dispatched_input.chunk(self.config.moe_experts, dim=1)
        expert_outputs = torch.empty((self.config.moe_experts, chunks[0].shape[0]) + chunks[0].shape[2:], dtype=dispatched_input.dtype, device=dispatched_input.device)
        for chunk, expert in zip(chunks, range(len(self.mlp))):
            expert_outputs[expert] = self.mlp[expert](chunk.view(-1, dispatched_input.shape[-2], dispatched_input.shape[-1]))
        return expert_outputs

    def _alltoall(self, dispatched_attention):
        if dist.get_world_size(group=self.ep_group) > 1:
            dispatched_input = torch.empty_like(dispatched_attention)
            dist.all_to_all_single(dispatched_input, dispatched_attention, group=self.ep_group)
            return dispatched_input
        else:
            return dispatched_attention

    def scale_expert_output(self, attention_output, expert_output, combined_weights):
        combined_output = torch.matmul(combined_weights.type_as(attention_output).reshape(combined_weights.shape[0], -1), expert_output.reshape(-1, expert_output.shape[-1]))
        return combined_output.reshape(attention_output.shape)

    def forward(self, input, input_mask=None, attention_mask=None, head_mask=None, layer_past=None, get_key_value=False, get_present=False, encoder_output=None, enc_dec_attn_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        get_present = get_present or get_key_value or use_cache
        input_mask = input_mask if attention_mask is None else attention_mask
        input_type = input.dtype
        if (self.config.fp16 or self.config.q_int8) and input.dtype == torch.float:
            input = input.half()
        with torch.no_grad():
            attention_output = self.attention(input, input_mask, head_mask, layer_past, get_present, encoder_hidden_states, encoder_attention_mask, output_attentions, self.norm_w, self.norm_b)
            if get_present:
                attention_output, p_key, p_value = attention_output[0:3]
                presents = p_key, p_value
            elif output_attentions:
                attention_output, _, _, context_output = attention_output[0:4]
            else:
                attention_output = attention_output[0]
            residual_add = attention_output + self.attention.attn_ob
            attention_output = self.ds_layernorm(residual_add, self.attn_nw, self.attn_nb, self.config.epsilon)
            if self.config.mlp_type == 'residual':
                res_mlp_out = self.res_mlp(attention_output, async_op=True)
                res_coef_out = self.res_coef_func(attention_output, async_op=True)
            if self.expert_mp_group is not None:
                tensor_list = [torch.empty_like(attention_output) for _ in range(dist.get_world_size(group=self.expert_mp_group))]
                tensor_list[dist.get_rank(group=self.expert_mp_group)] = attention_output
                dist.all_gather(tensor_list, attention_output, group=self.expert_mp_group)
                attention_output = torch.cat(tensor_list).contiguous()
            dispatched_attention, combined_weights = self.moe_gate_einsum(attention_output)
            dispatched_input = self._alltoall(dispatched_attention)
            expert_outputs = self.expert_exec(dispatched_input)
            expert_output = self._alltoall(expert_outputs)
            output = self.scale_expert_output(attention_output, expert_output, combined_weights)
            if self.expert_mp_group is not None:
                output = output.split(output.shape[0] // dist.get_world_size(group=self.expert_mp_group), dim=0)[dist.get_rank(group=self.expert_mp_group)]
            if self.config.mlp_type == 'residual':
                inference_cuda_module.moe_res_matmul(res_mlp_out, res_coef_out, output)
            output = self.bias_residual_func(output, residual_add, torch.empty(1))
            if not self.config.pre_layer_norm:
                output = self.ds_layernorm(output, self.norm_w, self.norm_b, self.config.epsilon)
            if input_type != output.dtype:
                output = output
        if get_present:
            output = output, presents
        if self.config.return_tuple:
            return output if type(output) is tuple else (output,)
        else:
            return output


class GatherTokens(torch.autograd.Function):

    @staticmethod
    def forward(ctx, activations: torch.Tensor, sorted_indices: torch.Tensor, batch_first: bool):
        global random_ltd_module
        if random_ltd_module is None:
            random_ltd_module = RandomLTDBuilder().load()
        ctx.save_for_backward(activations, sorted_indices)
        ctx.batch_first = batch_first
        return activations, random_ltd_module.token_gather(activations, sorted_indices, batch_first)

    @staticmethod
    def backward(ctx, a_gradients: torch.Tensor, g_gradients: torch.Tensor):
        g_gradients = g_gradients.contiguous()
        global random_ltd_module
        if random_ltd_module is None:
            random_ltd_module = RandomLTDBuilder().load()
        activations, sorted_indices = ctx.saved_tensors
        batch_first = ctx.batch_first
        return random_ltd_module.token_scatter_(a_gradients, g_gradients, sorted_indices, batch_first), None, None


RANDOM_LTD_ATTENTION_MASK = 'attention_mask'


RANDOM_LTD_HIDDEN_STATE_ORDER = 'hidden_state_order'


RANDOM_LTD_MAX_VALUE = 'max_value'


RANDOM_LTD_MICRO_BATCH_SIZE = 'micro_batch_size'


RANDOM_LTD_MODEL_MASK_NAME = 'model_mask_name'


RANDOM_LTD_MODEL_TYPE = 'model_type'


RANDOM_LTD_SAMPLE_INDEX = 'sample_idx'


class ScatterTokens(torch.autograd.Function):

    @staticmethod
    def forward(ctx, all_activations: torch.Tensor, layer_activations: torch.Tensor, sorted_indices: torch.Tensor, batch_first: bool):
        global random_ltd_module
        if random_ltd_module is None:
            random_ltd_module = RandomLTDBuilder().load()
        scatter_results = random_ltd_module.token_scatter_(all_activations.clone(), layer_activations, sorted_indices, batch_first)
        ctx.save_for_backward(sorted_indices)
        ctx.batch_first = batch_first
        return scatter_results

    @staticmethod
    def backward(ctx, out_gradients: torch.Tensor):
        out_gradients = out_gradients.contiguous()
        global random_ltd_module
        if random_ltd_module is None:
            random_ltd_module = RandomLTDBuilder().load()
        sorted_indices, = ctx.saved_tensors
        batch_first = ctx.batch_first
        ret_val = random_ltd_module.token_gather(out_gradients, sorted_indices, batch_first)
        return out_gradients, ret_val, None, None


def bert_sample_tokens(reserved_length: int, seq_length: int, batch_size: int, layers: int=1, device: str='cpu', attn_mask: torch.Tensor=None):
    assert attn_mask is not None
    prob_dist = torch.ones((layers * batch_size, seq_length), device=device)
    sampled_indices = torch.multinomial(prob_dist, reserved_length)
    sampled_indices = sampled_indices.reshape(layers, batch_size, reserved_length)
    global random_ltd_module
    if random_ltd_module is None:
        random_ltd_module = RandomLTDBuilder().load()
    sampled_indices = random_ltd_module.token_sort_(sampled_indices, seq_length)
    dtype = sampled_indices.dtype
    sampled_indices = sampled_indices
    new_mask = []
    for l in range(layers):
        tmp_mask_list = []
        for i in range(batch_size):
            mask_tmp = attn_mask[i:i + 1, :, sampled_indices[l][i], :]
            tmp_mask_list.append(mask_tmp[:, :, :, sampled_indices[l][i]])
        new_mask.append(torch.cat(tmp_mask_list, dim=0))
    return sampled_indices, new_mask


def gpt_sample_tokens(reserved_length: int, seq_length: int, batch_size: int, layers: int=1, device: str='cpu', attn_mask: torch.Tensor=None):
    prob_dist = torch.ones((layers * batch_size, seq_length), device=device)
    sampled_indices = torch.multinomial(prob_dist, reserved_length)
    sampled_indices = sampled_indices.reshape(layers, batch_size, reserved_length)
    global random_ltd_module
    if random_ltd_module is None:
        random_ltd_module = RandomLTDBuilder().load()
    sampled_indices = random_ltd_module.token_sort_(sampled_indices, seq_length)
    if attn_mask is not None:
        new_mask = attn_mask[:, :, :reserved_length, :reserved_length]
    else:
        new_mask = None
    return sampled_indices, new_mask


class RandomLayerTokenDrop(Module):
    """
    A  layer wrapper for random LTD
    """

    def __init__(self, layer: Module):
        super(RandomLayerTokenDrop, self).__init__()
        self.random_ltd_layer = layer
        self.reserved_length = None
        self.random_ltd_scheduler = None
        self.max_length = None
        self.reserved_length = -1
        self.curr_seq = -1
        self.batch_first = False

    def init_config(self, config, scheduler, random_ltd_layer_id):
        self.random_ltd_scheduler = scheduler
        self.random_ltd_layer_id = random_ltd_layer_id
        self.max_length = self.random_ltd_scheduler.state[RANDOM_LTD_MAX_VALUE]
        self.mask_name = config[RANDOM_LTD_MODEL_MASK_NAME]
        self.micro_bs = config[RANDOM_LTD_MICRO_BATCH_SIZE]
        self.random_ltd_num_layer = self.random_ltd_scheduler.random_ltd_layer_num
        hs_order = config[RANDOM_LTD_HIDDEN_STATE_ORDER]
        self.model_type = config[RANDOM_LTD_MODEL_TYPE]
        if hs_order == 'batch_seq_dim':
            self.get_hidden_tensor_shape = self.get_bsh
            self.batch_first = True
        elif hs_order == 'seq_batch_dim':
            self.get_hidden_tensor_shape = self.get_sbh
            self.batch_first = False
        else:
            logger.warning('************For now, we only support batch_seq_dim or seq_batch_dim inputs. You can easily                      your own input dimension orders************')
            raise NotImplementedError
        if self.model_type == 'encoder':
            self.index_generator = bert_sample_tokens
        elif self.model_type == 'decoder':
            self.index_generator = gpt_sample_tokens
        else:
            logger.warning('************For now, we only support encoder-only or decoder-only models************')
            raise NotImplementedError

    def get_bsh(self, hidden_stats):
        self.curr_seq, self.curr_micro_batch = hidden_stats.size()[1], hidden_stats.size()[0]

    def get_sbh(self, hidden_stats):
        self.curr_seq, self.curr_micro_batch = hidden_stats.size()[0], hidden_stats.size()[1]

    def forward(self, hidden_states, **kwargs) ->Tensor:
        if self.random_ltd_scheduler is not None:
            self.reserved_length = self.random_ltd_scheduler.get_current_seq()
            self.get_hidden_tensor_shape(hidden_states)
        if self.training and self.random_ltd_scheduler is not None and self.reserved_length < self.curr_seq:
            if self.mask_name is not None:
                mask = kwargs[self.mask_name]
            else:
                mask = None
            if self.random_ltd_layer_id == 0:
                sampled_indices, part_attention_mask = self.index_generator(self.reserved_length, self.curr_seq, self.curr_micro_batch, self.random_ltd_num_layer, hidden_states.device, mask)
                self.random_ltd_scheduler.state[RANDOM_LTD_SAMPLE_INDEX] = sampled_indices
                self.random_ltd_scheduler.state[RANDOM_LTD_ATTENTION_MASK] = part_attention_mask
            else:
                sampled_indices = self.random_ltd_scheduler.state[RANDOM_LTD_SAMPLE_INDEX]
                part_attention_mask = self.random_ltd_scheduler.state[RANDOM_LTD_ATTENTION_MASK]
            hidden_states, part_hidden_states = GatherTokens.apply(hidden_states, sampled_indices[self.random_ltd_layer_id, :, :], self.batch_first)
            if self.mask_name is not None:
                if self.model_type == 'encoder':
                    kwargs[self.mask_name] = part_attention_mask[self.random_ltd_layer_id]
                else:
                    kwargs[self.mask_name] = part_attention_mask
            outputs = self.random_ltd_layer(part_hidden_states, **kwargs)
            if isinstance(outputs, tuple):
                hidden_states = ScatterTokens.apply(hidden_states, outputs[0], sampled_indices[self.random_ltd_layer_id, :, :], self.batch_first)
                my_list = list(outputs)
                my_list[0] = hidden_states
                return tuple(my_list)
            elif isinstance(outputs, Tensor):
                hidden_states = ScatterTokens.apply(hidden_states, outputs, sampled_indices[self.random_ltd_layer_id, :, :], self.batch_first)
                return hidden_states
            else:
                logger.warning('************For now, we only support tuple and tensor output.                         You need to adjust the output according to the layer in your model************')
                raise NotImplementedError
        else:
            return self.random_ltd_layer(hidden_states, **kwargs)


ADAM_W_MODE = 'adam_w_mode'


ADAM_W_MODE_DEFAULT = True


BACKWARD_GLOBAL_TIMER = 'backward'


BACKWARD_INNER_GLOBAL_TIMER = 'backward_inner'


BACKWARD_REDUCE_GLOBAL_TIMER = 'backward_allreduce'


BASE_OPTIMIZER_STATE = 'base_optimizer_state'


CLIP_GRAD = 'clip_grad'


DS_VERSION = 'ds_version'


class DummyOptim:
    """
    Dummy optimizer presents model parameters as a param group, this is
    primarily used to allow ZeRO-3 without an optimizer
    """

    def __init__(self, params):
        self.param_groups = []
        self.param_groups.append({'params': params})


GROUP_PADDINGS = 'group_paddings'


PARAM_SLICE_MAPPINGS = 'param_slice_mappings'


PARTITION_COUNT = 'partition_count'


PIPE_REPLICATED = 'ds_pipe_replicated'


SINGLE_PARTITION_OF_FP32_GROUPS = 'single_partition_of_fp32_groups'


class DeepSpeedOptimizer(object):
    pass


class ZeROOptimizer(DeepSpeedOptimizer):
    pass


def _get_padded_tensor(src_tensor, size):
    if src_tensor.numel() >= size:
        return src_tensor
    padded_tensor = torch.zeros(size, dtype=src_tensor.dtype, device=src_tensor.device)
    slice_tensor = torch.narrow(padded_tensor, 0, 0, src_tensor.numel())
    slice_tensor.data.copy_(src_tensor.data)
    return padded_tensor


def align_dense_tensors(tensor_list, alignment):
    num_elements = sum(t.numel() for t in tensor_list)
    remaining = num_elements % alignment
    if remaining:
        elements_to_add = alignment - remaining
        pad_tensor = torch.zeros(elements_to_add, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
    else:
        padded_tensor_list = tensor_list
    return padded_tensor_list


def all_gather_dp_groups(partitioned_param_groups, dp_process_group, start_alignment_factor, allgather_bucket_size):
    for group_id, partitioned_params in enumerate(partitioned_param_groups):
        partition_id = dist.get_rank(group=dp_process_group[group_id])
        dp_world_size = dist.get_world_size(group=dp_process_group[group_id])
        num_shards = max(1, partitioned_params[partition_id].numel() * dp_world_size // allgather_bucket_size)
        shard_size = partitioned_params[partition_id].numel() // num_shards
        shard_size = shard_size - shard_size % start_alignment_factor
        num_elements = shard_size
        assert shard_size * num_shards <= partitioned_params[partition_id].numel()
        for shard_id in range(num_shards):
            if shard_id == num_shards - 1:
                num_elements = partitioned_params[partition_id].numel() - shard_id * shard_size
            shard_list = []
            for dp_id in range(dp_world_size):
                curr_shard = partitioned_params[dp_id].narrow(0, shard_id * shard_size, num_elements).detach()
                shard_list.append(curr_shard)
            dist.all_gather(shard_list, shard_list[partition_id], dp_process_group[group_id])


def bwc_tensor_model_parallel_rank(mpu=None):
    """Backwards-compatible way of querying the tensor model parallel rank from
    an ``mpu`` object.

    *Tensor* model parallelism means that tensors are physically split across
    processes. This contrasts with *pipeline* model parallelism, in which the
    layers are partitioned but tensors left intact.

    The API for tensor model parallelism has changed across versions and this
    helper provides a best-effort implementation across versions of ``mpu``
    objects.  The preferred mechanism is
    ``mpu.get_tensor_model_parallel_rank()``.

    This should "just work" with both Megatron-LM and DeepSpeed's pipeline
    parallelism.

    Args:
        mpu (model parallel unit, optional): The tensor model parallel rank.
            If ``mpu=None``, returns 0. Defaults to ``None``.

    Returns:
        int: the rank
    """
    if mpu is None:
        return 0
    if hasattr(mpu, 'get_tensor_model_parallel_rank'):
        return mpu.get_tensor_model_parallel_rank()
    elif hasattr(mpu, 'get_slice_parallel_rank'):
        return mpu.get_slice_parallel_rank()
    else:
        return mpu.get_model_parallel_rank()


def get_global_norm_of_tensors(input_tensors, norm_type=2, mpu=None):
    """Get norm of an iterable of tensors.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Taken from Nvidia Megatron.

    Arguments:
        input_tensors (Iterable[Tensor]): an iterable of Tensors will have norm computed
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    assert isinstance(input_tensors, Iterable), f'expected Iterable type not {type(input_tensors)}'
    assert all([torch.is_tensor(t) for t in input_tensors]), f'expected list of only tensors'
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(t.data.abs().max() for t in input_tensors)
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
            total_norm = total_norm_cuda[0].item()
    else:
        total_norm = sum([(t.data.float().norm(norm_type).item() ** norm_type) for t in input_tensors])
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)
    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1
    return total_norm


def clip_tensors_by_global_norm(input_tensors, max_norm=1.0, global_norm=None, mpu=None, eps=1e-06):
    """Clip list of tensors by global norm.
    Args:
        input_tensors: List of tensors to be clipped
        global_norm (float, optional): Precomputed norm. Defaults to None.
        mpu (optional): model parallelism unit. Defaults to None.
        eps (float, optional): epsilon value added to grad norm. Defaults to 1e-6
    Returns:
        float: the global norm
    """
    if global_norm is None:
        global_norm = get_global_norm_of_tensors(input_tensors, mpu=mpu)
    clip_coef = max_norm / (global_norm + eps)
    if clip_coef < 1:
        for t in input_tensors:
            t.detach().mul_(clip_coef)
    return global_norm


CAT_DIM = 'cat_dim'


FP32_WEIGHT_KEY = 'fp32'


PARAM = 'param'


VOCAB_DIVISIBILITY_PADDING_TENSOR = 'vocab_divisibility_padding_tensor'


def load_hp_checkpoint_state(self, folder, tp_rank, tp_world_size):
    hp_mapping = self._hp_mapping
    optim_state_keys = hp_mapping.get_optim_state_keys()
    hp_keys = [FP32_WEIGHT_KEY] + optim_state_keys
    checkpoint_files = {key: os.path.join(folder, f'{key}.pt') for key in hp_keys}
    for file in checkpoint_files.values():
        assert os.path.isfile(file), f'{file} is not a valid file'
    for key in hp_keys:
        ckpt_file = checkpoint_files[key]
        ckpt_dict = torch.load(ckpt_file)
        full_hp_param = ckpt_dict[PARAM]
        if full_hp_param.shape == self.shape:
            tp_rank = 0
            tp_world_size = 1
        vocab_divisibility_padding_tensor = ckpt_dict.get(VOCAB_DIVISIBILITY_PADDING_TENSOR, None)
        if vocab_divisibility_padding_tensor is not None:
            padded_target_vocab_size = self.shape[0] * tp_world_size
            if padded_target_vocab_size > full_hp_param.shape[0]:
                padding_size = padded_target_vocab_size - full_hp_param.shape[0]
                full_hp_param = torch.nn.functional.pad(full_hp_param, (0, 0, 0, padding_size), 'constant', 0)
                full_hp_param[:-padding_size, :] = vocab_divisibility_padding_tensor
            else:
                full_hp_param = full_hp_param[:padded_target_vocab_size, :]
        full_param_numel = full_hp_param.numel()
        tp_slice_numel = self.numel()
        assert full_param_numel == tp_world_size * tp_slice_numel, f'Loading {ckpt_file} full param numel {full_param_numel} != tensor slice numel {tp_slice_numel} * tp_world_size {tp_world_size}'
        dst_tensor = hp_mapping.hp_fragment if key == FP32_WEIGHT_KEY else hp_mapping.get_optim_state_fragment(key)
        chunk_dim = ckpt_dict.get(CAT_DIM, 0)
        tp_hp_slice = full_hp_param.chunk(tp_world_size, chunk_dim)[tp_rank]
        tp_hp_slice = tp_hp_slice.flatten()
        lp_frag_address = hp_mapping.lp_fragment_address
        tp_hp_fragment = tp_hp_slice.narrow(0, lp_frag_address.start, lp_frag_address.numel)
        assert dst_tensor.numel() == lp_frag_address.numel, f'Load checkpoint {key} dst_tensor numel {dst_tensor.numel()} != src numel {lp_frag_address.numel}'
        dst_tensor.data.copy_(tp_hp_fragment.data)


def enable_universal_checkpoint(param_list):
    for param in param_list:
        param.load_hp_checkpoint_state = types.MethodType(load_hp_checkpoint_state, param)


def is_model_parallel_parameter(p) ->bool:
    if hasattr(p, 'model_parallel') and p.model_parallel:
        return True
    if hasattr(p, 'tensor_model_parallel') and p.tensor_model_parallel:
        return True
    return False


def get_full_hp_param(self, optim_state_key=None):
    reduce_buffer = torch.zeros_like(self, dtype=torch.float32).flatten()
    if self._hp_mapping is not None:
        lp_frag_address = self._hp_mapping.lp_fragment_address
        reduce_fragment = torch.narrow(reduce_buffer, 0, lp_frag_address.start, lp_frag_address.numel)
        if optim_state_key is None:
            hp_fragment = self._hp_mapping.hp_fragment
        else:
            hp_fragment = self._hp_mapping.get_optim_state_fragment(optim_state_key)
        reduce_fragment.data.copy_(hp_fragment.data)
    dist.all_reduce(reduce_buffer, group=self._dp_group)
    return reduce_buffer.reshape_as(self)


def _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group):
    current_offset = 0
    param_and_offset_list = []
    partition_end = partition_start + partition_size
    for lp_param in lp_param_list:
        lp_param._hp_mapping = None
        lp_param._dp_group = dp_group
        lp_param.get_full_hp_param = types.MethodType(get_full_hp_param, lp_param)
        lp_param_end = current_offset + lp_param.numel()
        if current_offset < partition_end and lp_param_end > partition_start:
            param_and_offset_list.append((lp_param, current_offset))
        current_offset += lp_param.numel()
    return param_and_offset_list


def get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, partition_start, partition_size, optimizer_state_dict):
    lp_end = lp_param.numel() + lp_start
    hp_start = partition_start
    hp_end = partition_start + partition_size
    fragment_start = max(lp_start, hp_start)
    fragment_end = min(lp_end, hp_end)
    assert fragment_start < fragment_end, f'fragment start {fragment_start} should be < fragment_end {fragment_end}'
    fragment_numel = fragment_end - fragment_start
    hp_frag_address = fragment_address(start=fragment_start - hp_start, numel=fragment_numel)
    hp_fragment_tensor = flat_hp_partition.narrow(0, hp_frag_address.start, hp_frag_address.numel)
    optim_fragment = {key: value.narrow(0, hp_frag_address.start, hp_frag_address.numel) for key, value in optimizer_state_dict.items() if torch.is_tensor(value) and value.shape == flat_hp_partition.shape}
    lp_frag_address = fragment_address(start=fragment_start - lp_start, numel=fragment_numel)
    lp_fragment_tensor = lp_param.flatten().narrow(0, lp_frag_address.start, lp_frag_address.numel)
    return tensor_fragment(lp_fragment=lp_fragment_tensor, lp_fragment_address=lp_frag_address, hp_fragment=hp_fragment_tensor, hp_fragment_address=hp_frag_address, optim_fragment=optim_fragment)


def link_hp_params(lp_param_list, flat_hp_partition, partition_start, partition_size, partition_optimizer_state, dp_group):
    local_lp_param_and_offset = _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group)
    for lp_param, lp_start in local_lp_param_and_offset:
        lp_param._hp_mapping = get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, partition_start, partition_size, partition_optimizer_state)


def see_memory_usage(message):
    logger.info(message)
    logger.info('Memory Allocated %s GigaBytes ', torch.cuda.memory_allocated() / (1024 * 1024 * 1024))
    logger.info('Max Memory Allocated %s GigaBytes', torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024))
    logger.info('Cache Allocated %s GigaBytes', torch.cuda.memory_cached() / (1024 * 1024 * 1024))
    logger.info('Max cache Allocated %s GigaBytes', torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))


class BF16_Optimizer(ZeROOptimizer):

    def __init__(self, init_optimizer, param_names, mpu=None, clip_grad=0.0, norm_type=2, allgather_bucket_size=5000000000, dp_process_group=None, timers=None):
        super().__init__()
        see_memory_usage('begin bf16_optimizer', force=True)
        self.timers = timers
        self.optimizer = init_optimizer
        self.param_names = param_names
        self.using_real_optimizer = not isinstance(self.optimizer, DummyOptim)
        self.clip_grad = clip_grad
        self.norm_type = norm_type
        self.mpu = mpu
        self.allgather_bucket_size = int(allgather_bucket_size)
        self.dp_process_group = dp_process_group
        self.dp_rank = dist.get_rank(group=self.dp_process_group)
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        util_ops = UtilsBuilder().load()
        self.flatten = util_ops.flatten
        self.unflatten = util_ops.unflatten
        self.nccl_start_alignment_factor = 2
        self.bf16_groups = []
        self.bf16_groups_flat = []
        self.bf16_partitioned_groups = []
        self.fp32_groups_flat_partition = []
        self.fp32_groups_gradients = []
        self.fp32_groups_gradients_flat = []
        self.fp32_groups_actual_gradients_flat = []
        self.fp32_groups_gradient_flat_partition = []
        self.fp32_groups_has_gradients = []
        self.step_count = 0
        self.group_paddings = []
        if self.using_real_optimizer:
            self._setup_for_real_optimizer()
        see_memory_usage('end bf16_optimizer', force=True)

    def _setup_for_real_optimizer(self):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        self.partition_count = [dp_world_size for i in range(len(self.optimizer.param_groups))]
        for i, param_group in enumerate(self.optimizer.param_groups):
            see_memory_usage(f'before initializing group {i}', force=True)
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            self.bf16_groups.append(param_group['params'])
            self.bf16_groups_flat.append(self._flatten_dense_tensors_aligned(self.bf16_groups[i], self.nccl_start_alignment_factor * dp_world_size))
            self._update_storage_to_flattened_tensor(tensor_list=self.bf16_groups[i], flat_tensor=self.bf16_groups_flat[i])
            partition_size = self.bf16_groups_flat[i].numel() // dp_world_size
            bf16_dp_partitions = [self.bf16_groups_flat[i].narrow(0, dp_index * partition_size, partition_size) for dp_index in range(dp_world_size)]
            self.bf16_partitioned_groups.append(bf16_dp_partitions)
            self.fp32_groups_flat_partition.append(bf16_dp_partitions[partition_id].clone().float().detach())
            self.fp32_groups_flat_partition[i].requires_grad = True
            num_elem_list = [t.numel() for t in self.bf16_groups[i]]
            self.fp32_groups_gradients_flat.append(torch.zeros_like(self.bf16_groups_flat[i], dtype=torch.float32))
            fp32_gradients = self._split_flat_tensor(flat_tensor=self.fp32_groups_gradients_flat[i], num_elem_list=num_elem_list)
            self.fp32_groups_gradients.append(fp32_gradients)
            length_without_padding = sum(num_elem_list)
            self.fp32_groups_actual_gradients_flat.append(torch.narrow(self.fp32_groups_gradients_flat[i], 0, 0, length_without_padding))
            self.fp32_groups_gradient_flat_partition.append(torch.narrow(self.fp32_groups_gradients_flat[i], 0, partition_id * partition_size, partition_size))
            self.fp32_groups_has_gradients.append([False] * len(self.bf16_groups[i]))
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                padding = self.bf16_groups_flat[i].numel() - length_without_padding
            else:
                padding = 0
            self.group_paddings.append(padding)
            param_group['params'] = [self.fp32_groups_flat_partition[i]]
            see_memory_usage(f'after initializing group {i}', force=True)
        see_memory_usage('before initialize_optimizer', force=True)
        self.initialize_optimizer_states()
        see_memory_usage('end initialize_optimizer', force=True)
        self._link_all_hp_params()
        self._enable_universal_checkpoint()
        self._param_slice_mappings = self._create_param_mapping()

    def _enable_universal_checkpoint(self):
        for lp_param_group in self.bf16_groups:
            enable_universal_checkpoint(param_list=lp_param_group)

    def _create_param_mapping(self):
        param_mapping = []
        for i, _ in enumerate(self.optimizer.param_groups):
            param_mapping_per_group = OrderedDict()
            for lp in self.bf16_groups[i]:
                if lp._hp_mapping is not None:
                    lp_name = self.param_names[lp]
                    param_mapping_per_group[lp_name] = lp._hp_mapping.get_hp_fragment_address()
            param_mapping.append(param_mapping_per_group)
        return param_mapping

    def _link_all_hp_params(self):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        for i, _ in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            partition_size = self.bf16_groups_flat[i].numel() // dp_world_size
            flat_hp_partition = self.fp32_groups_flat_partition[i]
            link_hp_params(lp_param_list=self.bf16_groups[i], flat_hp_partition=flat_hp_partition, partition_start=partition_id * partition_size, partition_size=partition_size, partition_optimizer_state=self.optimizer.state[flat_hp_partition], dp_group=self.real_dp_process_group[i])

    def initialize_optimizer_states(self):
        """Take an optimizer step with zero-valued gradients to allocate internal
        optimizer state.

        This helps prevent memory fragmentation by allocating optimizer state at the
        beginning of training instead of after activations have been allocated.
        """
        for param_partition, grad_partition in zip(self.fp32_groups_flat_partition, self.fp32_groups_gradient_flat_partition):
            param_partition.grad = grad_partition
        self.optimizer.step()
        self.clear_hp_grads()

    def _split_flat_tensor(self, flat_tensor, num_elem_list):
        assert sum(num_elem_list) <= flat_tensor.numel()
        tensor_list = []
        offset = 0
        for num_elem in num_elem_list:
            dense_tensor = torch.narrow(flat_tensor, 0, offset, num_elem)
            tensor_list.append(dense_tensor)
            offset += num_elem
        return tensor_list

    def _update_storage_to_flattened_tensor(self, tensor_list, flat_tensor):
        updated_params = self.unflatten(flat_tensor, tensor_list)
        for p, q in zip(tensor_list, updated_params):
            p.data = q.data

    def _flatten_dense_tensors_aligned(self, tensor_list, alignment):
        return self.flatten(align_dense_tensors(tensor_list, alignment))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError(f'{self.__class__} does not support closure.')
        all_groups_norm = get_global_norm_of_tensors(input_tensors=self.get_grads_for_norm(), mpu=self.mpu, norm_type=self.norm_type)
        self._global_grad_norm = all_groups_norm
        assert all_groups_norm > 0.0
        if self.clip_grad > 0.0:
            clip_tensors_by_global_norm(input_tensors=self.get_grads_for_norm(for_clipping=True), max_norm=self.clip_grad, global_norm=all_groups_norm, mpu=self.mpu)
        self.optimizer.step()
        self.update_lp_params()
        self.clear_hp_grads()
        self.step_count += 1

    def backward(self, loss, update_hp_grads=True, clear_lp_grads=False, **bwd_kwargs):
        """Perform a backward pass and copy the low-precision gradients to the
        high-precision copy.

        We copy/accumulate to the high-precision grads now to prevent accumulating in the
        bf16 grads after successive backward() calls (i.e., grad accumulation steps > 1)

        The low-precision grads are deallocated during this procedure.
        """
        self.clear_lp_grads()
        loss.backward(**bwd_kwargs)
        if update_hp_grads:
            self.update_hp_grads(clear_lp_grads=clear_lp_grads)

    @torch.no_grad()
    def update_hp_grads(self, clear_lp_grads=False):
        for i, group in enumerate(self.bf16_groups):
            for j, lp in enumerate(group):
                if lp.grad is None:
                    continue
                hp_grad = self.fp32_groups_gradients[i][j]
                assert hp_grad is not None, f'high precision param has no gradient, lp param_id = {id(lp)} group_info = [{i}][{j}]'
                hp_grad.data.add_(lp.grad.data.view(hp_grad.shape))
                lp._hp_grad = hp_grad
                self.fp32_groups_has_gradients[i][j] = True
                if clear_lp_grads:
                    lp.grad = None

    @torch.no_grad()
    def get_grads_for_reduction(self):
        return self.fp32_groups_gradients_flat

    @torch.no_grad()
    def get_grads_for_norm(self, for_clipping=False):
        grads = []
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        for i, group in enumerate(self.bf16_groups):
            for j, lp in enumerate(group):
                if not for_clipping:
                    if hasattr(lp, PIPE_REPLICATED) and lp.ds_pipe_replicated:
                        continue
                    if not (tensor_mp_rank == 0 or is_model_parallel_parameter(lp)):
                        continue
                if not self.fp32_groups_has_gradients[i][j]:
                    continue
                grads.append(self.fp32_groups_gradients[i][j])
        return grads

    @torch.no_grad()
    def update_lp_params(self):
        for i, (bf16_partitions, fp32_partition) in enumerate(zip(self.bf16_partitioned_groups, self.fp32_groups_flat_partition)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            bf16_partitions[partition_id].data.copy_(fp32_partition.data)
        all_gather_dp_groups(partitioned_param_groups=self.bf16_partitioned_groups, dp_process_group=self.real_dp_process_group, start_alignment_factor=self.nccl_start_alignment_factor, allgather_bucket_size=self.allgather_bucket_size)

    def clear_hp_grads(self):
        for flat_gradients in self.fp32_groups_gradients_flat:
            flat_gradients.zero_()
        for i, group in enumerate(self.fp32_groups_gradients):
            self.fp32_groups_has_gradients[i] = [False] * len(group)

    def clear_lp_grads(self):
        for group in self.bf16_groups:
            for param in group:
                param.grad = None

    def state_dict(self):
        state_dict = {}
        state_dict[CLIP_GRAD] = self.clip_grad
        state_dict[BASE_OPTIMIZER_STATE] = self.optimizer.state_dict()
        state_dict[SINGLE_PARTITION_OF_FP32_GROUPS] = self.fp32_groups_flat_partition
        state_dict[GROUP_PADDINGS] = self.group_paddings
        state_dict[PARTITION_COUNT] = self.partition_count
        state_dict[DS_VERSION] = version
        state_dict[PARAM_SLICE_MAPPINGS] = self._param_slice_mappings
        return state_dict

    def _restore_from_bit16_weights(self):
        for i, group in enumerate(self.bf16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            for bf16_partitions, fp32_partition in zip(self.bf16_partitioned_groups, self.fp32_groups_flat_partition):
                fp32_partition.data.copy_(bf16_partitions[partition_id].data)

    def refresh_fp32_params(self):
        self._restore_from_bit16_weights()

    def load_state_dict(self, state_dict_list, checkpoint_folder, load_optimizer_states=True, load_from_fp32_weights=False):
        if checkpoint_folder:
            self._load_universal_checkpoint(checkpoint_folder, load_optimizer_states, load_from_fp32_weights)
        else:
            self._load_legacy_checkpoint(state_dict_list, load_optimizer_states, load_from_fp32_weights)

    def _load_legacy_checkpoint(self, state_dict_list, load_optimizer_states=True, load_from_fp32_weights=False):
        dp_rank = dist.get_rank(group=self.dp_process_group)
        current_rank_sd = state_dict_list[dp_rank]
        ckpt_version = current_rank_sd.get(DS_VERSION, False)
        assert ckpt_version, f'Empty ds_version in checkpoint, not clear how to proceed'
        ckpt_version = pkg_version.parse(ckpt_version)
        self.clip_grad = current_rank_sd.get(CLIP_GRAD, self.clip_grad)
        if load_optimizer_states:
            self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])
        if load_from_fp32_weights:
            for current, saved in zip(self.fp32_groups_flat_partition, current_rank_sd[SINGLE_PARTITION_OF_FP32_GROUPS]):
                src_tensor = _get_padded_tensor(saved, current.numel())
                current.data.copy_(src_tensor.data)
        if load_optimizer_states:
            self._link_all_hp_params()

    def _load_universal_checkpoint(self, checkpoint_folder, load_optimizer_states, load_from_fp32_weights):
        self._load_hp_checkpoint_state(checkpoint_folder)

    @property
    def param_groups(self):
        """Forward the wrapped optimizer's parameters."""
        return self.optimizer.param_groups

    def _load_hp_checkpoint_state(self, checkpoint_dir):
        checkpoint_dir = os.path.join(checkpoint_dir, 'zero')
        tp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        tp_world_size = self.mpu.get_slice_parallel_world_size()
        for i, _ in enumerate(self.optimizer.param_groups):
            for lp in self.bf16_groups[i]:
                if lp._hp_mapping is not None:
                    lp.load_hp_checkpoint_state(os.path.join(checkpoint_dir, self.param_names[lp]), tp_rank, tp_world_size)


CURRICULUM_LEARNING_CURRENT_DIFFICULTY = 'current_difficulty'


CURRICULUM_LEARNING_MAX_DIFFICULTY = 'max_difficulty'


CURRICULUM_LEARNING_MIN_DIFFICULTY = 'min_difficulty'


CURRICULUM_LEARNING_SCHEDULE_CONFIG = 'schedule_config'


CURRICULUM_LEARNING_SCHEDULE_CUSTOM = 'custom'


CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY = 'difficulty'


CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP = 'difficulty_step'


CURRICULUM_LEARNING_SCHEDULE_FIXED_DISCRETE = 'fixed_discrete'


CURRICULUM_LEARNING_SCHEDULE_FIXED_LINEAR = 'fixed_linear'


CURRICULUM_LEARNING_SCHEDULE_FIXED_ROOT = 'fixed_root'


CURRICULUM_LEARNING_SCHEDULE_MAX_STEP = 'max_step'


CURRICULUM_LEARNING_SCHEDULE_ROOT_DEGREE = 'root_degree'


CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP = 'total_curriculum_step'


CURRICULUM_LEARNING_SCHEDULE_TYPE = 'schedule_type'


class CurriculumScheduler(object):

    def __init__(self, config):
        super().__init__()
        self.state = {}
        assert CURRICULUM_LEARNING_MIN_DIFFICULTY in config, f"Curriculum learning requires the config '{CURRICULUM_LEARNING_MIN_DIFFICULTY}'"
        assert CURRICULUM_LEARNING_MAX_DIFFICULTY in config, f"Curriculum learning requires the config '{CURRICULUM_LEARNING_MAX_DIFFICULTY}'"
        assert CURRICULUM_LEARNING_SCHEDULE_TYPE in config, f"Curriculum learning requires the config '{CURRICULUM_LEARNING_SCHEDULE_TYPE}'"
        self.state[CURRICULUM_LEARNING_MIN_DIFFICULTY] = config[CURRICULUM_LEARNING_MIN_DIFFICULTY]
        self.state[CURRICULUM_LEARNING_MAX_DIFFICULTY] = config[CURRICULUM_LEARNING_MAX_DIFFICULTY]
        self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY] = config[CURRICULUM_LEARNING_MIN_DIFFICULTY]
        self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] = config[CURRICULUM_LEARNING_SCHEDULE_TYPE]
        self.first_step = True
        if config[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_DISCRETE:
            """
            The schedule_config is a list of difficulty and a list of max
            step belonging to each difficulty. Example json config:
            "schedule_config": {
              "difficulty": [1,2,3],
              "max_step": [5,10]
            }
            The "max_step" has one less element than "difficulty", because
            the last difficulty will be used for all following steps.
            The self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG] is a dictionary of
            difficulty : [max step for this difficulty, next difficulty].
            """
            assert CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], f"Curriculum learning with fixed_discrete schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY}'"
            assert CURRICULUM_LEARNING_SCHEDULE_MAX_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], f"Curriculum learning with fixed_discrete schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_MAX_STEP}'"
            assert len(config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_MAX_STEP]) > 0
            assert len(config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY]) > 0
            assert len(config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY]) == len(config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_MAX_STEP]) + 1
            self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG] = config[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        elif config[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_ROOT:
            """
            The schedule_config includes:
            total_curriculum_step: how many steps the curriculum learning takes to go
            from min difficulty to max difficulty.
            difficulty_step: the difficulty level determined every time must
            be a multiple of this difficulty_step. This is used to determine
            the step of difficulty increase, and to ensure the use of NVIDIA
            Tensor Core acceleration (requires multiple of 8 (FP16) or
            16 (INT8)).
            root_degree: the degree of the root function. Degree of 2 means
            square root and degree of 3 means cube root. Degree of 1 is
            equivalent to linear.
            "schedule_config": {
              "total_curriculum_step": 30000,
              "difficulty_step": 8,
              "root_degree": 2
            }
            """
            assert CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], f"Curriculum learning with fixed_root schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP}'"
            assert CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], f"Curriculum learning with fixed_root schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP}'"
            assert CURRICULUM_LEARNING_SCHEDULE_ROOT_DEGREE in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], f"Curriculum learning with fixed_root schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_ROOT_DEGREE}'"
            if config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP] % 8 != 0:
                logger.warning(f'When using seqlen metric, the difficulty_step for curriculum learning has to be multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. Disregard this warning if this is unrelated to your metric/hardware.')
            self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG] = config[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        elif config[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_LINEAR:
            """
            The schedule_config is the same as CURRICULUM_LEARNING_SCHEDULE_FIXED_ROOT but without the
            root_degree.
            "schedule_config": {
              "total_curriculum_step": 30000,
              "difficulty_step": 8
            }
            """
            assert CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], f"Curriculum learning with fixed_linear schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP}'"
            assert CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], f"Curriculum learning with fixed_linear schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP}'"
            if config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP] % 8 != 0:
                logger.warning(f'When using seqlen metric, the difficulty_step for curriculum learning has to be multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. Disregard this warning if this is unrelated to your metric/hardware.')
            self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG] = config[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        elif config[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_CUSTOM:
            """
            Fully customized schedule. User need to provide a custom schedule
            function by using the set_custom_curriculum_learning_schedule API
            in deepspeed/runtime/engine.py
            """
            self.custom_get_difficulty = None
        else:
            raise RuntimeError('Unsupported curriculum schedule type')

    def get_current_difficulty(self):
        return self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY]

    def set_current_difficulty(self, difficulty):
        self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY] = difficulty

    def set_custom_get_difficulty(self, schedule_function):
        self.custom_get_difficulty = schedule_function

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def __fixed_discrete_get_difficulty(self, global_steps):
        s_state = self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        if global_steps > s_state[CURRICULUM_LEARNING_SCHEDULE_MAX_STEP][-1]:
            return s_state[CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY][-1]
        for i in range(len(s_state[CURRICULUM_LEARNING_SCHEDULE_MAX_STEP])):
            if global_steps <= s_state[CURRICULUM_LEARNING_SCHEDULE_MAX_STEP][i]:
                return s_state[CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY][i]

    def __fixed_root_get_difficulty(self, global_steps, root_degree=None):
        s_state = self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        if root_degree is None:
            root_degree = s_state[CURRICULUM_LEARNING_SCHEDULE_ROOT_DEGREE]
        next_difficulty = (float(global_steps) / s_state[CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP]) ** (1.0 / root_degree)
        next_difficulty = math.floor(next_difficulty * (self.state[CURRICULUM_LEARNING_MAX_DIFFICULTY] - self.state[CURRICULUM_LEARNING_MIN_DIFFICULTY]) + self.state[CURRICULUM_LEARNING_MIN_DIFFICULTY])
        next_difficulty -= next_difficulty % s_state[CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP]
        next_difficulty = min(next_difficulty, self.state[CURRICULUM_LEARNING_MAX_DIFFICULTY])
        return next_difficulty

    def get_difficulty(self, global_steps):
        if self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_DISCRETE:
            return self.__fixed_discrete_get_difficulty(global_steps)
        elif self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_LINEAR:
            return self.__fixed_root_get_difficulty(global_steps, 1)
        elif self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_ROOT:
            return self.__fixed_root_get_difficulty(global_steps)
        elif self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_CUSTOM:
            return self.custom_get_difficulty(global_steps)
        else:
            raise RuntimeError('Unsupported curriculum schedule type')

    def update_difficulty(self, global_steps):
        if self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY] < self.state[CURRICULUM_LEARNING_MAX_DIFFICULTY]:
            self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY] = self.get_difficulty(global_steps)
        return self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY]


DATA_PARALLEL_GROUP = 'data_parallel_group'


CURRICULUM_LEARNING_BATCH = 'batch'


CURRICULUM_LEARNING_CLUSTERING_TYPE = 'clustering_type'


CURRICULUM_LEARNING_CLUSTER_PATH = 'data_cluster_path'


CURRICULUM_LEARNING_CLUSTER_PREFIX = 'cluster'


CURRICULUM_LEARNING_CONSUMED_SAMPLES = 'consumed_samples'


CURRICULUM_LEARNING_CURRENT_DIFFICULTIES = 'current_difficulties'


CURRICULUM_LEARNING_DATA_CLUSTER_CURRENT_POSITION = 'data_cluster_current_position'


CURRICULUM_LEARNING_DATA_CLUSTER_PATHS = 'data_cluster_paths'


CURRICULUM_LEARNING_DIFFICULTY_TYPE = 'difficulty_type'


CURRICULUM_LEARNING_METRIC_PATH = 'index_to_metric_path'


CURRICULUM_LEARNING_NP_RNG_STATE = 'np_rng_state'


CURRICULUM_LEARNING_PERCENTILE_BASED = 'percentile'


CURRICULUM_LEARNING_SAMPLE_PATH = 'index_to_sample_path'


CURRICULUM_LEARNING_SINGLE_CLUSTER = 'single_cluster'


CURRICULUM_LEARNING_STEP = 'curriculum_step'


CURRICULUM_LEARNING_VALUE_BASED = 'value'


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


dtypes = {(1): np.uint8, (2): np.int8, (3): np.int16, (4): np.int32, (5): np.int64, (6): np.float64, (7): np.double, (8): np.uint16, (9): np.uint32, (10): np.uint64}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def exscan_from_cumsum_(arr):
    if arr.size > 1:
        arr[1:] = arr[:-1]
    if arr.size > 0:
        arr[0] = 0


def get_pointers_with_total(sizes, elemsize, dtype):
    """Return a numpy array of type np.dtype giving the byte offsets.

    Multiplies values in the sizes array by elemsize (bytes),
    and then computes an exclusive scan to get byte offsets.
    Returns the total number of bytes as second item in a tuple.
    """
    pointers = np.array(sizes, dtype=dtype)
    pointers *= elemsize
    np.cumsum(pointers, axis=0, out=pointers)
    bytes_last = pointers[-1] if len(sizes) > 0 else 0
    exscan_from_cumsum_(pointers)
    return pointers, bytes_last


def index_file_path(prefix_path):
    return prefix_path + '.idx'


class MMapIndexedDataset(torch.utils.data.Dataset):


    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):


            class _Writer(object):

                def __enter__(self):
                    self._file = open(path, 'wb')
                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))
                    return self

                @staticmethod
                def _get_pointers(sizes, npdtype):
                    """Return a numpy array of byte offsets given a list of sizes.

                    Multiplies values in the sizes array by dtype size (bytes),
                    and then computes an exclusive scan to get byte offsets.
                    """
                    pointers, _ = get_pointers_with_total(sizes, dtype().itemsize, npdtype)
                    return pointers

                def write(self, sizes, doc_idx):
                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', len(doc_idx)))
                    sizes32 = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes32.tobytes(order='C'))
                    del sizes32
                    pointers = self._get_pointers(sizes, np.int64)
                    del sizes
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers
                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()
            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, "Index file doesn't match expected format. Make sure that --dataset-impl is configured properly."
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version
                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize
                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()
            if not skip_warmup:
                None
                _warmup_mmap_file(path)
            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            None
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            None
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes)
            None
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count, offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()
        self._path = None
        self._index = None
        self._bin_buffer = None
        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)
        if not skip_warmup:
            None
            _warmup_mmap_file(data_file_path(self._path))
        None
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        None
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError('Slices into indexed_dataset must be contiguous')
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    def size(self, index):
        return self._index.sizes[index]

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

    @property
    def dtype(self):
        return self._index.dtype


def close_mmap_dataset_builder(builder, fname):
    builder.end_document()
    builder.finalize(f'{fname}.idx')
    logger.info(f'Finalized mmap dataset builder at {fname}.')


class MMapIndexedDatasetBuilder(object):

    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def add_item_numpy(self, np_array):
        if np_array.dtype != self._dtype:
            np_array = np_array.astype(self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype
        total_len = len(index.sizes) + len(self._sizes)
        None
        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


def create_mmap_dataset_builder(fname, dtype):
    logger.info(f'Creating mmap dataset builder at {fname}.')
    return MMapIndexedDatasetBuilder(f'{fname}.bin', dtype=dtype)


def find_fit_int_dtype(min_value, max_value):
    if min_value >= 0:
        if max_value <= 255:
            return np.uint8
        elif max_value <= 65535:
            return np.uint16
        elif max_value <= 4294967295:
            return np.uint32
        else:
            return np.uint64
    elif max_value <= 127 and min_value >= -128:
        return np.int8
    elif max_value <= 32767 and min_value >= -32768:
        return np.int16
    elif max_value <= 2147483647 and min_value >= -2147483648:
        return np.int32
    else:
        return np.int64


class DeepSpeedDataSampler(object):

    def __init__(self, data_efficiency_config, one_epoch_total_samples, micro_batch_size, data_parallel_rank, data_parallel_size, data_parallel_group, gradient_accumulation_steps, global_rank, drop_last=True):
        self.data_efficiency_config = data_efficiency_config
        self.one_epoch_total_samples = one_epoch_total_samples
        self.index_dtype = find_fit_int_dtype(0, one_epoch_total_samples)
        self.total_samples = one_epoch_total_samples * self.data_efficiency_config[DATA_SAMPLING][DATA_SAMPLING_NUM_EPOCHS]
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_group = data_parallel_group
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_batch_size = self.micro_batch_times_data_parallel_size * self.gradient_accumulation_steps
        self.global_rank = global_rank
        self.drop_last = drop_last
        self.np_rng = np.random.default_rng(self.data_efficiency_config[DATA_EFFICIENCY_SEED])
        self.state = {}
        self.batch = []
        self.consumed_samples = 0
        if self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]:
            self.curriculum_step = 0
            self.current_difficulties = {}
            self.data_cluster_paths = []
            self.data_cluster_current_position = []
            self.curriculum_schedulers = {}
            self.curriculum_index_to_sample = {}
            self.curriculum_index_to_metric = {}
            self.difficulty_type = {}
            self.clustering_type = {}
            self.data_1epoch_size = None
            if self.global_rank == 0:
                self.data_clusters = []
                self.data_cluster_sizes = []
                cluster_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_CLUSTER_PATH]
                if not os.path.exists(cluster_path):
                    os.makedirs(cluster_path)
            for metric in self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS]:
                self.curriculum_schedulers[metric] = CurriculumScheduler(data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][metric])
                self.difficulty_type[metric] = data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][metric][CURRICULUM_LEARNING_DIFFICULTY_TYPE]
                self.clustering_type[metric] = data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][metric][CURRICULUM_LEARNING_CLUSTERING_TYPE]
                if self.global_rank == 0:
                    if self.clustering_type[metric] != CURRICULUM_LEARNING_SINGLE_CLUSTER:
                        self.curriculum_index_to_sample[metric] = MMapIndexedDataset(data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][metric][CURRICULUM_LEARNING_SAMPLE_PATH], skip_warmup=True)
                        if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                            self.curriculum_index_to_metric[metric] = MMapIndexedDataset(data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][metric][CURRICULUM_LEARNING_METRIC_PATH], skip_warmup=True)
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, 'data_parallel_rank should be smaller than data size: {}, {}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def set_custom_curriculum_learning_schedule(self, schedule_func_dict):
        for metric in self.curriculum_schedulers:
            if metric in schedule_func_dict:
                self.curriculum_schedulers[metric].set_custom_get_difficulty(schedule_func_dict[metric])

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def get_sample_based_on_metric_value(self, metric, value_start, value_end):
        new_samples = None
        for row in range(len(self.curriculum_index_to_sample[metric])):
            if self.curriculum_index_to_metric[metric][row] <= value_end and self.curriculum_index_to_metric[metric][row] > value_start:
                row_samples = np.copy(self.curriculum_index_to_sample[metric][row])
                new_samples = row_samples if new_samples is None else np.concatenate((new_samples, row_samples), axis=None)
        return new_samples

    def get_sample_based_on_metric_percentile(self, metric, percentile_start, percentile_end):
        new_samples = None
        if self.data_1epoch_size is None:
            self.data_1epoch_size = sum(len(x) for x in self.curriculum_index_to_sample[metric])
        max_percentile = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][metric][CURRICULUM_LEARNING_MAX_DIFFICULTY]
        sample_per_percentile = self.data_1epoch_size // max_percentile
        start_count = sample_per_percentile * percentile_start
        end_count = sample_per_percentile * percentile_end
        if percentile_end == max_percentile:
            end_count = self.data_1epoch_size
        current_count = 0
        for row in range(len(self.curriculum_index_to_sample[metric])):
            row_size = len(self.curriculum_index_to_sample[metric][row])
            if current_count + row_size > start_count:
                row_start = max(0, start_count - current_count)
                if current_count + row_size <= end_count:
                    row_end = row_size
                else:
                    row_end = end_count - current_count
                row_samples = np.copy(self.curriculum_index_to_sample[metric][row][row_start:row_end])
                new_samples = row_samples if new_samples is None else np.concatenate((new_samples, row_samples), axis=None)
            current_count += row_size
            if current_count >= end_count:
                break
        return new_samples

    def get_new_cluster(self, previous_difficulties):
        cluster_fname = CURRICULUM_LEARNING_CLUSTER_PREFIX
        for metric in self.curriculum_schedulers:
            cluster_fname = f'{cluster_fname}_{metric}{self.current_difficulties[metric]}'
        cluster_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_CLUSTER_PATH]
        cluster_path = f'{cluster_path}/{cluster_fname}'
        if self.global_rank == 0:
            new_cluster = None
            need_clustering = 0
            for metric in self.clustering_type:
                if self.clustering_type[metric] != CURRICULUM_LEARNING_SINGLE_CLUSTER:
                    need_clustering += 1
            if need_clustering > 1:
                for metric in self.curriculum_schedulers:
                    if self.clustering_type[metric] == CURRICULUM_LEARNING_SINGLE_CLUSTER:
                        metric_cluster = np.arange(start=0, stop=self.one_epoch_total_samples, step=1, dtype=self.index_dtype)
                    elif self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                        metric_cluster = self.get_sample_based_on_metric_value(metric, float('-inf'), self.current_difficulties[metric])
                    elif self.difficulty_type[metric] == CURRICULUM_LEARNING_PERCENTILE_BASED:
                        metric_cluster = self.get_sample_based_on_metric_percentile(metric, 0, self.current_difficulties[metric])
                    new_cluster = metric_cluster if new_cluster is None else np.intersect1d(new_cluster, metric_cluster, assume_unique=True)
                for cluster in self.data_clusters:
                    new_cluster = np.setdiff1d(new_cluster, cluster[0], assume_unique=True)
            else:
                if len(self.data_clusters) == 0:
                    new_cluster = np.arange(start=0, stop=self.one_epoch_total_samples, step=1, dtype=self.index_dtype)
                for metric in self.curriculum_schedulers:
                    if self.clustering_type[metric] != CURRICULUM_LEARNING_SINGLE_CLUSTER:
                        if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                            new_cluster = self.get_sample_based_on_metric_value(metric, previous_difficulties[metric], self.current_difficulties[metric])
                        elif self.difficulty_type[metric] == CURRICULUM_LEARNING_PERCENTILE_BASED:
                            new_cluster = self.get_sample_based_on_metric_percentile(metric, previous_difficulties[metric], self.current_difficulties[metric])
            if new_cluster is not None and len(new_cluster) > 0:
                logger.info(f'new data cluster (previous_difficulties {previous_difficulties}, current_difficulties {self.current_difficulties}) with size {len(new_cluster)} generated.')
                self.np_rng.shuffle(new_cluster)
                cluster_builder = create_mmap_dataset_builder(cluster_path, self.index_dtype)
                cluster_builder.add_item_numpy(new_cluster)
                close_mmap_dataset_builder(cluster_builder, cluster_path)
                self.data_clusters.append(MMapIndexedDataset(cluster_path, skip_warmup=True))
                self.data_cluster_sizes.append(len(self.data_clusters[-1][0]))
            else:
                logger.info(f'new data cluster (previous_difficulties {previous_difficulties}, current_difficulties {self.current_difficulties}) has no matched data thus skipped.')
        dist.barrier(group=self.data_parallel_group)
        if os.path.isfile(f'{cluster_path}.bin'):
            self.data_cluster_paths.append(cluster_fname)
            self.data_cluster_current_position.append(0)

    def sample_from_clusters(self):
        num_clusters = len(self.data_clusters)
        weight_sum = sum(self.data_cluster_sizes)
        weights = [(x / weight_sum) for x in self.data_cluster_sizes]
        samples = self.np_rng.choice(num_clusters, self.global_batch_size, replace=True, p=weights)
        samples = np.bincount(samples, minlength=num_clusters)
        return samples

    def reshuffle_clusters(self, cidx):
        cluster_fname = self.data_cluster_paths[cidx]
        cluster_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_CLUSTER_PATH]
        cluster_path = f'{cluster_path}/{cluster_fname}'
        cluster = np.copy(self.data_clusters[cidx][0])
        self.np_rng.shuffle(cluster)
        cluster_builder = create_mmap_dataset_builder(cluster_path, self.index_dtype)
        cluster_builder.add_item_numpy(cluster)
        close_mmap_dataset_builder(cluster_builder, cluster_path)
        self.data_clusters[cidx] = MMapIndexedDataset(cluster_path, skip_warmup=True)

    def get_sample_from_cluster(self, cidx, num_samples):
        start_idx = self.data_cluster_current_position[cidx]
        samples = list(np.copy(self.data_clusters[cidx][0][start_idx:start_idx + num_samples]))
        self.data_cluster_current_position[cidx] += num_samples
        if len(samples) < num_samples:
            num_samples_remained = num_samples - len(samples)
            logger.info(f'reshuffling cluster {cidx}.')
            self.reshuffle_clusters(cidx)
            samples += list(np.copy(self.data_clusters[cidx][0][:num_samples_remained]))
            self.data_cluster_current_position[cidx] = num_samples_remained
        return samples

    def get_next_global_batch(self):
        if self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]:
            self.curriculum_step += 1
            new_cluster = False
            previous_difficulties = {}
            for metric in self.curriculum_schedulers:
                next_difficulty = self.curriculum_schedulers[metric].update_difficulty(self.curriculum_step)
                if metric not in self.current_difficulties or next_difficulty != self.current_difficulties[metric]:
                    new_cluster = True
                if metric in self.current_difficulties:
                    previous_difficulties[metric] = self.current_difficulties[metric]
                elif self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                    previous_difficulties[metric] = float('-inf')
                elif self.difficulty_type[metric] == CURRICULUM_LEARNING_PERCENTILE_BASED:
                    previous_difficulties[metric] = 0
                self.current_difficulties[metric] = next_difficulty
            if new_cluster:
                self.get_new_cluster(previous_difficulties)
            if self.global_rank == 0:
                samples_per_cluster = self.sample_from_clusters()
                batch = []
                for cidx in range(len(samples_per_cluster)):
                    batch += self.get_sample_from_cluster(cidx, samples_per_cluster[cidx])
                self.np_rng.shuffle(batch)
                batch = torch.tensor(batch, device=torch.cuda.current_device(), dtype=torch.long).view(-1)
            else:
                batch = torch.empty(self.global_batch_size, device=torch.cuda.current_device(), dtype=torch.long)
            dist.broadcast(batch, 0, group=self.data_parallel_group)
            self.batch = batch.tolist()

    def __iter__(self):
        while self.consumed_samples <= self.total_samples:
            if len(self.batch) == 0:
                self.get_next_global_batch()
            current_batch = self.batch[:self.micro_batch_times_data_parallel_size]
            self.batch = self.batch[self.micro_batch_times_data_parallel_size:]
            if len(current_batch) == self.micro_batch_times_data_parallel_size or len(current_batch) > 0 and not self.drop_last:
                start_idx, end_idx = self.get_start_end_idx()
                yield current_batch[start_idx:end_idx]
                self.consumed_samples += len(current_batch)
                current_batch = []

    def state_dict(self):
        return {CURRICULUM_LEARNING_BATCH: self.batch, CURRICULUM_LEARNING_CONSUMED_SAMPLES: self.consumed_samples, CURRICULUM_LEARNING_STEP: self.curriculum_step, CURRICULUM_LEARNING_CURRENT_DIFFICULTIES: self.current_difficulties, CURRICULUM_LEARNING_DATA_CLUSTER_PATHS: self.data_cluster_paths, CURRICULUM_LEARNING_DATA_CLUSTER_CURRENT_POSITION: self.data_cluster_current_position, CURRICULUM_LEARNING_NP_RNG_STATE: np.random.get_state()}

    def load_state_dict(self, state_dict):
        self.batch = state_dict[CURRICULUM_LEARNING_BATCH]
        self.consumed_samples = state_dict[CURRICULUM_LEARNING_CONSUMED_SAMPLES]
        self.curriculum_step = state_dict[CURRICULUM_LEARNING_STEP]
        self.current_difficulties = state_dict[CURRICULUM_LEARNING_CURRENT_DIFFICULTIES]
        self.data_cluster_paths = state_dict[CURRICULUM_LEARNING_DATA_CLUSTER_PATHS]
        self.data_cluster_current_position = state_dict[CURRICULUM_LEARNING_DATA_CLUSTER_CURRENT_POSITION]
        np.random.set_state(state_dict[CURRICULUM_LEARNING_NP_RNG_STATE])
        cluster_root_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_CLUSTER_PATH]
        for idx in range(len(self.data_cluster_paths)):
            if '/' in self.data_cluster_paths[idx]:
                self.data_cluster_paths[idx] = self.data_cluster_paths[idx].split('/')[-1]
        if self.global_rank == 0:
            for cluster_fname in self.data_cluster_paths:
                cluster_path = f'{cluster_root_path}/{cluster_fname}'
                self.data_clusters.append(MMapIndexedDataset(cluster_path, skip_warmup=True))
                self.data_cluster_sizes.append(len(self.data_clusters[-1][0]))


GLOBAL_RANK = 'global_rank'


class DeepSpeedDataLoader(object):

    def __init__(self, dataset, batch_size, pin_memory, local_rank, tput_timer, collate_fn=None, num_local_io_workers=None, data_sampler=None, data_parallel_world_size=None, data_parallel_rank=None, dataloader_drop_last=False, deepspeed_dataloader_config={}):
        self.deepspeed_dataloader_config = deepspeed_dataloader_config
        self.tput_timer = tput_timer
        self.batch_size = batch_size
        self.curriculum_learning_enabled = False
        if CURRICULUM_LEARNING in deepspeed_dataloader_config:
            self.curriculum_learning_enabled = deepspeed_dataloader_config[CURRICULUM_LEARNING]
        if self.curriculum_learning_enabled:
            data_sampler = DeepSpeedDataSampler(self.deepspeed_dataloader_config[DATA_EFFICIENCY], len(dataset), self.batch_size, data_parallel_rank, data_parallel_world_size, self.deepspeed_dataloader_config[DATA_PARALLEL_GROUP], self.deepspeed_dataloader_config[GRADIENT_ACCUMULATION_STEPS], self.deepspeed_dataloader_config[GLOBAL_RANK], drop_last=dataloader_drop_last)
            device_count = torch.cuda.device_count()
            num_local_io_workers = self.deepspeed_dataloader_config[DATA_SAMPLING_NUM_WORKERS]
        else:
            if local_rank >= 0:
                if data_sampler is None:
                    data_sampler = DistributedSampler(dataset=dataset, num_replicas=data_parallel_world_size, rank=data_parallel_rank)
                device_count = 1
            else:
                if data_sampler is None:
                    data_sampler = RandomSampler(dataset)
                device_count = torch.cuda.device_count()
                batch_size *= device_count
            if num_local_io_workers is None:
                num_local_io_workers = 2 * device_count
        self.num_local_io_workers = num_local_io_workers
        self.data_sampler = data_sampler
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.device_count = device_count
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.data = None
        self.dataloader_drop_last = dataloader_drop_last
        self.post_process_func = None
        if self.dataloader_drop_last:
            self.len = len(self.data_sampler) // self.batch_size
        else:
            from math import ceil
            self.len = ceil(len(self.data_sampler) / self.batch_size)

    def __iter__(self):
        self._create_dataloader()
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        if self.tput_timer:
            self.tput_timer.start()
        if self.curriculum_learning_enabled:
            data = next(self.data_iterator)
            if self.post_process_func is not None:
                data = self.post_process_func(data, self.data_sampler.state_dict())
            return data
        else:
            return next(self.data)

    def _create_dataloader(self):
        if self.curriculum_learning_enabled:
            self.dataloader = DataLoader(self.dataset, pin_memory=self.pin_memory, batch_sampler=self.data_sampler, num_workers=self.num_local_io_workers)
            self.data_iterator = iter(self.dataloader)
            return self.dataloader
        else:
            if self.collate_fn is None:
                self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=self.pin_memory, sampler=self.data_sampler, num_workers=self.num_local_io_workers, drop_last=self.dataloader_drop_last)
            else:
                self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=self.pin_memory, sampler=self.data_sampler, collate_fn=self.collate_fn, num_workers=self.num_local_io_workers, drop_last=self.dataloader_drop_last)
            self.data = (x for x in self.dataloader)
            return self.dataloader


class ZeroParamStatus(Enum):
    AVAILABLE = 1
    NOT_AVAILABLE = 2
    INFLIGHT = 3


def instrument_w_nvtx(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""
    if hasattr(torch.cuda.nvtx, 'range'):

        def wrapped_fn(*args, **kwargs):
            with torch.cuda.nvtx.range(func.__qualname__):
                return func(*args, **kwargs)
        return wrapped_fn
    else:
        return func


class AllGatherCoalescedHandle:

    def __init__(self, allgather_handle, params: List[Parameter], partitions: List[Tensor], world_size: int) ->None:
        self.__allgather_handle = allgather_handle
        self.__params = params
        self.__partitions = partitions
        self.__world_size = world_size
        self.__complete = False
        for param in self.__params:
            if param.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(f'expected param {param.ds_summary()} to not be available')

    @instrument_w_nvtx
    def wait(self) ->None:
        if self.__complete:
            return
        instrument_w_nvtx(self.__allgather_handle.wait)()
        param_offset = 0
        for param in self.__params:
            assert param.ds_status == ZeroParamStatus.INFLIGHT, f'expected param {param.ds_summary()} to be inflight'
            partitions: List[Tensor] = []
            for rank in range(self.__world_size):
                param_start = rank * param.ds_tensor.ds_numel
                if param_start < param.ds_numel:
                    part_to_copy = self.__partitions[rank].narrow(0, param_offset, min(param.ds_numel - param_start, param.ds_tensor.ds_numel))
                    partitions.append(part_to_copy)
            param.data = instrument_w_nvtx(torch.cat)(partitions).view(param.ds_shape)
            param.ds_status = ZeroParamStatus.AVAILABLE
            for part_to_copy in partitions:
                part_to_copy.record_stream(torch.cuda.current_stream())
            param_offset += param.ds_tensor.ds_numel
        self.__complete = True


class AllGatherHandle:

    def __init__(self, handle, param: Parameter) ->None:
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f'expected param {param.ds_summary()} to be available')
        self.__handle = handle
        self.__param = param

    def wait(self) ->None:
        instrument_w_nvtx(self.__handle.wait)()
        self.__param.ds_status = ZeroParamStatus.AVAILABLE


AIO_ALIGNED_BYTES = 1024


MIN_AIO_BYTES = 1024 ** 2


class PartitionedParamStatus(Enum):
    AVAILABLE = 1
    NOT_AVAILABLE = 2
    INFLIGHT = 3


class SwapBuffer(object):

    def __init__(self, buffer):
        self.buffer = buffer
        self.reset()

    def reset(self):
        self.offset = 0
        self.swap_tensors = {}
        self.compute_tensors = {}
        self.swap_paths = {}
        self.num_elem = 0

    def insert_tensor(self, tensor, swap_path, aligned_numel):
        swap_tensor, compute_tensor = self.allocate_tensor(swap_path, tensor.numel(), aligned_numel)
        compute_tensor.data.copy_(tensor.data)
        return swap_tensor, compute_tensor

    def allocate_tensor(self, swap_path, numel, aligned_numel):
        assert self.has_space(aligned_numel)
        assert not self.offset in self.swap_tensors
        allocate_offset = self.offset
        swap_tensor = self.buffer.narrow(0, allocate_offset, aligned_numel)
        dest_tensor = swap_tensor.narrow(0, 0, numel)
        self.swap_tensors[allocate_offset] = swap_tensor
        self.compute_tensors[allocate_offset] = dest_tensor
        self.swap_paths[allocate_offset] = swap_path
        self.offset += aligned_numel
        self.num_elem += numel
        return self.swap_tensors[allocate_offset], self.compute_tensors[allocate_offset]

    def has_space(self, numel):
        return self.offset + numel <= self.buffer.numel()

    def get_swap_tensors(self):
        return [tensor for tensor in self.swap_tensors.values()]

    def get_swap_paths(self):
        return [path for path in self.swap_paths.values()]

    def get_compute_tensors(self):
        return [tensor for tensor in self.compute_tensors.values()]

    def get_num_elem(self):
        return self.num_elem

    def get_swap_tensor(self, offset):
        return self.swap_tensors.get(offset, None)

    def get_compute_tensor(self, offset):
        return self.compute_tensors.get(offset, None)

    def get_swap_path(self, offset):
        return self.swap_paths(offset, None)


def swap_in_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert swap_handle.async_pread(buffer, path) == 0


def swap_out_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert swap_handle.async_pwrite(buffer, path) == 0


class SwapBufferPool(object):

    def __init__(self, buffers):
        assert all([buf.is_pinned() for buf in buffers])
        self.buffers = [SwapBuffer(buf) for buf in buffers]
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        for buffer in self.buffers:
            buffer.reset()

    def allocate_tensor(self, numel, swap_path, aligned_numel):
        if self.has_space(aligned_numel):
            swap_tensor, compute_tensor = self._get_current_buffer().allocate_tensor(swap_path, numel, aligned_numel)
            return swap_tensor, compute_tensor
        return None, None

    def insert_tensor(self, tensor, swap_path, aligned_numel):
        if self.has_space(aligned_numel):
            swap_tensor, compute_tensor = self._get_current_buffer().insert_tensor(tensor, swap_path, aligned_numel)
            return swap_tensor, compute_tensor
        return None, None

    def get_swap_tensors(self):
        swap_tensors = []
        for buffer in self._get_used_buffers():
            swap_tensors += buffer.get_swap_tensors()
        return swap_tensors

    def get_swap_paths(self):
        swap_paths = []
        for buffer in self._get_used_buffers():
            swap_paths += buffer.get_swap_paths()
        return swap_paths

    def get_compute_tensors(self):
        compute_tensors = []
        for buffer in self._get_used_buffers():
            compute_tensors += buffer.get_compute_tensors()
        return compute_tensors

    def has_space(self, numel):
        if self._get_current_buffer().has_space(numel):
            return True
        if self.current_index == len(self.buffers) - 1:
            return False
        self.current_index += 1
        return self._get_current_buffer().has_space(numel)

    def swap_out(self, aio_handle, async_op=False):
        swap_tensors = self.get_swap_tensors()
        swap_paths = self.get_swap_paths()
        assert all([(p is not None) for p in swap_paths])
        swap_out_tensors(aio_handle, swap_tensors, swap_paths)
        if not async_op:
            assert len(swap_tensors) == aio_handle.wait()

    def swap_in(self, aio_handle, async_op=False):
        swap_tensors = self.get_swap_tensors()
        swap_paths = self.get_swap_paths()
        assert all([(p is not None) for p in swap_paths])
        swap_in_tensors(aio_handle, swap_tensors, swap_paths)
        if not async_op:
            assert len(swap_tensors) == aio_handle.wait()

    def _get_current_buffer(self):
        return self.buffers[self.current_index]

    def _get_used_buffers(self):
        return self.buffers[:self.current_index + 1]


def print_object(obj, name, exclude_list=[]):
    logger.info('{}:'.format(name))
    for arg in sorted(vars(obj)):
        if not arg in exclude_list:
            dots = '.' * (29 - len(arg))
            logger.info('  {} {} {}'.format(arg, dots, getattr(obj, arg)))


class AsyncPartitionedParameterSwapper(object):

    def __init__(self, ds_config, model_dtype):
        aio_op = AsyncIOBuilder().load(verbose=False)
        self.aio_handle = aio_op.aio_handle
        self.dtype = model_dtype
        self._configure_aio(ds_config)
        self.id_to_path = {}
        self.param_id_to_buffer_id = {}
        self.param_id_to_swap_buffer = {}
        self.param_id_to_numel = {}
        self.pending_writes = 0
        self.pending_reads = 0
        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0
        self.available_params = set()
        self.available_numel = 0
        self.partitioned_swap_buffer = None
        self.partitioned_swap_pool = None
        self.invalid_buffer = torch.tensor(1).half()
        if dist.get_rank() == 0:
            exclude_list = ['aio_read_handle', 'aio_write_handle', 'buffers']
            print_object(obj=self, name='AsyncPartitionedParameterSwapper', exclude_list=exclude_list)

    def available_swap_in_buffers(self):
        return len(self.available_buffer_ids)

    def _configure_aio(self, ds_config):
        self.swap_config = ds_config.zero_config.offload_param
        torch_dtype_string = str(self.dtype).split('.')[1]
        self.swap_folder = os.path.join(self.swap_config.nvme_path, 'zero_stage_3', f'{torch_dtype_string}params', f'rank{dist.get_rank()}')
        shutil.rmtree(self.swap_folder, ignore_errors=True)
        os.makedirs(self.swap_folder, exist_ok=True)
        self.swap_element_size = torch.tensor([], dtype=self.dtype).element_size()
        self.aio_config = ds_config.aio_config
        self.min_aio_bytes = max(MIN_AIO_BYTES, self.aio_config[AIO_BLOCK_SIZE])
        self.aligned_bytes = AIO_ALIGNED_BYTES * self.aio_config[AIO_THREAD_COUNT]
        self.numel_alignment = self.aligned_bytes // self.swap_element_size
        self.elements_per_buffer = self.swap_config.buffer_size
        self.aligned_elements_per_buffer = self._io_aligned_numel(self.elements_per_buffer)
        self.param_buffer_count = self.swap_config.buffer_count
        self.available_buffer_ids = [i for i in range(self.param_buffer_count)]
        self.reserved_buffer_ids = []
        self.buffers = torch.empty(int(self.aligned_elements_per_buffer * self.param_buffer_count), dtype=self.dtype, pin_memory=True, requires_grad=False)
        self.aio_read_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE], self.aio_config[AIO_QUEUE_DEPTH], self.aio_config[AIO_SINGLE_SUBMIT], self.aio_config[AIO_OVERLAP_EVENTS], self.aio_config[AIO_THREAD_COUNT])
        self.aio_write_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE], self.aio_config[AIO_QUEUE_DEPTH], self.aio_config[AIO_SINGLE_SUBMIT], self.aio_config[AIO_OVERLAP_EVENTS], self.aio_config[AIO_THREAD_COUNT])
        self.swap_out_params = []

    def swappable_tensor(self, param=None, numel=None):
        if param is not None:
            assert numel is None, 'Both parma and numel cannot be provided'
            numel = param.ds_tensor.ds_numel
        if numel is not None:
            return self.min_aio_bytes <= numel * self.swap_element_size
        assert False, 'Either param or numel must be provided'

    def get_path(self, param, must_exist=False):
        paths = self._get_swap_paths([param], must_exist=must_exist)
        return paths[0]

    def _get_swap_paths(self, params, must_exist=False):
        paths = []
        for param in params:
            param_id = param.ds_id
            if param_id in self.id_to_path.keys():
                param_path = self.id_to_path[param_id]
            else:
                assert not must_exist, f'Path for param id {param_id} does not exist'
                param_path = os.path.join(self.swap_folder, f'{param_id}_param.tensor.swp')
                self.id_to_path[param_id] = param_path
            paths.append(param_path)
        return paths

    def _get_swap_buffers(self, params):
        buffers = []
        for param in params:
            param_id = param.ds_id
            assert param_id in self.param_id_to_swap_buffer.keys(), f'param {param_id} has not been assigned a swap buffer'
            buffers.append(self.param_id_to_swap_buffer[param_id])
        return buffers

    def _track_numel(self, params):
        for param in params:
            assert param.ds_tensor is not None, 'Partitioned tensor is None'
            self.param_id_to_numel[param.ds_id] = param.ds_tensor.ds_numel

    def _allocate_and_return_buffers_for_swap_in(self, params):
        compute_buffers = []
        swap_buffers = []
        for param in params:
            param_id = param.ds_id
            assert param_id in self.param_id_to_numel.keys(), f' Number of elements in param {param_id} is unknown'
            assert param_id not in self.param_id_to_buffer_id.keys(), f'param {param_id} already assigned swap buffer id {self.param_id_to_buffer_id[param_id]}'
            assert param_id not in self.param_id_to_swap_buffer.keys(), f'param {param_id} has already been assigned a swap buffer'
            buffer_id = self.available_buffer_ids.pop()
            print_rank_0(f'param {param.ds_id} is assigned swap in buffer id {buffer_id}  ')
            self.param_id_to_buffer_id[param_id] = buffer_id
            aligned_swap_numel = self._io_aligned_numel(self.param_id_to_numel[param_id])
            swap_buffer = self.buffers.narrow(0, int(buffer_id * self.aligned_elements_per_buffer), aligned_swap_numel)
            self.param_id_to_swap_buffer[param_id] = swap_buffer
            compute_buffer = swap_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
            compute_buffers.append(compute_buffer)
            swap_buffers.append(swap_buffer)
        return compute_buffers, swap_buffers

    def synchronize_writes(self):
        if self.pending_writes == 0:
            return
        assert self.pending_writes == self.aio_write_handle.wait()
        self.pending_writes = 0
        self.remove_partition_and_release_buffers(self.swap_out_params)
        self.swap_out_params = []

    def synchronize_reads(self):
        if self.pending_reads == 0:
            return
        assert self.pending_reads == self.aio_read_handle.wait()
        self.pending_reads = 0
        for param, swap_in_buffer in zip(self.inflight_params, self.inflight_swap_in_buffers):
            param_id = param.ds_id
            compute_buffer = swap_in_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
            param.ds_tensor.data = compute_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
        self.available_params.update([param.ds_id for param in self.inflight_params])
        self.available_numel += self.inflight_numel
        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0

    def remove_partition_and_release_buffers(self, params):
        for param in params:
            param_id = param.ds_id
            if param_id in self.param_id_to_buffer_id.keys():
                buffer_id = self.param_id_to_buffer_id[param_id]
                assert buffer_id is not None, 'Missing buffer id for releasing'
                self.available_buffer_ids.append(buffer_id)
                del self.param_id_to_buffer_id[param_id]
                del self.param_id_to_swap_buffer[param_id]
                print_rank_0(f'param {param.ds_id} releases buffer id {buffer_id}  ')
                if param_id in self.available_params:
                    self.available_params.remove(param_id)
                    self.available_numel -= self.param_id_to_numel[param_id]
            param.ds_tensor.data = self.invalid_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE

    def _swap_out(self, params, async_op=True):
        swap_out_paths = self._get_swap_paths(params)
        swap_out_params = self._get_swap_buffers(params)
        self._track_numel(params)
        swap_out_tensors(self.aio_write_handle, swap_out_params, swap_out_paths)
        self.pending_writes += len(swap_out_params)
        self.swap_out_params += params
        if not async_op:
            self.synchronize_writes()

    def swap_out_and_release(self, params, async_op=False, force_buffer_release=False):
        if async_op:
            assert force_buffer_release, 'Should not release preallocated buffers without completing the swap out. Set force_buffer_release to True to do it anyways'
        self._swap_out(params, async_op=async_op)

    def _update_inflight_swap_in(self, params, swap_in_buffers, inflight_numel):
        self.inflight_params.extend(params)
        self.inflight_swap_in_buffers.extend(swap_in_buffers)
        self.inflight_numel += inflight_numel
        for param in params:
            param.ds_tensor.status = PartitionedParamStatus.INFLIGHT
        self.pending_reads += len(params)

    def swap_in(self, params, async_op=True, swap_in_buffers=None):
        assert all([(param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE) for param in params]), 'Some params are already available or in flight'
        swap_in_paths = self._get_swap_paths(params)
        if swap_in_buffers is None:
            if len(self.available_buffer_ids) < len(swap_in_paths):
                ids = [p.ds_id for p in params]
                print_rank_0(f'Not enough swap in buffers {len(self.available_buffer_ids)} for {len(swap_in_paths)} params, ids = {ids}', force=True)
                print_rank_0(f'Num inflight: params {len(self.inflight_params)}, buffers {len(self.inflight_swap_in_buffers)}, numel = {self.inflight_numel}', force=True)
                print_rank_0(f'Num available params: count = {len(self.available_params)}, ids = {self.available_params}, numel = {self.available_numel}', force=True)
            assert len(swap_in_paths) <= len(self.available_buffer_ids), f'Not enough buffers {len(self.available_buffer_ids)} for swapping {len(swap_in_paths)}'
            compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in(params)
            inflight_numel = sum([t.numel() for t in compute_buffers])
        else:
            inflight_numel = sum([t.numel() for t in swap_in_buffers])
        swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)
        self._update_inflight_swap_in(params, swap_in_buffers, inflight_numel)
        if not async_op:
            self.synchronize_reads()

    def swap_into_buffer(self, param, dest_buffer):
        assert param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE, f'param {param.ds_id} is already available or inflight'
        require_swap_buffer = not (dest_buffer.is_pinned() and self._is_io_aligned(dest_buffer.numel()))
        if require_swap_buffer:
            assert len(self.available_buffer_ids) > 0, f'No buffer available to swap param {param.ds_id}.'
            compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in([param])
            inflight_numel = compute_buffers[0].numel()
        else:
            swap_in_buffers = [dest_buffer]
            inflight_numel = dest_buffer.numel()
        swap_in_paths = self._get_swap_paths([param])
        swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)
        self._update_inflight_swap_in([param], swap_in_buffers, inflight_numel)
        self.synchronize_reads()
        if require_swap_buffer:
            dest_buffer.data.copy_(param.ds_tensor.data)
            self.remove_partition_and_release_buffers([param])

    def get_buffer(self, param, numel):
        param_id = param.ds_id
        assert self.available_swap_in_buffers() > 0, f'No swap buffers to allocate for fp16 param {param_id} of numel = {numel}'
        assert numel < self.elements_per_buffer, f'More elements {numel} than buffer size {self.elements_per_buffer}'
        self.param_id_to_numel[param_id] = numel
        buffer_id = self.available_buffer_ids.pop()
        self.param_id_to_buffer_id[param_id] = buffer_id
        aligned_swap_numel = self._io_aligned_numel(self.param_id_to_numel[param_id])
        swap_buffer = self.buffers.narrow(0, int(buffer_id * self.aligned_elements_per_buffer), aligned_swap_numel)
        self.param_id_to_swap_buffer[param_id] = swap_buffer
        compute_buffer = swap_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
        print_rank_0(f'param {param.ds_id} is assigned swap in buffer id {buffer_id}')
        return compute_buffer

    def reserve_available_buffers(self):
        buffers = []
        for id in self.available_buffer_ids:
            buffers.append(self.buffers.narrow(0, int(id * self.aligned_elements_per_buffer), int(self.aligned_elements_per_buffer)))
            self.reserved_buffer_ids.append(id)
        self.available_buffer_ids = []
        return buffers

    def release_reserved_buffers(self):
        for id in self.reserved_buffer_ids:
            self.available_buffer_ids.append(id)
        self.reserved_buffer_ids = []

    def _io_aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else numel + self.numel_alignment - remainder

    def _is_io_aligned(self, numel):
        return numel % self.numel_alignment == 0

    def reserve_partitioned_swap_space(self, partition_num_elems):
        aligned_numel = sum([self._io_aligned_numel(numel) for numel in partition_num_elems])
        self.partitioned_swap_buffer = torch.zeros(aligned_numel, device='cpu', dtype=self.dtype).pin_memory()
        self.partitioned_swap_pool = SwapBufferPool([self.partitioned_swap_buffer])

    def swap_out_partitioned_params(self, dst_fp16_params, src_fp32_params):
        assert self.partitioned_swap_buffer is not None, f'partitioned swap buffers for fp16 params not initialized'
        assert self.partitioned_swap_pool is not None, f'partitioned swap pool for fp16 params not initialized'
        assert len(dst_fp16_params) == len(src_fp32_params), f'mismatch in number of fp16 params {len(dst_fp16_params)} and fp32 params {len(src_fp32_params)}'
        fp16_swap_paths = self._get_swap_paths(dst_fp16_params, must_exist=True)
        self.synchronize_writes()
        self.partitioned_swap_pool.reset()
        for i, fp32_tensor in enumerate(src_fp32_params):
            swap_tensor, _ = self.partitioned_swap_pool.insert_tensor(fp32_tensor, fp16_swap_paths[i], self._io_aligned_numel(fp32_tensor.numel()))
            assert swap_tensor is not None
            dst_fp16_params[i].ds_tensor.status = PartitionedParamStatus.AVAILABLE
        self.partitioned_swap_pool.swap_out(self.aio_write_handle)
        for param in dst_fp16_params:
            param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE


_orig_torch_empty = torch.empty


_orig_torch_full = torch.full


_orig_torch_ones = torch.ones


_orig_torch_zeros = torch.zeros


def get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)
    recurse(cls)
    return set(subclass_list)


def get_new_tensor_fn_for_dtype(dtype: torch.dtype) ->Callable:

    def new_tensor(cls, *args) ->Tensor:
        device = torch.device('cuda:{}'.format(os.environ['LOCAL_RANK']))
        tensor = _orig_torch_empty(0, device=device).new_empty(*args)
        if tensor.is_floating_point():
            tensor = tensor
        return tensor
    return new_tensor


def is_zero_param(parameter):
    if not torch.is_tensor(parameter):
        return False
    return hasattr(parameter, 'ds_id')


param_count = 0


def shutdown_init_context():
    global zero_init_enabled
    if not zero_init_enabled:
        return

    def _disable_class(cls):
        cls.__init__ = cls._old_init
    for subclass in get_all_subclasses(torch.nn.modules.module.Module):
        _disable_class(subclass)
    torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass
    torch.nn.modules.module.Module.apply = torch.nn.modules.module.Module._old_apply
    torch.Tensor.__new__ = torch.Tensor.__old_new__
    torch.empty = _orig_torch_empty
    torch.zeros = _orig_torch_zeros
    torch.ones = _orig_torch_ones
    torch.full = _orig_torch_full
    zero_init_enabled = False


tensor_map = {}


def zero3_linear_wrap(input, weight, bias=None):
    if bias is None:
        return LinearFunctionForZeroStage3.apply(input, weight)
    else:
        return LinearFunctionForZeroStage3.apply(input, weight, bias)


def zero_wrapper_for_fp_tensor_constructor(fn: Callable, target_fp_dtype: torch.dtype) ->Callable:

    def wrapped_fn(*args, **kwargs) ->Tensor:
        if kwargs.get('device', None) is None:
            kwargs['device'] = torch.device('cuda:{}'.format(os.environ['LOCAL_RANK']))
        tensor: Tensor = fn(*args, **kwargs)
        if tensor.is_floating_point():
            tensor = tensor
        return tensor
    return wrapped_fn


class InsertPostInitMethodToModuleSubClasses(object):

    def __init__(self, enabled=True, mem_efficient_linear=True, ds_config=None, dtype=None):
        self.mem_efficient_linear = mem_efficient_linear
        self.enabled = enabled
        self._set_dtype(ds_config, dtype)
        assert self.dtype in [torch.half, torch.bfloat16, torch.float], f'Invalid data type {self.dtype}, allowed values are [torch.half, torch.bfloat16, torch.float]'

    def __enter__(self):
        global zero_init_enabled
        if not self.enabled:
            return
        zero_init_enabled = True

        def apply_with_gather(orig_module_apply_fn: Callable) ->Callable:
            """many models make use of child modules like Linear or Embedding which
            perform their own weight initialization in their __init__ methods,
            but will then have more weight initialization in a parent module's __init__
            method that modifies weights of child modules, which is typically done
            using the Module.apply method.

            since the Init context manager partitions child modules immediately after
            they are initialized, without modifying apply we would entirely skip
            any initialization done by parent modules.

            to get around this issue, we wrap the function passed to Module.apply
            so that the applied function is applied to child modules correctly.
            """

            def get_wrapped_fn_to_apply(fn_to_apply: Callable) ->Callable:
                if hasattr(fn_to_apply, 'wrapped'):
                    return fn_to_apply

                @functools.wraps(fn_to_apply)
                def wrapped_fn_to_apply(module_to_apply_fn_to: Module) ->None:
                    """gathers parameters before calling apply function. afterwards
                    parameters are broadcasted to ensure consistency across all ranks
                    then re-partitioned.

                    takes the following steps:
                    1. allgathers parameters for the current module being worked on
                    2. calls the original function
                    3. broadcasts root rank's parameters to the other ranks
                    4. re-partitions the parameters
                    """
                    if not all(is_zero_param(p) for p in module_to_apply_fn_to.parameters(recurse=False)):
                        raise RuntimeError(f'not all parameters for {module_to_apply_fn_to.__class__.__name__}, were zero params, is it possible that the parameters were overwritten after they were initialized? params: {[p for p in module_to_apply_fn_to.parameters(recurse=False)]} ')
                    params_to_apply_fn_to: Iterable[Parameter] = list(sorted(module_to_apply_fn_to.parameters(recurse=False), key=lambda p: p.ds_id))
                    for param in params_to_apply_fn_to:
                        param.all_gather()
                    fn_to_apply(module_to_apply_fn_to)
                    for param in params_to_apply_fn_to:
                        dist.broadcast(param.data, 0, group=param.ds_process_group)
                    for param in params_to_apply_fn_to:
                        param.partition(has_been_updated=True)
                wrapped_fn_to_apply.wrapped = True
                return wrapped_fn_to_apply

            @functools.wraps(orig_module_apply_fn)
            def wrapped_apply(module: Module, fn_to_apply: Callable) ->None:
                orig_module_apply_fn(module, get_wrapped_fn_to_apply(fn_to_apply))
            return wrapped_apply

        def partition_after(f):

            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):
                print_rank_0(f'Before initializing {module.__class__.__name__}', force=False)
                is_child_module = False
                if not hasattr(module, '_ds_child_entered'):
                    is_child_module = True
                    setattr(module, '_ds_child_entered', True)
                f(module, *args, **kwargs)
                if is_child_module:
                    delattr(module, '_ds_child_entered')
                    print_rank_0(f'Running post_init for {module.__class__.__name__}', force=False)
                    self._post_init_method(module)
                print_rank_0(f'After initializing followed by post init for {module.__class__.__name__}', force=False)
            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = partition_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = partition_after(cls.__init__)
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            _enable_class(subclass)
        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
        torch.nn.modules.module.Module._old_apply = torch.nn.modules.module.Module.apply
        torch.Tensor.__old_new__ = torch.Tensor.__new__
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
        torch.nn.modules.module.Module.apply = apply_with_gather(torch.nn.modules.module.Module._old_apply)
        torch.Tensor.__new__ = get_new_tensor_fn_for_dtype(self.dtype)
        torch.empty = zero_wrapper_for_fp_tensor_constructor(_orig_torch_empty, self.dtype)
        torch.zeros = zero_wrapper_for_fp_tensor_constructor(_orig_torch_zeros, self.dtype)
        torch.ones = zero_wrapper_for_fp_tensor_constructor(_orig_torch_ones, self.dtype)
        torch.full = zero_wrapper_for_fp_tensor_constructor(_orig_torch_full, self.dtype)
        if self.mem_efficient_linear:
            print_rank_0('nn.functional.linear has been overridden with a more memory efficient version. This will persist unless manually reset.', force=False)
            self.linear_bk = torch.nn.functional.linear
            torch.nn.functional.linear = zero3_linear_wrap

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return
        shutdown_init_context()
        if dist.get_rank() == 0:
            logger.info('finished initializing model with %.2fB parameters', param_count / 1000000000.0)
        if exc_type is not None:
            return False

    def _post_init_method(self, module):
        pass

    def _set_dtype(self, ds_config, dtype):
        if ds_config is not None and dtype is None:
            if ds_config.bfloat16_enabled and ds_config.fp16_enabled:
                raise RuntimeError('bfloat16 and fp16 cannot be enabled at once')
            if ds_config.bfloat16_enabled:
                self.dtype = torch.bfloat16
            elif ds_config.fp16_enabled:
                self.dtype = torch.half
            else:
                self.dtype = torch.float
        else:
            self.dtype = dtype or torch.half


class ZeroParamType(Enum):
    NORMAL = 1
    PARTITIONED = 2
    REMOTE = 3


def _dist_allgather_fn(input_tensor: Tensor, output_tensor: Tensor, group=None):
    return instrument_w_nvtx(dist.allgather_fn)(output_tensor, input_tensor, group=group, async_op=True)


def get_lst_from_rank0(lst: List[int]) ->None:
    """
    NOTE: creates both communication and synchronization overhead so should be used
    sparingly
    """
    lst_tensor = torch.tensor(lst if dist.get_rank() == 0 else [-1] * len(lst), dtype=int, device=torch.device('cuda:{}'.format(os.environ['LOCAL_RANK'])), requires_grad=False)
    dist.broadcast(lst_tensor, src=0, async_op=False)
    return list(lst_tensor.cpu().numpy())


@instrument_w_nvtx
def assert_ints_same_as_other_ranks(ints: List[int]) ->None:
    """
    NOTE: creates both communication and synchronization overhead so should be
    used sparingly

    takes a list of ints from each rank and ensures that they are the same
    across ranks, throwing an exception if they are not.
    """
    rank0_ints = get_lst_from_rank0(ints)
    if ints != rank0_ints:
        raise RuntimeError(f'disagreement between rank0 and rank{dist.get_rank()}: rank0: {rank0_ints}, rank{dist.get_rank()}: {ints}')


module_names = {}


def debug_module2name(module):
    if module in module_names:
        return module_names[module]
    else:
        return 'unknown'


param_names = {}


def debug_param2name(param):
    if param in param_names:
        return param_names[param]
    else:
        return 'unknown'


def debug_param2name_id(param):
    return f'name={debug_param2name(param)} id={param.ds_id}'


def debug_param2name_id_shape(param):
    return f'name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape}'


def debug_param2name_id_shape_device(param):
    return f'name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape} device={param.device}'


def debug_param2name_id_shape_status(param):
    return f'name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape} status={param.ds_status}'


def debug_rank0(message: str) ->None:
    if dist.get_rank() == 0:
        logger.debug(message)


@instrument_w_nvtx
def free_param(param: Parameter) ->None:
    """Free underlying storage of a parameter."""
    assert not param.ds_active_sub_modules, param.ds_summary()
    if param.data.is_cuda:
        param.data.record_stream(torch.cuda.current_stream())
    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    param.ds_status = ZeroParamStatus.NOT_AVAILABLE


def get_only_unique_item(items):
    item_set = set(items)
    if len(item_set) != 1:
        raise RuntimeError(f'expected there to be only one unique element in {items}')
    unique_item, = item_set
    return unique_item


class Init(InsertPostInitMethodToModuleSubClasses):
    param_id = 0

    def __init__(self, module=None, data_parallel_group=None, mem_efficient_linear=True, remote_device=None, pin_memory=False, config_dict_or_path=None, config=None, enabled=True, dtype=None, mpu=None):
        """A context to enable massive model construction for training with
        ZeRO-3. Models are automatically partitioned (or, sharded) across the
        system and converted to half precision.

        Args:
            module (``torch.nn.Module``, optional): If provided, partition the model as
                if it was constructed in the context.
            data_parallel_group (``deepspeed.comm`` process group, optional):
                The group of processes to partition among. Defaults to all processes.
            mem_efficient_linear (bool, optional): Replace
                torch.nn.functional.linear with an implementation that allows
                DeepSpeed to partition parameters. Defaults to ``True``.
            remote_device (string, optional): The initial device to store model
                weights e.g., ``cpu``, ``nvme``. Passing ``"cpu"`` will create the model in CPU
                memory. The model may still be moved to GPU based on the
                offload settings for training. Defaults to param offload device if a config is
                defined, otherwise GPU.
            pin_memory (bool, optional): Potentially increase performance by
                using pinned memory for model weights. ``remote_device`` must be
                ``"cpu"``. Defaults to pin_memory value in config, otherwise ``False``.
            config_dict_or_path (dict or ``json file``, optional): If provided, provides configuration
                for swapping fp16 params to NVMe.
            config (dict or ``json file``, optional): Deprecated, use config_dict_or_path instead.
            enabled (bool, optional): If ``False``, this context has no
                effect. Defaults to ``True``.
            dtype (``dtype``, optional): Can be used to change the data type of the parameters.
                Supported options are ``torch.half`` and ``torch.float``. Defaults to ``None``
            mpu (``object``, optional): A model parallelism unit object that implements get_{model,data}_parallel_{rank,group,world_size}.

        This context accelerates model initialization and enables models that
        are too large to allocate in their entirety in CPU memory. It has the
        following effects:

        #. allocates tensors to either GPU or CPU memory or NVMe
        #. converts floating point tensors to half precision
        #. immediately partitions tensors among the group of data-parallel devices
        #. (*optional*) replaces ``torch.nn.functional.linear`` with a more
           memory-efficient implementation

        These modifications allow for models that exceed the size of local CPU/GPU
        memory/NVMe, but fit within the total NVMe capacity (*i.e.*, aggregate CPU
        or GPU memory or NVMe) across all nodes. Consider initializing a model with one
        trillion parameters, whose weights occupy two terabytes (TB) in half
        precision. The initial CPU allocation in full precision requires 4TB of
        memory *per process*, and so a system with 8 GPUs per node would need 32TB of
        CPU memory due to data-parallel redundancies. Instead, by immediately
        partitioning tensors we remove the redundancies. The result is that
        regardless of the number of GPUs, we still only require the original 4TB. This
        allows for a linear increase in model size with the aggregate system memory.
        For example, if a node has 1TB of memory and 8 GPUs, we could fit a trillion
        parameter model with 4 nodes and 32 GPUs.

        Important: If the fp16 weights of the model can't fit onto a single GPU memory
        this feature must be used.

        .. note::
            Initializes ``deepspeed.comm`` if it has not already been done so.
            See :meth:`deepspeed.init_distributed` for more information.

        .. note::
            Can also be used as a decorator:

            .. code-block:: python

                @deepspeed.zero.Init()
                def get_model():
                    return MyLargeModel()

        .. note::
            Only applicable to training with ZeRO-3.

        Examples
        --------

        #. Allocate a model and partition it among all processes:

            .. code-block:: python

                with deepspeed.zero.Init():
                    model = MyLargeModel()


        #. Allocate a model in pinned CPU memory and partition it among a subgroup of processes:

            .. code-block:: python

                with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                                         remote_device="cpu",
                                         pin_memory=True):
                    model = MyLargeModel()


        #. Partition an already-allocated model in CPU memory:

            .. code-block:: python

                model = deepspeed.zero.Init(module=model)
        """
        if config is not None:
            config_dict_or_path = config
            logger.warning(f'zero.Init: the `config` argument is deprecated. Please use `config_dict_or_path` instead.')
        _ds_config = deepspeed.runtime.config.DeepSpeedConfig(config_dict_or_path, mpu) if config_dict_or_path is not None else None
        super().__init__(enabled=enabled, mem_efficient_linear=mem_efficient_linear, ds_config=_ds_config, dtype=dtype)
        if not dist.is_initialized():
            init_distributed()
            assert dist.is_initialized(), 'Parameters cannot be scattered without initializing deepspeed.comm'
        if data_parallel_group is None:
            self.ds_process_group = dist.get_world_group()
        else:
            self.ds_process_group = data_parallel_group
        self.rank = dist.get_rank(group=self.ds_process_group)
        self.world_size = dist.get_world_size(group=self.ds_process_group)
        self.local_device = torch.device('cuda:{}'.format(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(self.local_device)
        if _ds_config is not None and _ds_config.zero_config.offload_param is not None:
            remote_device = _ds_config.zero_config.offload_param.device
            pin_memory = _ds_config.zero_config.offload_param.pin_memory
        self._validate_remote_device(remote_device, _ds_config)
        self.remote_device = self.local_device if remote_device in [None, OffloadDeviceEnum.none] else remote_device
        self.pin_memory = pin_memory if self.remote_device in [OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme] else False
        if self.remote_device == OffloadDeviceEnum.nvme:
            self.param_swapper = AsyncPartitionedParameterSwapper(_ds_config, self.dtype)
        else:
            self.param_swapper = None
        if module is not None:
            assert isinstance(module, torch.nn.Module)
            self._convert_to_zero_parameters(module.parameters(recurse=True))
        self.use_all_gather_base = False
        if dist.has_allgather_base():
            self.use_all_gather_base = True
        else:
            logger.info(f'_all_gather_base API is not available in torch {torch.__version__}')

    def _convert_to_zero_parameters(self, param_list):
        for param in param_list:
            if is_zero_param(param):
                continue
            self._convert_to_deepspeed_param(param)
            param.partition()

    def _validate_remote_device(self, remote_device, ds_config):
        if ds_config is not None:
            if remote_device in [None, OffloadDeviceEnum.cpu]:
                if ds_config.zero_config.offload_param is not None:
                    offload_param_device = ds_config.zero_config.offload_param.device
                    assert offload_param_device != OffloadDeviceEnum.nvme, f"'device' in DeepSpeed Config cannot be {offload_param_device} if remote device is {remote_device}."
            if remote_device == OffloadDeviceEnum.nvme:
                assert ds_config.zero_config.offload_param is not None, f'"offload_param" must be defined in DeepSpeed Config if remote device is {OffloadDeviceEnum.nvme}.'
                assert ds_config.zero_config.offload_param.nvme_path is not None, f'"nvme_path" in DeepSpeed Config cannot be None if remote device is {OffloadDeviceEnum.nvme}'

    def _post_init_method(self, module):
        print_rank_0(f'Converting Params in {module.__class__.__name__}', force=False)
        see_memory_usage(f'Before converting and partitioning parmas in {module.__class__.__name__}', force=False)
        global param_count
        for name, param in module.named_parameters(recurse=False):
            param_count += param.numel()
            if not is_zero_param(param):
                self._convert_to_deepspeed_param(param)
                print_rank_0(f'Partitioning param {debug_param2name_id_shape(param)} module={debug_module2name(module)}')
                if param.is_cuda:
                    dist.broadcast(param, 0, self.ds_process_group)
                elif dist.get_rank() == 0:
                    logger.warn(f'param `{name}` in {module.__class__.__name__} not on GPU so was not broadcasted from rank 0')
                param.partition()
        see_memory_usage(f'Param count {param_count}. After converting and partitioning parmas in {module.__class__.__name__}', force=False)

    def _convert_to_deepspeed_param(self, param):
        param.ds_param_type = ZeroParamType.PARTITIONED
        param.ds_status = ZeroParamStatus.AVAILABLE
        param.ds_shape = param.shape
        param.ds_numel = param.numel()
        param.ds_tensor = None
        param.ds_active_sub_modules = set()
        param.ds_persist = False
        param.is_external_param = False
        param.ds_process_group = self.ds_process_group
        param.nvme_swapper = self.param_swapper
        param.ds_id = Init.param_id
        Init.param_id += 1

        def all_gather(param_list=None, async_op=False, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            return self._all_gather(param_list, async_op=async_op, hierarchy=hierarchy)

        @instrument_w_nvtx
        def all_gather_coalesced(params: Iterable[Parameter], safe_mode: bool=False) ->AllGatherCoalescedHandle:
            self._ensure_availability_of_partitioned_params(params)
            for param in params:
                if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                    raise RuntimeError(param.ds_summary())
                param.ds_status = ZeroParamStatus.INFLIGHT
            params = sorted(params, key=lambda p: p.ds_id)
            debug_rank0(f'-allgather_coalesced: {[p.ds_id for p in params]}')
            if safe_mode:
                assert_ints_same_as_other_ranks([p.ds_id for p in params])
                assert_ints_same_as_other_ranks([p.ds_tensor.ds_numel for p in params])
            if len(params) == 1:
                param, = params
                param_buffer = torch.empty(math.ceil(param.ds_numel / self.world_size) * self.world_size, dtype=param.dtype, device=torch.cuda.current_device(), requires_grad=False)
                handle = _dist_allgather_fn(param.ds_tensor, param_buffer, self.ds_process_group)
                param.data = param_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape)
                return AllGatherHandle(handle, param)
            else:
                partition_sz = sum(p.ds_tensor.ds_numel for p in params)
                flat_tensor = torch.empty(partition_sz * self.world_size, dtype=get_only_unique_item(p.dtype for p in params), device=torch.cuda.current_device(), requires_grad=False)
                partitions: List[Parameter] = []
                for i in range(self.world_size):
                    partitions.append(flat_tensor.narrow(0, partition_sz * i, partition_sz))
                instrument_w_nvtx(torch.cat)([p.ds_tensor for p in params], out=partitions[self.rank])
                handle = _dist_allgather_fn(partitions[self.rank], flat_tensor, self.ds_process_group)
                return AllGatherCoalescedHandle(allgather_handle=handle, params=params, partitions=partitions, world_size=self.world_size)

        def partition(param_list=None, hierarchy=0, has_been_updated=False):
            cls = param
            print_rank_0(f"{'--' * hierarchy}----Partitioning param {debug_param2name_id_shape_device(cls)}")
            if param_list is None:
                param_list = [cls]
            self._partition(param_list, has_been_updated=has_been_updated)

        def reduce_gradients_at_owner(param_list=None, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            print_rank_0(f"{'--' * hierarchy}----Reducing Gradients for param with ids {[param.ds_id for param in param_list]} to owner")
            self._reduce_scatter_gradients(param_list)

        def partition_gradients(param_list=None, partition_buffers=None, hierarchy=0, accumulate=False):
            cls = param
            print_rank_0(f"{'--' * hierarchy}----Partitioning param gradient with id {debug_param2name_id_shape_device(cls)}")
            if param_list is None:
                param_list = [cls]
                if isinstance(partition_buffers, torch.Tensor):
                    partition_buffers = [partition_buffers]
            self._partition_gradients(param_list, partition_buffers=partition_buffers, accumulate=accumulate)

        def aligned_size():
            return self._aligned_size(param)

        def padding_size():
            return self._padding_size(param)

        def partition_numel():
            return self._partition_numel(param)

        def item_override():
            param.all_gather()
            return param._orig_item()

        def ds_summary(slf: torch.Tensor, use_debug_name: bool=False) ->dict:
            return {'id': debug_param2name_id(slf) if use_debug_name else slf.ds_id, 'status': slf.ds_status.name, 'numel': slf.numel(), 'ds_numel': slf.ds_numel, 'shape': tuple(slf.shape), 'ds_shape': tuple(slf.ds_shape), 'requires_grad': slf.requires_grad, 'grad_shape': tuple(slf.grad.shape) if slf.grad is not None else None, 'persist': slf.ds_persist, 'active_sub_modules': slf.ds_active_sub_modules}

        def convert_to_zero_parameters(param_list):
            self._convert_to_zero_parameters(param_list)

        def allgather_before(func: Callable) ->Callable:

            def wrapped(*args, **kwargs):
                param.all_gather()
                return func(*args, **kwargs)
            return wrapped
        param.all_gather = all_gather
        param.all_gather_coalesced = all_gather_coalesced
        param.partition = partition
        param.reduce_gradients_at_owner = reduce_gradients_at_owner
        param.partition_gradients = partition_gradients
        param.aligned_size = aligned_size
        param.padding_size = padding_size
        param.partition_numel = partition_numel
        param.ds_summary = types.MethodType(ds_summary, param)
        param.item = allgather_before(param.item)
        param.convert_to_zero_parameters = convert_to_zero_parameters

    def _aligned_size(self, param):
        return param.ds_numel + self._padding_size(param)

    def _padding_size(self, param):
        remainder = param.ds_numel % self.world_size
        return self.world_size - remainder if remainder else 0

    def _partition_numel(self, param):
        return param.ds_tensor.ds_numel

    def _ensure_availability_of_partitioned_params(self, params):
        swap_in_list = []
        swap_in_flight = []
        for param in params:
            if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                assert param.ds_tensor.final_location == OffloadDeviceEnum.nvme and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                swap_in_list.append(param)
            if param.ds_tensor.status == PartitionedParamStatus.INFLIGHT:
                assert param.ds_tensor.final_location == OffloadDeviceEnum.nvme and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                swap_in_flight.append(param)
        if len(swap_in_list) > 0:
            swap_in_list[0].nvme_swapper.swap_in(swap_in_list, async_op=False)
        elif len(swap_in_flight) > 0:
            swap_in_flight[0].nvme_swapper.synchronize_reads()

    @instrument_w_nvtx
    def _all_gather(self, param_list, async_op=False, hierarchy=None):
        self._ensure_availability_of_partitioned_params(param_list)
        handles = []
        all_gather_list = []
        for param in param_list:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if async_op:
                    handle = self._allgather_param(param, async_op=async_op, hierarchy=hierarchy)
                    param.ds_status = ZeroParamStatus.INFLIGHT
                    handles.append(handle)
                else:
                    all_gather_list.append(param)
        if not async_op:
            if len(param_list) == 1:
                ret_value = self._allgather_params(all_gather_list, hierarchy=hierarchy)
            else:
                ret_value = self._allgather_params_coalesced(all_gather_list, hierarchy)
            for param in all_gather_list:
                param.ds_status = ZeroParamStatus.AVAILABLE
            return ret_value
        return handles

    def _partition(self, param_list, force=False, has_been_updated=False):
        for param in param_list:
            self._partition_param(param, has_been_updated=has_been_updated)
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE

    @instrument_w_nvtx
    def _partition_param(self, param, buffer=None, has_been_updated=False):
        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f' {param} Cannot partition a param in flight'
        global reuse_buffers
        if param.ds_status is ZeroParamStatus.AVAILABLE:
            print_rank_0(f'Partitioning param id {param.ds_id} reuse buffers {reuse_buffers}', force=False)
            if param.ds_tensor is not None and not has_been_updated:
                see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}', force=False)
                free_param(param)
                see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}', force=False)
                if param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
                    print_rank_0(f'Param {param.ds_id} partition released since it exists in nvme', force=False)
                    param.nvme_swapper.remove_partition_and_release_buffers([param])
                return
            tensor_size = self._aligned_size(param)
            partition_size = tensor_size // self.world_size
            if param.ds_tensor is None:
                final_location = None
                if self.remote_device == OffloadDeviceEnum.nvme and self.param_swapper.swappable_tensor(numel=partition_size):
                    final_location = OffloadDeviceEnum.nvme
                    buffer = self.param_swapper.get_buffer(param, partition_size)
                    partitioned_tensor = torch.empty(0, dtype=param.dtype, device=buffer.device)
                    partitioned_tensor.data = buffer.data
                    print_rank_0(f'ID {param.ds_id} Initializing partition for the first time for nvme offload.')
                else:
                    partitioned_tensor = torch.empty(partition_size, dtype=param.dtype, device=OffloadDeviceEnum.cpu if self.remote_device == OffloadDeviceEnum.nvme else self.remote_device)
                    if self.pin_memory:
                        partitioned_tensor = partitioned_tensor.pin_memory()
                partitioned_tensor.requires_grad = False
                param.ds_tensor = partitioned_tensor
                param.ds_tensor.ds_numel = partition_size
                param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_tensor.final_location = final_location
            start = partition_size * self.rank
            end = start + partition_size
            one_dim_param = param.contiguous().view(-1)
            if start < param.ds_numel and end <= param.ds_numel:
                src_tensor = one_dim_param.narrow(0, start, partition_size)
                param.ds_tensor.copy_(src_tensor)
            elif start < param.ds_numel:
                elements_to_copy = param.ds_numel - start
                param.ds_tensor.narrow(0, 0, elements_to_copy).copy_(one_dim_param.narrow(0, start, elements_to_copy))
            see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}', force=False)
            free_param(param)
            see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}', force=False)
            if param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
                self.param_swapper.swap_out_and_release([param])
                print_rank_0(f'ID {param.ds_id} Offloaded to nvme offload and buffers released.')
                see_memory_usage(f'ID {param.ds_id} Offloaded to nvme offload and buffers released.', force=False)
            print_rank_0(f'ID {param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}')

    def _param_status(self, param):
        if param.ds_tensor is not None:
            print_rank_0(f'Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned numel {param.ds_tensor.numel()}, data numel {param.data.numel()}')
        else:
            print_rank_0(f'Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned ds_tensor {param.ds_tensor}, data numel {param.data.numel()}')

    def _allgather_param(self, param, async_op=False, hierarchy=0):
        partition_size = param.ds_tensor.ds_numel
        tensor_size = partition_size * self.world_size
        aligned_param_size = self._aligned_size(param)
        assert tensor_size == aligned_param_size, f'param id {param.ds_id} aligned size {aligned_param_size} does not match tensor size {tensor_size}'
        print_rank_0(f"{'--' * hierarchy}---- Before allocating allgather param {debug_param2name_id_shape_status(param)} partition size={partition_size}")
        see_memory_usage(f'Before allocate allgather param {debug_param2name_id_shape_status(param)} partition_size={partition_size} ', force=False)
        flat_tensor = torch.zeros(aligned_param_size, dtype=param.dtype, device=param.device).view(-1)
        see_memory_usage(f'After allocate allgather param {debug_param2name_id_shape_status(param)} {aligned_param_size} {partition_size} ', force=False)
        torch.cuda.synchronize()
        print_rank_0(f"{'--' * hierarchy}----allgather param with {debug_param2name_id_shape_status(param)} partition size={partition_size}")
        if self.use_all_gather_base:
            handle = dist.all_gather_base(flat_tensor, param.ds_tensor, group=self.ds_process_group, async_op=async_op)
        else:
            partitions = []
            for i in range(self.world_size):
                partitions.append(flat_tensor.narrow(0, partition_size * i, partition_size))
                if i == dist.get_rank(group=self.ds_process_group):
                    partitions[i].data.copy_(param.ds_tensor.data, non_blocking=True)
            handle = dist.all_gather(partitions, partitions[self.rank], group=self.ds_process_group, async_op=async_op)
        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
        param.data = replicated_tensor.data
        return handle

    def _allgather_params_coalesced(self, param_list, hierarchy=0):
        """ blocking call
        avoid explicit memory copy in _allgather_params
        """
        if len(param_list) == 0:
            return
        partition_sizes = []
        local_tensors = []
        for param in param_list:
            partition_sizes.append(param.ds_tensor.ds_numel)
            local_tensors.append(param.ds_tensor)
        allgather_params = []
        for psize in partition_sizes:
            tensor_size = psize * self.world_size
            flat_tensor = torch.empty(tensor_size, dtype=param_list[0].dtype, device=self.local_device).view(-1)
            flat_tensor.requires_grad = False
            allgather_params.append(flat_tensor)
        launch_handles = []
        for param_idx, param in enumerate(param_list):
            input_tensor = local_tensors[param_idx].view(-1)
            if self.use_all_gather_base:
                h = dist.all_gather_base(allgather_params[param_idx], input_tensor, group=self.ds_process_group, async_op=True)
            else:
                output_list = []
                for i in range(self.world_size):
                    psize = partition_sizes[param_idx]
                    partition = allgather_params[param_idx].narrow(0, i * psize, psize)
                    output_list.append(partition)
                    if not partition.is_cuda:
                        logger.warning(f'param {param_idx}, partition {i} is not on CUDA, partition shape {partition.size()}')
                h = dist.all_gather(output_list, input_tensor, group=self.ds_process_group, async_op=True)
            launch_handles.append(h)
        launch_handles[-1].wait()
        for i, param in enumerate(param_list):
            gathered_tensor = allgather_params[i]
            param.data = gathered_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape).data
        torch.cuda.synchronize()
        return None

    def _allgather_params(self, param_list, hierarchy=0):
        if len(param_list) == 0:
            return
        partition_size = sum([param.ds_tensor.ds_numel for param in param_list])
        tensor_size = partition_size * self.world_size
        flat_tensor = torch.empty(tensor_size, dtype=param_list[0].dtype, device=self.local_device)
        flat_tensor.requres_grad = False
        partitions = []
        for i in range(self.world_size):
            start = partition_size * i
            partitions.append(flat_tensor.narrow(0, start, partition_size))
            if i == self.rank:
                offset = 0
                for param in param_list:
                    param_numel = param.ds_tensor.ds_numel
                    partitions[i].narrow(0, offset, param_numel).copy_(param.ds_tensor.data)
                    offset += param_numel
        dist.all_gather(partitions, partitions[self.rank], group=self.ds_process_group, async_op=False)
        param_offset = 0
        for param in param_list:
            param_partition_size = param.ds_tensor.ds_numel
            param_size = param.ds_numel
            replicated_tensor = torch.empty(param.ds_shape, dtype=param.dtype, device=self.local_device)
            for i in range(self.world_size):
                start = i * partition_size
                param_start = i * param_partition_size
                if param_start < param_size:
                    numel_to_copy = min(param_size - param_start, param_partition_size)
                    part_to_copy = partitions[i].narrow(0, param_offset, numel_to_copy)
                    replicated_tensor.view(-1).narrow(0, param_start, numel_to_copy).copy_(part_to_copy)
            param_offset += param.ds_tensor.ds_numel
            param.data = replicated_tensor.data
        return None

    def _reduce_scatter_gradients(self, param_list):
        handles_and_reduced_partitions = []
        for param in param_list:
            assert param.grad.numel() == param.ds_numel, f'{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter gradients whose size is not same as the params'
            handles_and_reduced_partitions.append(self._reduce_scatter_gradient(param))
        for param, (handle, reduced_partition) in zip(param_list, handles_and_reduced_partitions):
            if handle is not None:
                handle.wait()
            partition_size = param.ds_tensor.ds_numel
            start = self.rank * partition_size
            end = start + partition_size
            if start < param.ds_numel and end > param.ds_numel:
                elements = param.ds_numel - start
                param.grad.view(-1).narrow(0, start, elements).copy_(reduced_partition.narrow(0, 0, elements))

    def _reduce_scatter_gradient(self, param):
        partition_size = param.ds_tensor.ds_numel
        total_size = partition_size * self.world_size
        input_list = []
        for i in range(self.world_size):
            start = i * partition_size
            end = start + partition_size
            if start < param.ds_numel and end <= param.ds_numel:
                input = param.grad.view(-1).narrow(0, start, partition_size)
            else:
                input = torch.zeros(partition_size, dtype=param.dtype, device=param.device)
                if start < param.ds_numel:
                    elements = param.ds_numel - start
                    input.narrow(0, 0, elements).copy_(param.grad.view(-1).narrow(0, start, elements))
            input_list.append(input)
        rank = dist.get_rank(group=self.ds_process_group)
        handle = dist.reduce_scatter(input_list[rank], input_list, group=self.ds_process_group, async_op=True)
        return handle, input_list[rank]

    def _partition_gradients(self, param_list, partition_buffers=None, accumulate=False):
        if partition_buffers is None:
            partition_buffers = [None] * len(param_list)
        for param, partition_buffer in zip(param_list, partition_buffers):
            self._partition_gradient(param, partition_buffer=partition_buffer, accumulate=accumulate)

    def _partition_gradient(self, param, partition_buffer=None, accumulate=False):
        print_rank_0(f'Partitioning param {param.ds_id} gradient of size {param.grad.numel()} type {param.grad.dtype} part_size {param.ds_tensor.ds_numel}')
        see_memory_usage('Before partitioning gradients', force=False)
        partition_size = param.ds_tensor.ds_numel
        if partition_buffer is None:
            assert not accumulate, 'No buffer to accumulate to'
            partition_buffer = torch.zeros(partition_size, dtype=param.dtype, device=param.device)
        else:
            assert partition_buffer.numel() >= partition_size, f'The partition buffer size {partition_buffer.numel()} should match the size of param.ds_tensor {partition_size}'
        rank = dist.get_rank(group=self.ds_process_group)
        start = partition_size * rank
        end = start + partition_size
        dest_tensor_full_buffer = partition_buffer.view(-1).narrow(0, 0, partition_size)
        if start < param.ds_numel:
            elements = min(param.ds_numel - start, partition_size)
            dest_tensor = dest_tensor_full_buffer.narrow(0, 0, elements)
            src_tensor = param.grad.view(-1).narrow(0, start, elements)
            if not accumulate:
                dest_tensor.copy_(src_tensor)
            elif src_tensor.device == dest_tensor.device:
                dest_tensor.add_(src_tensor)
            else:
                acc_tensor = torch.empty(src_tensor.numel(), dtype=param.dtype, device=param.device)
                acc_tensor.copy_(dest_tensor)
                acc_tensor.add_(src_tensor)
                dest_tensor.copy_(acc_tensor)
        param.grad.data = dest_tensor_full_buffer.data
        see_memory_usage('After partitioning gradients', force=False)


class ZeRoTraceMode(Enum):
    RECORD = 1
    COMPLETE = 2
    INVALID = 3


def debug_module2name_id(module):
    return f'name={debug_module2name(module)} id={module.id}'


@instrument_w_nvtx
def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


def iter_params(module: Module, recurse=False) ->Iterable[Parameter]:
    return map(lambda pair: pair[1], get_all_parameters(module, recurse))


class PostBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        if output.requires_grad:
            module.ds_grads_remaining += 1
            ctx.pre_backward_function = pre_backward_function
        output = output.detach()
        return output

    @staticmethod
    def backward(ctx, *args):
        ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
        if ctx.module.ds_grads_remaining == 0:
            ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PreBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        if not hasattr(module, 'applied_pre_backward_ref_cnt'):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


FWD_MODULE_STACK = list()


def _init_external_params(module):
    if not hasattr(module, '_external_params'):
        module._external_params = {}

        def external_parameters(self):
            return self._external_params.items()

        def all_parameters(self):
            return itertools.chain(self.named_parameters(self, recurse=False), external_parameters(self))
        module.ds_external_parameters = types.MethodType(external_parameters, module)
        module.all_parameters = types.MethodType(all_parameters, module)


def register_external_parameter(module, parameter):
    """Instruct DeepSpeed to coordinate ``parameter``'s collection and partitioning in
    the forward and backward passes of ``module``.

    This is used when a parameter is accessed outside of its owning module's
    ``forward()``. DeepSpeed must know to collect it from its partitioned
    state and when to release the memory.

    .. note::
        This is only applicable to training with ZeRO stage 3.

    Args:
        module (``torch.nn.Module``): The module that requires ``parameter`` in its forward pass.
        parameter (``torch.nn.Parameter``): The parameter to register.

    Raises:
        RuntimeError: If ``parameter`` is not of type ``torch.nn.Parameter``.


    Examples
    ========

    #. Register a weight that is used in another module's forward pass (line 6).
       Parameter ``layer1.weight`` is used by ``layer2`` (line 11).

        .. code-block:: python
            :linenos:
            :emphasize-lines: 6,11

            class ModuleZ3(torch.nn.Module):
                def __init__(self, *args):
                    super().__init__(self, *args)
                    self.layer1 = SomeLayer()
                    self.layer2 = OtherLayer()
                    deepspeed.zero.register_external_parameter(self, self.layer1.weight)

                def forward(self, input):
                    x = self.layer1(input)
                    # self.layer1.weight is required by self.layer2.forward
                    y = self.layer2(x, self.layer1.weight)
                    return y
    """
    if not isinstance(parameter, torch.nn.Parameter):
        raise RuntimeError('Parameter is not a torch.nn.Parameter')
    if not hasattr(module, '_external_params'):
        _init_external_params(module)
    key = id(parameter)
    module._external_params[key] = parameter


class ZeROOrderedDict(OrderedDict):

    def __init__(self, parent_module, *args, **kwargs):
        """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

        Args:
            parent_module (``collections.OrderedDict``): the collection to replace
        """
        super().__init__(*args, **kwargs)
        self._parent_module = parent_module
        self._in_forward = False

    def __getitem__(self, key):
        param = super().__getitem__(key)
        if param is None:
            return param
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if self._parent_module._parameters._in_forward:
                register_external_parameter(FWD_MODULE_STACK[-1], param)
                param.all_gather()
                print_rank_0(f'Registering external parameter from getter {key} ds_id = {param.ds_id}', force=False)
        return param


def _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function, output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad:
            outputs.register_hook(backward_function)
        return outputs
    else:
        return outputs


def is_builtin_type(obj):
    return obj.__class__.__module__ == '__builtin__' or obj.__class__.__module__ == 'builtins'


def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if isinstance(outputs, (tuple, list)):
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module, functional, backward_function, output)
            touched_outputs.append(touched_output)
        return outputs.__class__(touched_outputs)
    elif isinstance(outputs, dict):
        for key in outputs.keys():
            outputs[key] = _apply_to_tensors_only(module, functional, backward_function, outputs[key])
        return outputs
    elif isinstance(outputs, torch.Tensor):
        touched_outputs = functional.apply(module, backward_function, outputs)
        if not is_zero_param(touched_outputs) and is_zero_param(outputs):
            touched_outputs.ds_param_alias = outputs
        return touched_outputs
    else:
        if not is_builtin_type(outputs):
            global warned
            if not warned and dist.get_rank() == 0:
                logger.warning(f'A module has unknown inputs or outputs type ({type(outputs)}) and the tensors embedded in it cannot be detected. The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and output tensors and therefore may not get triggered properly.')
                warned = True
        return outputs


def _inject_parameters(module, cls):
    for module in module.modules():
        if cls == ZeROOrderedDict:
            new_param = cls(parent_module=module)
        else:
            new_param = cls()
        for key, param in module._parameters.items():
            new_param[key] = param
        module._parameters = new_param


def unregister_external_parameter(module, parameter):
    """Reverses the effects of :meth:`register_external_parameter`.

    Args:
        module (``torch.nn.Module``): The module to affect.
        parameter (``torch.nn.Parameter``): The parameter to unregister.

    Raises:
        RuntimeError: If ``parameter`` is not of type ``torch.nn.Parameter``.
        RuntimeError: If ``parameter`` is not a registered external parameter of ``module``.
    """
    if not isinstance(parameter, torch.nn.Parameter):
        raise RuntimeError('Parameter is not a torch.nn.Parameter')
    if not hasattr(module, '_external_params') or id(parameter) not in module._external_params:
        raise RuntimeError('Parameter is not a registered external parameter of module.')
    key = id(parameter)
    del module._external_params[key]


class DeepSpeedZeRoOffload(object):

    def __init__(self, module, timers, ds_config, overlap_comm=True, prefetch_bucket_size=50000000, max_reuse_distance=1000000000, max_live_parameters=1000000000, param_persistence_threshold=100000, model_persistence_threshold=sys.maxsize, offload_param_config=None, mpu=None):
        see_memory_usage('DeepSpeedZeRoOffload initialize [begin]', force=True)
        print_rank_0(f'initialized {__class__.__name__} with args: {locals()}', force=False)
        self.module = module
        self.dtype = list(module.parameters())[0].dtype
        self.offload_device = None
        self.offload_param_pin_memory = False
        if offload_param_config is not None and offload_param_config.device != OffloadDeviceEnum.none:
            self.offload_device = offload_param_config.device
            self.offload_param_pin_memory = offload_param_config.pin_memory
        self._convert_to_zero_parameters(ds_config, module, mpu)
        for m in module.modules():
            _init_external_params(m)
        _inject_parameters(module, ZeROOrderedDict)
        self.param_numel_persistence_threshold = int(param_persistence_threshold)
        self.model_persistence_threshold = int(model_persistence_threshold)
        self.persistent_parameters = self.mark_persistent_parameters(self.param_numel_persistence_threshold, self.model_persistence_threshold)
        self.param_coordinators = {}
        self._prefetch_bucket_sz = int(prefetch_bucket_size)
        self._max_reuse_distance_in_numel = int(max_reuse_distance)
        self._max_available_parameters_in_numel = int(max_live_parameters)
        self.__allgather_stream = Stream() if overlap_comm else torch.cuda.default_stream()
        self.forward_hooks = []
        self.backward_hooks = []
        self.setup_zero_stage3_hooks()
        print_rank_0(f'Created module hooks: forward = {len(self.forward_hooks)}, backward = {len(self.backward_hooks)}', force=False)
        see_memory_usage('DeepSpeedZeRoOffload initialize [end]', force=True)

    @instrument_w_nvtx
    def partition_all_parameters(self):
        """Partitioning Parameters that were not partitioned usually if parameters
        of modules whose input parameters do not require grad computation do not
        trigger post call and will therefore will remain unpartitioned"""
        self.get_param_coordinator(training=self.module.training).release_and_reset_all(self.module)
        for param in iter_params(self.module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f'{param.ds_summary()} expected to be released')

    def get_param_coordinator(self, training):
        if not training in self.param_coordinators:
            self.param_coordinators[training] = PartitionedParameterCoordinator(prefetch_bucket_sz=self._prefetch_bucket_sz, max_reuse_distance_in_numel=self._max_reuse_distance_in_numel, max_available_parameters_in_numel=self._max_available_parameters_in_numel, allgather_stream=self.__allgather_stream, prefetch_nvme=self.offload_device == OffloadDeviceEnum.nvme)
        return self.param_coordinators[training]

    def _convert_to_zero_parameters(self, ds_config, module, mpu):
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)]
        if non_zero_params:
            zero_params = [p for p in module.parameters() if is_zero_param(p)]
            if zero_params:
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params)
            else:
                group = None
                if mpu:
                    group = mpu.get_data_parallel_group()
                Init(module=module, data_parallel_group=group, dtype=self.dtype, config_dict_or_path=ds_config, remote_device=self.offload_device, pin_memory=self.offload_param_pin_memory, mpu=mpu)

    def destroy(self):
        self._remove_module_hooks()

    def _remove_module_hooks(self):
        num_forward_hooks = len(self.forward_hooks)
        num_backward_hooks = len(self.backward_hooks)
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()
        print_rank_0(f'Deleted module hooks: forward = {num_forward_hooks}, backward = {num_backward_hooks}', force=False)

    def setup_zero_stage3_hooks(self):
        self.hierarchy = 0

        @instrument_w_nvtx
        def _end_of_forward_hook(module, *args):
            if not torch._C.is_grad_enabled():
                self.get_param_coordinator(training=False).reset_step()
        self._register_hooks_recursively(self.module)
        self.module.register_forward_hook(_end_of_forward_hook)
        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(self.module)

    def mark_persistent_parameters(self, param_threshold, model_threshold):
        persistent_params = []
        total_persistent_parameters = 0
        params_count = 0
        for _, param in self.module.named_parameters(recurse=True):
            if param.ds_numel + total_persistent_parameters > model_threshold:
                continue
            if param.ds_numel < param_threshold:
                params_count += 1
                param.ds_persist = True
                persistent_params.append(param)
                total_persistent_parameters += param.ds_numel
        print_rank_0(f'Parameter Offload: Total persistent parameters: {total_persistent_parameters} in {params_count} params', force=True)
        return persistent_params

    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count
        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively(child, count=count)

        @instrument_w_nvtx
        def _pre_forward_module_hook(module, *args):
            self.pre_sub_module_forward_function(module)

        @instrument_w_nvtx
        def _post_forward_module_hook(module, input, output):
            global FWD_MODULE_STACK
            FWD_MODULE_STACK.pop()
            if output is None:
                output = []
            elif not isinstance(output, (list, tuple)):
                if torch.is_tensor(output):
                    output = [output]
                else:
                    outputs = []
                    output = output if isinstance(output, dict) else vars(output)
                    for name, val in output.items():
                        if not name.startswith('__') and torch.is_tensor(val):
                            outputs.append(val)
                    output = outputs
            for item in filter(lambda item: is_zero_param(item) or hasattr(item, 'ds_param_alias'), output):
                key = id(item) if hasattr(item, 'ds_id') else id(item.ds_param_alias)
                actual_external_param = item if hasattr(item, 'ds_id') else item.ds_param_alias
                if not any(key in m._external_params for m in FWD_MODULE_STACK):
                    actual_external_param.is_external_param = True
                    module_to_register = FWD_MODULE_STACK[-1]
                    register_external_parameter(module_to_register, actual_external_param)
                    print_rank_0(f'Registering dangling parameter for module {module_to_register.__class__.__name__}, ds_id = {actual_external_param.ds_id}.', force=False)
                    if key in module._external_params:
                        print_rank_0(f'  Unregistering nested dangling parameter from module {module.__class__.__name__}, ds_id = {actual_external_param.ds_id}', force=False)
                        unregister_external_parameter(module, actual_external_param)
                    actual_external_param.all_gather()
            self.post_sub_module_forward_function(module)

        def _pre_backward_module_hook(module, inputs, output):

            @instrument_w_nvtx
            def _run_before_backward_function(sub_module):
                if sub_module.applied_pre_backward_ref_cnt > 0:
                    self.pre_sub_module_backward_function(sub_module)
                    sub_module.applied_pre_backward_ref_cnt -= 1
            return _apply_to_tensors_only(module, PreBackwardFunction, _run_before_backward_function, output)

        def _alternate_post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            def _run_after_backward_hook(*unused):
                module.ds_grads_remaining = module.ds_grads_remaining - 1
                if module.ds_grads_remaining == 0:
                    self.post_sub_module_backward_function(module)

            def _run_before_forward_function(input):
                if input.requires_grad:
                    module.ds_grads_remaining += 1
            return _apply_forward_and_backward_to_tensors_only(module, _run_before_forward_function, _run_after_backward_hook, inputs)

        def _post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            @instrument_w_nvtx
            def _run_after_backward_function(sub_module):
                if sub_module.ds_grads_remaining == 0:
                    self.post_sub_module_backward_function(sub_module)
            return _apply_to_tensors_only(module, PostBackwardFunction, _run_after_backward_function, inputs)
        self.forward_hooks.append(module.register_forward_pre_hook(_pre_forward_module_hook))
        self.forward_hooks.append(module.register_forward_hook(_post_forward_module_hook))
        self.backward_hooks.append(module.register_forward_hook(_pre_backward_module_hook))
        self.backward_hooks.append(module.register_forward_pre_hook(_post_backward_module_hook))

    @torch.no_grad()
    def pre_sub_module_forward_function(self, sub_module):
        see_memory_usage(f'Before sub module function {sub_module.__class__.__name__}', force=False)
        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(sub_module)
        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)
        param_coordinator.fetch_sub_module(sub_module)
        see_memory_usage(f'Before sub module function {sub_module.__class__.__name__} after fetch', force=False)

    @torch.no_grad()
    def post_sub_module_forward_function(self, sub_module):
        see_memory_usage(f'After sub module function {sub_module.__class__.__name__} {sub_module.id} before release', force=False)
        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.release_sub_module(sub_module)
        see_memory_usage(f'After sub module function {sub_module.__class__.__name__}  {sub_module.id} after release', force=False)

    @torch.no_grad()
    def pre_sub_module_backward_function(self, sub_module):
        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)
        param_coordinator.fetch_sub_module(sub_module)

    @torch.no_grad()
    def post_sub_module_backward_function(self, sub_module):
        see_memory_usage(f'After sub module backward function {sub_module.__class__.__name__} {sub_module.id} before release', force=False)
        self.get_param_coordinator(training=sub_module.training).release_sub_module(sub_module)
        see_memory_usage(f'After sub module backward function {sub_module.__class__.__name__} {sub_module.id} after release', force=False)


def get_current_level():
    """
    Return logger's current log level
    """
    return logger.getEffectiveLevel()


log_levels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}


def should_log_le(max_log_level_str):
    """
    Args:
        max_log_level_str: maximum log level as a string

    Returns ``True`` if the current log_level is less or equal to the specified log level. Otherwise ``False``.

    Example:

        ``should_log_le("info")`` will return ``True`` if the current log level is either ``logging.INFO`` or ``logging.DEBUG``
    """
    if not isinstance(max_log_level_str, str):
        raise ValueError(f'{max_log_level_str} is not a string')
    max_log_level_str = max_log_level_str.lower()
    if max_log_level_str not in log_levels:
        raise ValueError(f'{max_log_level_str} is not one of the `logging` levels')
    return get_current_level() <= log_levels[max_log_level_str]


class DeepSpeedCPUAdam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(self, model_params, lr=0.001, bias_correction=True, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, adamw_mode=True, fp32_optimizer_states=True):
        """Fast vectorized implementation of two variations of Adam optimizer on CPU:

        * Adam: A Method for Stochastic Optimization: (https://arxiv.org/abs/1412.6980);
        * AdamW: Fixing Weight Decay Regularization in Adam (https://arxiv.org/abs/1711.05101)

        DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
        In order to apply this optimizer, the model requires to have its master parameter (in FP32)
        reside on the CPU memory.

        To train on a heterogeneous system, such as coordinating CPU and GPU, DeepSpeed offers
        the ZeRO-Offload technology which efficiently offloads the optimizer states into CPU memory,
        with minimal impact on training throughput. DeepSpeedCPUAdam plays an important role to minimize
        the overhead of the optimizer's latency on CPU. Please refer to ZeRO-Offload tutorial
        (https://www.deepspeed.ai/tutorials/zero-offload/) for more information on how to enable this technology.

        For calling step function, there are two options available: (1) update optimizer's states and (2) update
        optimizer's states and copy the parameters back to GPU at the same time. We have seen that the second
        option can bring 30% higher throughput than the doing the copy separately using option one.


        .. note::
                We recommend using our `config
                <https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`_
                to allow :meth:`deepspeed.initialize` to build this optimizer
                for you.


        Arguments:
            model_params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups.
            lr (float, optional): learning rate. (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square. (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability. (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_
                (default: False) NOT SUPPORTED in DeepSpeed CPUAdam!
            adamw_mode: select between Adam and AdamW implementations (default: AdamW)
            full_precision_optimizer_states: creates momementum and variance in full precision regardless of
                        the precision of the parameters (default: True)
        """
        default_args = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bias_correction, amsgrad=amsgrad)
        super(DeepSpeedCPUAdam, self).__init__(model_params, default_args)
        self.cpu_vendor = get_cpu_info()['vendor_id_raw'].lower()
        if 'amd' in self.cpu_vendor:
            for group_id, group in enumerate(self.param_groups):
                for param_id, p in enumerate(group['params']):
                    if p.dtype == torch.half:
                        logger.warning('FP16 params for CPUAdam may not work on AMD CPUs')
                        break
                else:
                    continue
                break
        self.opt_id = DeepSpeedCPUAdam.optimizer_id
        DeepSpeedCPUAdam.optimizer_id = DeepSpeedCPUAdam.optimizer_id + 1
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        self.ds_opt_adam = CPUAdamBuilder().load()
        self.ds_opt_adam.create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode, should_log_le('info'))

    def __del__(self):
        self.ds_opt_adam.destroy_adam(self.opt_id)

    def __setstate__(self, state):
        super(DeepSpeedCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None, fp16_param_groups=None):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.
            fp16_param_groups: FP16 GPU parameters to update. Performing the
                copy here reduces communication time. Defaults to ``None``.

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        device = torch.device('cpu')
        if type(fp16_param_groups) is list:
            if type(fp16_param_groups[0]) is not list:
                fp16_param_groups = [fp16_param_groups]
        elif fp16_param_groups is not None:
            fp16_param_groups = [[fp16_param_groups]]
        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                assert p.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                state['step'] += 1
                beta1, beta2 = group['betas']
                if fp16_param_groups is not None:
                    self.ds_opt_adam.adam_update_copy(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'], group['weight_decay'], group['bias_correction'], p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'], fp16_param_groups[group_id][param_id].data)
                else:
                    self.ds_opt_adam.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'], group['weight_decay'], group['bias_correction'], p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'])
        return loss


class LossScalerBase:
    """LossScalarBase
    Base class for a loss scaler
    """

    def __init__(self, cur_scale):
        self.cur_scale = cur_scale

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def update_scale(self, overflow):
        pass

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)


class DynamicLossScaler(LossScalerBase):
    """
    Class that manages dynamic loss scaling.  It is recommended to use :class:`DynamicLossScaler`
    indirectly, by supplying ``dynamic_loss_scale=True`` to the constructor of
    :class:`FP16_Optimizer`.  However, it's important to understand how :class:`DynamicLossScaler`
    operates, because the default options can be changed using the
    the ``dynamic_loss_args`` argument to :class:`FP16_Optimizer`'s constructor.

    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.  If overflowing gradients are
    encountered, :class:`DynamicLossScaler` informs :class:`FP16_Optimizer` that an overflow has
    occurred.
    :class:`FP16_Optimizer` then skips the update step for this particular iteration/minibatch,
    and :class:`DynamicLossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients detected,
    :class:`DynamicLossScaler` increases the loss scale once more.
    In this way :class:`DynamicLossScaler` attempts to "ride the edge" of
    always using the highest loss scale possible without incurring overflow.

    Args:
        init_scale (float, optional, default=2**32):  Initial loss scale attempted by :class:`DynamicLossScaler.`
        scale_factor (float, optional, default=2.0):  Factor used when adjusting the loss scale. If an overflow is encountered, the loss scale is readjusted to loss scale/``scale_factor``.  If ``scale_window`` consecutive iterations take place without an overflow, the loss scale is readjusted to loss_scale*``scale_factor``.
        scale_window (int, optional, default=1000):  Number of consecutive iterations without an overflow to wait before increasing the loss scale.
    """

    def __init__(self, init_scale=2 ** 32, scale_factor=2.0, scale_window=1000, min_scale=1, delayed_shift=1, consecutive_hysteresis=False, raise_error_at_min_scale=True):
        super(DynamicLossScaler, self).__init__(init_scale)
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis
        self.raise_error_at_min_scale = raise_error_at_min_scale

    def has_overflow_serial(self, params):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
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
            if cpu_sum in [float('inf'), -float('inf')] or cpu_sum != cpu_sum:
                return True
            return False

    def update_scale(self, overflow):
        if overflow:
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                if self.cur_scale == self.min_scale and self.raise_error_at_min_scale:
                    raise Exception('Current loss scale already at minimum - cannot decrease scale anymore. Exiting run.')
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


class LossScaler(LossScalerBase):
    """
    Class that manages a static loss scale.  This class is intended to interact with
    :class:`FP16_Optimizer`, and should not be directly manipulated by the user.

    Use of :class:`LossScaler` is enabled via the ``static_loss_scale`` argument to
    :class:`FP16_Optimizer`'s constructor.

    Args:
        scale (float, optional, default=1.0):  The loss scale.
    """

    def __init__(self, scale=1):
        super(LossScaler, self).__init__(scale)

    def has_overflow(self, params):
        return False

    def _has_inf_or_nan(x):
        return False


ZERO_STAGE = 'zero_stage'


def empty_cache():
    torch.cuda.empty_cache()


def get_global_norm(norm_list):
    """ Compute total from a list of norms
    """
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm ** 2.0
    return sqrt(total_norm)


def is_moe_param(param: torch.Tensor) ->bool:
    if hasattr(param, 'allreduce') and not param.allreduce:
        return True
    return False


def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()


pg_correctness_test = False


def split_half_float_double(tensors):
    dtypes = ['torch.cuda.HalfTensor', 'torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor', 'torch.cuda.BFloat16Tensor']
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


class DeepSpeedZeroOptimizer(ZeROOptimizer):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed Tutorial

    """

    def __init__(self, init_optimizer, param_names, timers, static_loss_scale=1.0, dynamic_loss_scale=False, dynamic_loss_args=None, verbose=True, contiguous_gradients=True, reduce_bucket_size=500000000, allgather_bucket_size=5000000000, dp_process_group=None, expert_parallel_group=None, expert_data_parallel_group=None, reduce_scatter=True, overlap_comm=False, cpu_offload=False, mpu=None, clip_grad=0.0, communication_data_type=torch.float16, postscale_gradients=True, gradient_predivide_factor=1.0, gradient_accumulation_steps=1, ignore_unused_parameters=True, partition_grads=True, round_robin_gradients=False, has_moe_layers=False, fp16_master_weights_and_gradients=False, elastic_checkpoint=False):
        if dist.get_rank() == 0:
            logger.info(f'Reduce bucket size {reduce_bucket_size}')
            logger.info(f'Allgather bucket size {allgather_bucket_size}')
            logger.info(f'CPU Offload: {cpu_offload}')
            logger.info(f'Round robin gradient partitioning: {round_robin_gradients}')
        self.elastic_checkpoint = elastic_checkpoint
        self.param_names = param_names
        self.mpu = mpu
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA.')
        self.optimizer = init_optimizer
        util_ops = UtilsBuilder().load()
        self.flatten = util_ops.flatten
        self.unflatten = util_ops.unflatten
        self.partition_gradients = partition_grads
        self.timers = timers
        self.reduce_scatter = reduce_scatter
        self.overlap_comm = overlap_comm
        self.cpu_offload = cpu_offload
        self.deepspeed_adam_offload = cpu_offload
        self.device = torch.cuda.current_device() if not self.cpu_offload else 'cpu'
        self.dp_process_group = dp_process_group
        self.ep_process_group = expert_parallel_group
        self.expert_dp_process_group = expert_data_parallel_group
        dp_size = dist.get_world_size(group=self.dp_process_group)
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]
        self.is_gradient_accumulation_boundary = True
        self.contiguous_gradients = contiguous_gradients or cpu_offload
        self.has_moe_layers = has_moe_layers
        if self.has_moe_layers:
            self._configure_moe_settings()
        self._global_grad_norm = 0.0
        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_world_size = 1
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_world_size = mpu.get_model_parallel_world_size()
            self.model_parallel_rank = bwc_tensor_model_parallel_rank(mpu)
        self.overflow = False
        self.clip_grad = clip_grad
        self.communication_data_type = communication_data_type
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = 0
        self.ignore_unused_parameters = ignore_unused_parameters
        self.round_robin_gradients = round_robin_gradients
        self.extra_large_param_to_reduce = None
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients
        if self.fp16_master_weights_and_gradients:
            assert self.cpu_offload and type(self.optimizer) in [DeepSpeedCPUAdam], f'fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32. Currently only supported using ZeRO-Offload with DeepSpeedCPUAdam. But current setting is ZeRO-Offload:{self.cpu_offload} and optimizer type {type(self.optimizer)}. Either disable fp16_master_weights_and_gradients or enable ZeRO-2 Offload with DeepSpeedCPUAdam'
        if self.reduce_scatter:
            assert self.communication_data_type in (torch.float16, torch.bfloat16), f"ZeRO-2 supports only float16 or bfloat16 communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
            assert self.gradient_predivide_factor == 1.0, 'gradient_predivide_factor != 1.0 is not yet supported with ZeRO-2 with reduce scatter enabled'
            assert self.postscale_gradients, 'pre-scale gradients is not yet supported with ZeRO-2 with reduce scatter enabled'
        self.bit16_groups = []
        self.bit16_groups_flat = []
        self.parallel_partitioned_bit16_groups = []
        self.single_partition_of_fp32_groups = []
        self.params_not_in_partition = []
        self.params_in_partition = []
        self.first_offset = []
        self.partition_size = []
        self.nccl_start_alignment_factor = 2
        assert allgather_bucket_size % self.nccl_start_alignment_factor == 0, f'allgather_bucket_size must be a multiple of nccl_start_alignment_factor, {self.nccl_start_alignment_factor} '
        self.all_reduce_print = False
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        self.round_robin_bit16_groups = []
        self.round_robin_bit16_indices = []
        self.groups_padding = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            trainable_parameters = [param for param in param_group['params'] if param.requires_grad]
            self.bit16_groups.append(trainable_parameters)
            see_memory_usage(f'Before moving param group {i} to CPU')
            move_to_cpu(self.bit16_groups[i])
            empty_cache()
            see_memory_usage(f'After moving param group {i} to CPU', force=False)
            if self.round_robin_gradients:
                round_robin_tensors, round_robin_indices = self._round_robin_reorder(self.bit16_groups[i], dist.get_world_size(group=self.real_dp_process_group[i]))
            else:
                round_robin_tensors = self.bit16_groups[i]
                round_robin_indices = list(range(len(self.bit16_groups[i])))
            self.round_robin_bit16_groups.append(round_robin_tensors)
            self.round_robin_bit16_indices.append(round_robin_indices)
            self.bit16_groups_flat.append(self.flatten_dense_tensors_aligned(self.round_robin_bit16_groups[i], self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i])))
            see_memory_usage(f'After flattening and moving param group {i} to GPU', force=False)
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                padding = self.bit16_groups_flat[i].numel() - sum([t.numel() for t in self.round_robin_bit16_groups[i]])
            else:
                padding = 0
            self.groups_padding.append(padding)
            if dist.get_rank(group=self.real_dp_process_group[i]) == 0:
                see_memory_usage(f'After Flattening and after emptying param group {i} cache', force=False)
            self._update_model_bit16_weights(i)
            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)
            for partitioned_data in data_parallel_partitions:
                assert partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0
            for partitioned_data in data_parallel_partitions:
                assert partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0
            if not fp16_master_weights_and_gradients:
                self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].clone().float().detach())
            else:
                self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].clone().half().detach())
            self.single_partition_of_fp32_groups[i].requires_grad = True
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]
            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(self.round_robin_bit16_groups[i], partition_size, partition_id)
            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)
        for rank in range(dist.get_world_size()):
            if dist.get_rank() == rank:
                None
                dist.barrier()
        self.reduce_bucket_size = int(reduce_bucket_size)
        self.allgather_bucket_size = int(allgather_bucket_size)
        self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.reduction_stream = torch.Stream()
        self.cpu_computation_stream = torch.Stream()
        self.copy_grad_stream = torch.Stream()
        self.callback_queued = False
        self.param_dict = {}
        self.is_param_in_current_partition = {}
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None
        self.ipg_bucket_has_moe_params = False
        self.param_id = {}
        largest_param_numel = 0
        count = 0
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1
        for param_group in self.params_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True
        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False
        if self.cpu_offload:
            self.accumulated_grads_in_cpu = {}
            self.norm_for_param_grads = {}
            self.local_overflow = False
            self.grad_position = {}
            self.temp_grad_buffer_for_cpu_offload = torch.zeros(largest_param_numel, device=self.device, dtype=self.dtype).pin_memory()
            self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel, device=torch.cuda.current_device(), dtype=self.dtype)
            for i, params_group in enumerate(self.bit16_groups):
                self.get_grad_position(i, self.params_in_partition[i], self.first_offset[i], self.partition_size[i])
        self.param_to_partition_ids = {}
        self.is_partition_reduced = {}
        self.remaining_grads_in_partition = {}
        self.total_grads_in_partition = {}
        self.is_grad_computed = {}
        self.grad_partition_insertion_offset = {}
        self.grad_start_offset = {}
        self.averaged_gradients = {}
        self.first_param_index_in_partition = {}
        self.initialize_gradient_partitioning_data_structures()
        self.reset_partition_gradient_structures()
        if self.partition_gradients or self.overlap_comm:
            self.create_reduce_and_remove_grad_hooks()
        self.custom_loss_scaler = False
        self.external_loss_scale = None
        if self.dtype == torch.float or self.dtype == torch.bfloat16 or not dynamic_loss_scale:
            loss_scale_value = 1.0 if self.dtype == torch.float or self.dtype == torch.bfloat16 else static_loss_scale
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(scale=loss_scale_value)
            cur_iter = 0
        else:
            if dynamic_loss_args is None:
                self.loss_scaler = DynamicLossScaler()
            else:
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)
            self.dynamic_loss_scale = True
        see_memory_usage('Before initializing optimizer states', force=True)
        self.initialize_optimizer_states()
        see_memory_usage('After initializing optimizer states', force=True)
        if dist.get_rank() == 0:
            logger.info(f'optimizer state initialized')
        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f'After initializing ZeRO optimizer', force=True)
        self._link_all_hp_params()
        self._enable_universal_checkpoint()
        self._param_slice_mappings = self._create_param_mapping()

    def _enable_universal_checkpoint(self):
        for lp_param_group in self.bit16_groups:
            enable_universal_checkpoint(param_list=lp_param_group)

    def _create_param_mapping(self):
        param_mapping = []
        for i, _ in enumerate(self.optimizer.param_groups):
            param_mapping_per_group = OrderedDict()
            for lp in self.bit16_groups[i]:
                if lp._hp_mapping is not None:
                    lp_name = self.param_names[lp]
                    param_mapping_per_group[lp_name] = lp._hp_mapping.get_hp_fragment_address()
            param_mapping.append(param_mapping_per_group)
        return param_mapping

    def _link_all_hp_params(self):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        for i, _ in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            partition_size = self.bit16_groups_flat[i].numel() // dp_world_size
            flat_hp_partition = self.single_partition_of_fp32_groups[i]
            link_hp_params(lp_param_list=self.bit16_groups[i], flat_hp_partition=flat_hp_partition, partition_start=partition_id * partition_size, partition_size=partition_size, partition_optimizer_state=self.optimizer.state[flat_hp_partition], dp_group=self.real_dp_process_group[i])

    def is_moe_group(self, group):
        return 'moe' in group and group['moe']

    def _configure_moe_settings(self):
        if self.partition_gradients:
            assert self.contiguous_gradients, 'Contiguous Gradients in ZeRO Stage 2 must be set to True for MoE. Other code paths are not tested with MoE'
        if not self.partition_gradients and not self.contiguous_gradients:
            logger.warn('ZeRO Stage 1 has not been thoroughly tested with MoE. This configuration is still experimental.')
        assert self.reduce_scatter, 'Reduce Scatter in ZeRO Stage 2 must be set to True for MoE. Other code paths are not tested with MoE'
        assert any([self.is_moe_group(group) for group in self.optimizer.param_groups]), "The model has moe layers, but None of the param groups are marked as MoE. Create a param group with 'moe' key set to True before creating optimizer"
        self.is_moe_param_group = []
        for i, group in enumerate(self.optimizer.param_groups):
            if self.is_moe_group(group):
                assert all([is_moe_param(param) for param in group['params']]), 'All params in MoE group must be MoE params'
                self.real_dp_process_group[i] = self.expert_dp_process_group[group['name']]
                self.partition_count[i] = dist.get_world_size(group=self.expert_dp_process_group[group['name']])
                self.is_moe_param_group.append(True)
            else:
                self.is_moe_param_group.append(False)
        assert self.expert_dp_process_group is not None, 'Expert data parallel group should be configured with MoE'
        assert self.ep_process_group is not None, 'Expert parallel group should be configured with MoE'

    def _update_model_bit16_weights(self, group_index):
        updated_params = self.unflatten(self.bit16_groups_flat[group_index], self.round_robin_bit16_groups[group_index])
        for p, q in zip(self.round_robin_bit16_groups[group_index], updated_params):
            p.data = q.data
        for param_index, param in enumerate(self.bit16_groups[group_index]):
            new_index = self.round_robin_bit16_indices[group_index][param_index]
            param.data = self.round_robin_bit16_groups[group_index][new_index].data

    def _round_robin_reorder(self, tensor_list, num_partitions):
        partition_tensors = {}
        for i, tensor in enumerate(tensor_list):
            j = i % num_partitions
            if not j in partition_tensors:
                partition_tensors[j] = []
            partition_tensors[j].append((i, tensor))
        reordered_tensors = []
        reordered_indices = {}
        for partition_index in partition_tensors.keys():
            for i, (original_index, tensor) in enumerate(partition_tensors[partition_index]):
                reordered_indices[original_index] = len(reordered_tensors)
                reordered_tensors.append(tensor)
        return reordered_tensors, reordered_indices

    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            self.ipg_buffer = None
            self.grads_in_partition = None
            self.grads_in_partition_offset = 0

    def initialize_optimizer_states(self):
        for i, group in enumerate(self.bit16_groups):
            single_grad_partition = torch.zeros(int(self.partition_size[i]), dtype=self.single_partition_of_fp32_groups[i].dtype, device=self.device)
            self.single_partition_of_fp32_groups[i].grad = single_grad_partition.pin_memory() if self.cpu_offload else single_grad_partition
        self.optimizer.step()
        if not self.cpu_offload:
            for group in self.single_partition_of_fp32_groups:
                group.grad = None
        return

    def reduce_gradients(self, pipeline_parallel=False):
        world_size = dist.get_world_size(self.dp_process_group)
        my_rank = dist.get_rank(self.dp_process_group)
        if pipeline_parallel and self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(int(self.reduce_bucket_size), dtype=self.dtype, device=torch.cuda.current_device())
            self.ipg_buffer.append(buf_0)
            self.ipg_index = 0
        if not self.overlap_comm:
            for i, group in enumerate(self.bit16_groups):
                for param in group:
                    if param.grad is not None:
                        self.reduce_ready_partitions_and_remove_grads(param, i)
        self.overlapping_partition_gradients_reduce_epilogue()

    def get_first_param_index(self, group_id, param_group, partition_id):
        for index, param in enumerate(param_group):
            param_id = self.get_param_id(param)
            if partition_id in self.param_to_partition_ids[group_id][param_id]:
                return index
        return None

    def initialize_gradient_partitioning_data_structures(self):
        for i, param_group in enumerate(self.round_robin_bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
            self.param_to_partition_ids[i] = {}
            self.is_partition_reduced[i] = {}
            self.total_grads_in_partition[i] = {}
            self.remaining_grads_in_partition[i] = {}
            self.is_grad_computed[i] = {}
            self.grad_partition_insertion_offset[i] = {}
            self.grad_start_offset[i] = {}
            self.first_param_index_in_partition[i] = {}
            for partition_id in range(total_partitions):
                self.is_grad_computed[i][partition_id] = {}
                self.grad_partition_insertion_offset[i][partition_id] = {}
                self.grad_start_offset[i][partition_id] = {}
                self.total_grads_in_partition[i][partition_id] = 0
                self.initialize_gradient_partition(i, param_group, partition_id)
                self.is_partition_reduced[i][partition_id] = False
                self.first_param_index_in_partition[i][partition_id] = self.get_first_param_index(i, param_group, partition_id)

    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f'In ipg_epilogue before reduce_ipg_grads', 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage(f'In ipg_epilogue after reduce_ipg_grads', 0)
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False
        if self.overlap_comm:
            torch.cuda.synchronize()
            self._clear_previous_reduced_grads()
        if self.cpu_offload is False:
            for i, _ in enumerate(self.bit16_groups):
                if not i in self.averaged_gradients or self.averaged_gradients[i] is None:
                    self.averaged_gradients[i] = self.get_flat_partition(self.params_in_partition[i], self.first_offset[i], self.partition_size[i], dtype=self.dtype, device=torch.cuda.current_device(), return_tensor_list=True)
                else:
                    avg_new = self.get_flat_partition(self.params_in_partition[i], self.first_offset[i], self.partition_size[i], dtype=self.dtype, device=torch.cuda.current_device(), return_tensor_list=True)
                    for accumulated_grad, new_avg_grad in zip(self.averaged_gradients[i], avg_new):
                        accumulated_grad.add_(new_avg_grad)
        self._release_ipg_buffers()
        self.zero_grad()
        see_memory_usage(f'End ipg_epilogue')

    def reset_partition_gradient_structures(self):
        for i, _ in enumerate(self.bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
            for partition_id in range(total_partitions):
                self.is_partition_reduced[i][partition_id] = False
                self.remaining_grads_in_partition[i][partition_id] = self.total_grads_in_partition[i][partition_id]
                for param_id in self.is_grad_computed[i][partition_id]:
                    self.is_grad_computed[i][partition_id][param_id] = False

    def initialize_gradient_partition(self, i, param_group, partition_id):

        def set_key_value_list(dictionary, key, value):
            if key in dictionary:
                dictionary[key].append(value)
            else:
                dictionary[key] = [value]

        def increment_value(dictionary, key):
            if key in dictionary:
                dictionary[key] += 1
            else:
                dictionary[key] = 1
        partition_size = self.partition_size[i]
        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)
        current_index = 0
        first_offset = 0
        for param in param_group:
            param_size = param.numel()
            param_id = self.get_param_id(param)
            if current_index >= start_index and current_index < end_index:
                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)
                self.is_grad_computed[i][partition_id][param_id] = False
                self.grad_partition_insertion_offset[i][partition_id][param_id] = current_index - start_index
                self.grad_start_offset[i][partition_id][param_id] = 0
            elif start_index > current_index and start_index < current_index + param_size:
                assert first_offset == 0, 'This can happen either zero or only once as this must be the first tensor in the partition'
                first_offset = start_index - current_index
                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)
                self.is_grad_computed[i][partition_id][param_id] = False
                self.grad_partition_insertion_offset[i][partition_id][param_id] = 0
                self.grad_start_offset[i][partition_id][param_id] = first_offset
            current_index = current_index + param_size

    def overlapping_partition_gradients_reduce_epilogue(self):
        self.independent_gradient_partition_epilogue()

    def create_reduce_and_remove_grad_hooks(self):
        self.grad_accs = []
        for i, param_group in enumerate(self.bit16_groups):
            for param in param_group:
                if param.requires_grad:

                    def wrapper(param, i):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)
                        grad_acc.register_hook(reduce_partition_and_remove_grads)
                        self.grad_accs.append(grad_acc)
                    wrapper(param, i)

    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]

    def report_ipg_memory_usage(self, tag, param_elems):
        elem_count = self.elements_in_ipg_bucket + param_elems
        percent_of_bucket_size = 100.0 * elem_count // self.reduce_bucket_size
        see_memory_usage(f'{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}')

    def flatten_dense_tensors_aligned(self, tensor_list, alignment):
        return self.flatten(align_dense_tensors(tensor_list, alignment))

    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        if self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
            self.report_ipg_memory_usage('In ipg_remove_grads before reduce_ipg_grads', param.numel())
            self.reduce_ipg_grads()
            if self.contiguous_gradients and self.overlap_comm:
                self.ipg_index = 1 - self.ipg_index
            self.report_ipg_memory_usage('In ipg_remove_grads after reduce_ipg_grads', param.numel())
        param_id = self.get_param_id(param)
        assert self.params_already_reduced[param_id] == False, f'The parameter {param_id} has already been reduced.             Gradient computed twice for this partition.             Multiple gradient reduction is currently not supported'
        if param.numel() > self.reduce_bucket_size:
            self.extra_large_param_to_reduce = param
        elif self.contiguous_gradients:
            new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(0, self.elements_in_ipg_bucket, param.numel())
            new_grad_tensor.copy_(param.grad.view(-1))
            param.grad.data = new_grad_tensor.data.view_as(param.grad)
        self.elements_in_ipg_bucket += param.numel()
        assert param.grad is not None, f'rank {dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient'
        self.grads_in_ipg_bucket.append(param.grad)
        self.params_in_ipg_bucket.append((i, param, param_id))
        if is_moe_param(param):
            self.ipg_bucket_has_moe_params = True
        self.report_ipg_memory_usage('End ipg_remove_grads', 0)

    def print_rank_0(self, message):
        if dist.get_rank() == 0:
            logger.info(message)

    def gradient_reduction_w_predivide(self, tensor):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        tensor_to_allreduce = tensor
        if self.communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor
        if self.postscale_gradients:
            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1.0 / self.gradient_predivide_factor)
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
            if self.gradient_predivide_factor != dp_world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor / dp_world_size)
        else:
            tensor_to_allreduce.div_(dp_world_size)
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
        if self.communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)
        return tensor

    def average_tensor(self, tensor):
        if self.overlap_comm:
            torch.cuda.synchronize()
            stream = self.reduction_stream
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            if not self.reduce_scatter:
                self.gradient_reduction_w_predivide(tensor)
                return
            rank_and_offsets = []
            real_dp_process_group = []
            curr_size = 0
            prev_id = -1
            process_group = self.dp_process_group
            for i, param, param_id in self.params_in_ipg_bucket:
                process_group = self.dp_process_group
                if self.ipg_bucket_has_moe_params:
                    process_group = self.expert_dp_process_group[param.group_name] if is_moe_param(param) else self.dp_process_group
                    param.grad.data.div_(dist.get_world_size(group=process_group))
                partition_ids = self.param_to_partition_ids[i][param_id]
                assert all([(p_id < dist.get_world_size(group=process_group)) for p_id in partition_ids]), f'world size {dist.get_world_size(group=process_group)} and p_ids: {partition_ids}'
                partition_size = self.partition_size[i]
                partition_ids_w_offsets = []
                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))
                partition_ids_w_offsets.sort(key=lambda t: t[1])
                for idx in range(len(partition_ids_w_offsets)):
                    partition_id, offset = partition_ids_w_offsets[idx]
                    if idx == len(partition_ids_w_offsets) - 1:
                        numel = param.numel() - offset
                    else:
                        numel = partition_ids_w_offsets[idx + 1][1] - offset
                    if partition_id == prev_id:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = prev_pid, prev_size, prev_numel + numel
                    else:
                        rank_and_offsets.append((partition_id, curr_size, numel))
                        real_dp_process_group.append(process_group)
                    curr_size += numel
                    prev_id = partition_id
            if not self.ipg_bucket_has_moe_params:
                tensor.div_(dist.get_world_size(group=self.dp_process_group))
            tensor_to_reduce = tensor
            if self.communication_data_type != tensor.dtype:
                tensor_to_reduce = tensor
            async_handles = []
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor_to_reduce.narrow(0, int(bucket_offset), int(numel))
                dst_rank = dist.get_global_rank(real_dp_process_group[i], dst)
                async_handle = dist.reduce(grad_slice, dst=dst_rank, group=real_dp_process_group[i], async_op=True)
                async_handles.append(async_handle)
            for handle in async_handles:
                handle.wait()
            if self.communication_data_type != tensor.dtype:
                tensor.copy_(tensor_to_reduce)

    def get_grad_position(self, group_id, tensor_list, first_offset, partition_size):
        current_offset = 0
        for i, tensor in enumerate(tensor_list):
            param_id = self.get_param_id(tensor)
            param_start_offset = 0
            num_elements = tensor.numel()
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset
                param_start_offset = first_offset
            if num_elements > partition_size - current_offset:
                num_elements = partition_size - current_offset
            self.grad_position[param_id] = [int(group_id), int(param_start_offset), int(current_offset), int(num_elements)]
            current_offset += num_elements

    def update_overflow_tracker_for_param_grad(self, param):
        if param.grad is not None and self._has_inf_or_nan(param.grad.data):
            self.local_overflow = True

    def async_accumulate_grad_in_cpu_via_gpu(self, param):
        param_id = self.get_param_id(param)
        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]
        dest_buffer = self.temp_grad_buffer_for_gpu_offload.view(-1).narrow(0, 0, param.numel())

        def buffer_to_accumulate_to_in_cpu():
            if not self.fp16_master_weights_and_gradients:
                return torch.zeros(param.numel(), dtype=param.dtype, device=self.device).pin_memory()
            else:
                return self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)

        def accumulate_gradients():
            if not self.fp16_master_weights_and_gradients:
                dest_buffer.copy_(self.accumulated_grads_in_cpu[param_id].view(-1), non_blocking=True)
                param.grad.data.view(-1).add_(dest_buffer)
            else:
                dest_buffer.narrow(0, source_offset, num_elements).copy_(self.accumulated_grads_in_cpu[param_id].view(-1), non_blocking=True)
                param.grad.data.view(-1).narrow(0, source_offset, num_elements).add_(dest_buffer.narrow(0, source_offset, num_elements))

        def copy_gradients_to_cpu():
            if not self.fp16_master_weights_and_gradients:
                self.accumulated_grads_in_cpu[param_id].data.copy_(param.grad.data.view(-1), non_blocking=True)
            else:
                self.accumulated_grads_in_cpu[param_id].data.copy_(param.grad.data.view(-1).narrow(0, source_offset, num_elements), non_blocking=True)
        if param_id not in self.accumulated_grads_in_cpu:
            self.accumulated_grads_in_cpu[param_id] = buffer_to_accumulate_to_in_cpu()
        if self.micro_step_id > 0:
            accumulate_gradients()
        if not self.is_gradient_accumulation_boundary:
            copy_gradients_to_cpu()

    def set_norm_for_param_grad(self, param):
        param_id = self.get_param_id(param)
        accumulated_grad = self.accumulated_grads_in_cpu[param_id] if self.gradient_accumulation_steps > 1 else param.grad
        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]
        start = source_offset
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)
        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    def set_norm_for_param_grad_in_gpu(self, param):
        param_id = self.get_param_id(param)
        accumulated_grad = param.grad
        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]
        start = source_offset
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)
        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    def async_inplace_copy_grad_to_fp32_buffer_from_gpu(self, param):
        param_id = self.get_param_id(param)
        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]
        dest_tensor = self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)
        src_tensor = param.grad.view(-1).narrow(0, source_offset, num_elements)
        if not self.fp16_master_weights_and_gradients:
            src_tensor = src_tensor.float()
        dest_tensor.copy_(src_tensor, non_blocking=True)
        param.grad = None

    def complete_grad_norm_calculation_for_cpu_offload(self, params):
        total_norm = 0.0
        norm_type = 2.0
        for p in params:
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue
            if is_model_parallel_parameter(p) or self.model_parallel_rank == 0:
                param_id = self.get_param_id(p)
                if param_id in self.norm_for_param_grads:
                    param_norm = self.norm_for_param_grads[param_id]
                    total_norm += param_norm.item() ** 2
                else:
                    assert self.ignore_unused_parameters, """
                        This assert indicates that your module has parameters that
                        were not used in producing loss.
                        You can avoid this assert by
                        (1) enable ignore_unused_parameters option in zero_optimization config;
                        (2) making sure all trainable parameters and `forward` function
                            outputs participate in calculating loss.
                    """
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)
        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)
        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1
        return total_norm

    def copy_grads_in_partition(self, param):
        if self.cpu_offload:
            if self.gradient_accumulation_steps > 1:
                self.async_accumulate_grad_in_cpu_via_gpu(param)
            if self.is_gradient_accumulation_boundary:
                self.set_norm_for_param_grad_in_gpu(param)
                self.update_overflow_tracker_for_param_grad(param)
                self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)
            return
        if self.grads_in_partition is None:
            self.grads_in_partition_offset = 0
            total_size = 0
            for group in self.params_in_partition:
                for param_in_partition in group:
                    total_size += param_in_partition.numel()
            see_memory_usage(f'before copying {total_size} gradients into partition')
            self.grads_in_partition = torch.empty(int(total_size), dtype=self.dtype, device=torch.cuda.current_device())
            see_memory_usage(f'after copying {total_size} gradients into partition')
        new_grad_tensor = self.grads_in_partition.view(-1).narrow(0, self.grads_in_partition_offset, param.numel())
        new_grad_tensor.copy_(param.grad.view(-1))
        param.grad.data = new_grad_tensor.data.view_as(param.grad)
        self.grads_in_partition_offset += param.numel()

    def reduce_ipg_grads(self):
        if self.contiguous_gradients:
            if self.extra_large_param_to_reduce is not None:
                assert len(self.params_in_ipg_bucket) == 1, "more than 1 param in ipg bucket, this shouldn't happen"
                _, _, param_id = self.params_in_ipg_bucket[0]
                assert self.get_param_id(self.extra_large_param_to_reduce) == param_id, 'param in ipg bucket does not match extra-large param'
                self.average_tensor(self.extra_large_param_to_reduce.grad.view(-1))
                self.extra_large_param_to_reduce = None
            else:
                self.average_tensor(self.ipg_buffer[self.ipg_index])
        else:
            self.buffered_reduce_fallback(None, self.grads_in_ipg_bucket, elements_per_buffer=self.elements_in_ipg_bucket)
        if self.overlap_comm:
            stream = self.reduction_stream
        elif self.cpu_offload:
            stream = torch.cuda.current_stream()
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for _, param, param_id in self.params_in_ipg_bucket:
                assert self.params_already_reduced[param_id] == False, f'The parameter {param_id} has already been reduced.                     Gradient computed twice for this partition.                     Multiple gradient reduction is currently not supported'
                self.params_already_reduced[param_id] = True
                if self.partition_gradients:
                    if not self.is_param_in_current_partition[param_id]:
                        if self.overlap_comm and self.contiguous_gradients is False:
                            if self.previous_reduced_grads is None:
                                self.previous_reduced_grads = []
                            self.previous_reduced_grads.append(param)
                        else:
                            param.grad = None
                    elif self.contiguous_gradients:
                        self.copy_grads_in_partition(param)
                elif self.contiguous_gradients and self.is_param_in_current_partition[param_id]:
                    self.copy_grads_in_partition(param)
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.ipg_bucket_has_moe_params = False
        self.elements_in_ipg_bucket = 0

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        if self.partition_gradients or self.is_gradient_accumulation_boundary:
            self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def zero_reduced_gradients(self, partition_id, i):

        def are_all_related_partitions_reduced(params_id):
            for partition_id in self.param_to_partition_ids[i][params_id]:
                if not self.is_partition_reduced[i][partition_id]:
                    return False
            return True
        for params_id in self.is_grad_computed[i][partition_id]:
            if are_all_related_partitions_reduced(params_id):
                self.param_dict[params_id].grad = None

    def flatten_and_print(self, message, tensors, start=0, n=5):
        flatten_tensor = self.flatten(tensors)

        def print_func():
            logger.info(flatten_tensor.contiguous().view(-1).narrow(0, start, n))
        self.sequential_execution(print_func, message)

    def get_grads_to_reduce(self, i, partition_id):

        def get_reducible_portion(key):
            grad = self.param_dict[key].grad
            total_elements = grad.numel()
            start = self.grad_start_offset[i][partition_id][key]
            num_elements = min(total_elements - start, self.partition_size[i] - self.grad_partition_insertion_offset[i][partition_id][key])
            if not pg_correctness_test:
                if num_elements == total_elements:
                    return grad
                else:
                    return grad.contiguous().view(-1).narrow(0, int(start), int(num_elements))
            elif num_elements == total_elements:
                return grad.clone()
            else:
                return grad.clone().contiguous().view(-1).narrow(0, int(start), int(num_elements))
        grads_to_reduce = []
        for key in self.is_grad_computed[i][partition_id]:
            grad = get_reducible_portion(key)
            grads_to_reduce.append(grad)
        return grads_to_reduce

    def sequential_execution(self, function, message, group=None):
        if group is None:
            group = self.dp_process_group
        if dist.get_rank(group=group) == 0:
            logger.info(message)
        for id in range(dist.get_world_size(group=group)):
            if id == dist.get_rank(group=group):
                function()
            dist.barrier(group=group)

    def set_none_gradients_to_zero(self, i, partition_id):
        for param_id in self.is_grad_computed[i][partition_id]:
            param = self.param_dict[param_id]
            if param.grad is None:
                param.grad = torch.zero_like(param)

    def allreduce_bucket(self, bucket, rank=None, log=None):
        rank = None
        tensor = self.flatten(bucket)
        tensor_to_allreduce = tensor
        if pg_correctness_test:
            communication_data_type = torch.float32
        else:
            communication_data_type = self.communication_data_type
        if communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor
        tensor_to_allreduce.div_(dist.get_world_size(group=self.dp_process_group))
        if rank is None:
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
        else:
            global_rank = dist.get_global_rank(self.dp_process_group, rank)
            dist.reduce(tensor_to_allreduce, global_rank, group=self.dp_process_group)
        if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                tensor.copy_(tensor_to_allreduce)
        return tensor

    def _clear_previous_reduced_grads(self):
        if self.previous_reduced_grads is not None:
            for param in self.previous_reduced_grads:
                param.grad = None
            self.previous_reduced_grads = None

    def allreduce_and_copy(self, small_bucket, rank=None, log=None):
        if self.overlap_comm:
            torch.cuda.synchronize()
            self._clear_previous_reduced_grads()
            stream = self.reduction_stream
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            allreduced = self.allreduce_bucket(small_bucket, rank=rank, log=log)
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
                    buf.copy_(synced)

    def allreduce_no_retain(self, bucket, numel_per_bucket=500000000, rank=None, log=None):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, rank=rank, log=None)
                small_bucket = []
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, rank=rank, log=log)

    def buffered_reduce_fallback(self, rank, grads, elements_per_buffer=500000000, log=None):
        split_buckets = split_half_float_double(grads)
        for i, bucket in enumerate(split_buckets):
            self.allreduce_no_retain(bucket, numel_per_bucket=elements_per_buffer, rank=rank, log=log)

    def get_data_parallel_partitions(self, tensor, group_id):
        partitions = []
        dp = dist.get_world_size(group=self.real_dp_process_group[group_id])
        total_num_elements = tensor.numel()
        base_size = total_num_elements // dp
        remaining = total_num_elements % dp
        start = 0
        for id in range(dp):
            partition_size = base_size
            if id < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def get_partition_info(self, tensor_list, partition_size, partition_id):
        params_in_partition = []
        params_not_in_partition = []
        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)
        current_index = 0
        first_offset = 0
        for tensor in tensor_list:
            tensor_size = tensor.numel()
            if current_index >= start_index and current_index < end_index:
                params_in_partition.append(tensor)
            elif start_index > current_index and start_index < current_index + tensor_size:
                params_in_partition.append(tensor)
                assert first_offset == 0, 'This can happen either zero or only once as this must be the first tensor in the partition'
                first_offset = start_index - current_index
            else:
                params_not_in_partition.append(tensor)
            current_index = current_index + tensor_size
        return params_in_partition, params_not_in_partition, first_offset

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        for group in self.bit16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        """
        if self.model_parallel_group is None or self.model_parallel_world_size == 1:
            pass
        else:
            dist.all_reduce(tensor=tensor, op=op, group=self.model_parallel_group)

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        """Clips gradient norm of an iterable of parameters.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(g.data.abs().max() for g in gradients)
            total_norm_cuda = torch.FloatTensor([float(total_norm)])
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.MAX)
            total_norm = total_norm_cuda[0].item()
        else:
            total_norm = 0.0
            for g, p in zip(gradients, params):
                if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                    continue
                if is_model_parallel_parameter(p) or self.model_parallel_rank == 0:
                    param_norm = g.data.double().norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm_cuda = torch.FloatTensor([float(total_norm)])
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)
            total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)
        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1
        return total_norm

    def get_flat_partition(self, tensor_list, first_offset, partition_size, dtype, device, return_tensor_list=False):
        flat_tensor_list = []
        current_size = 0
        for i, tensor in enumerate(tensor_list):
            if tensor.grad is None:
                tensor.grad = torch.zeros_like(tensor)
            tensor = tensor.grad
            num_elements = tensor.numel()
            tensor_offset = 0
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset
            if num_elements > partition_size - current_size:
                num_elements = partition_size - current_size
            if tensor_offset > 0 or num_elements < tensor.numel():
                flat_tensor_list.append(tensor.contiguous().view(-1).narrow(0, int(tensor_offset), int(num_elements)))
            else:
                flat_tensor_list.append(tensor)
            current_size = current_size + num_elements
        if current_size < partition_size:
            flat_tensor_list.append(torch.zeros(int(partition_size - current_size), dtype=dtype, device=device))
        if return_tensor_list:
            return flat_tensor_list
        return self.flatten(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None

    def reset_cpu_buffers(self):
        self.norm_for_param_grads = {}
        self.local_overflow = False

    def log_timers(self, timer_names):
        if self.timers is None:
            return
        self.timers.log(names=list(timer_names))

    def start_timers(self, timer_names):
        if self.timers is None:
            return
        for name in timer_names:
            self.timers(name).start()

    def stop_timers(self, timer_names):
        if self.timers is None:
            return
        for name in timer_names:
            self.timers(name).stop()

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def override_loss_scale(self, loss_scale):
        if loss_scale != self.external_loss_scale:
            logger.info(f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    def scaled_global_norm(self, norm_type=2):
        assert norm_type == 2, 'only L2 norm supported'
        norm_groups = []
        for i, group in enumerate(self.bit16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                norm_groups.append(self.complete_grad_norm_calculation_for_cpu_offload(self.params_in_partition[i]))
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
            else:
                norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.params_in_partition[i]))
        if self.has_moe_layers:
            self._average_expert_grad_norms(norm_groups)
        return get_global_norm(norm_list=norm_groups)

    def get_bit16_param_group(self, group_no):
        bit16_partitions = self.parallel_partitioned_bit16_groups[group_no]
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_no])
        return [bit16_partitions[dist.get_rank(group=self.real_dp_process_group[group_no])]]

    def _optimizer_step(self, group_no):
        original_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [original_param_groups[group_no]]
        if type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half:
            self.optimizer.step(fp16_param_groups=[self.get_bit16_param_group(group_no)])
        else:
            self.optimizer.step()
        self.optimizer.param_groups = original_param_groups

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        self.micro_step_id = -1
        see_memory_usage(f'In step before checking overflow')
        self.check_overflow()
        OPTIMIZER_ALLGATHER = 'optimizer_allgather'
        OPTIMIZER_GRADIENTS = 'optimizer_gradients'
        OPTIMIZER_STEP = 'optimizer_step'
        timer_names = [OPTIMIZER_ALLGATHER, OPTIMIZER_GRADIENTS, OPTIMIZER_STEP]
        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if dist.get_rank() == 0:
                logger.info('[deepspeed] OVERFLOW! Rank {} Skipping step. Attempted loss scale: {}, reducing to {}'.format(dist.get_rank(), prev_scale, self.loss_scale))
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad()
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}
            see_memory_usage('After overflow after clearing gradients')
            self.start_timers(timer_names)
            self.stop_timers(timer_names)
            return
        see_memory_usage('Before norm calculation')
        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / self.loss_scale
        see_memory_usage('After norm before optimizer')
        for i, group in enumerate(self.bit16_groups):
            self.start_timers([OPTIMIZER_GRADIENTS])
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
                self.stop_timers([OPTIMIZER_GRADIENTS])
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)
                if not (type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half):
                    bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                    fp32_partition = self.single_partition_of_fp32_groups[i]
                    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.stop_timers([OPTIMIZER_STEP])
            else:
                self.free_grad_in_param_list(self.params_not_in_partition[i])
                if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(self.averaged_gradients[i], int(self.partition_size[i]))
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[i])
                assert single_grad_partition.numel() == self.partition_size[i], 'averaged gradients have different number of elements that partition size {} {} {} {}'.format(single_grad_partition.numel(), self.partition_size[i], i, partition_id)
                self.single_partition_of_fp32_groups[i].grad = single_grad_partition
                self.free_grad_in_param_list(self.params_in_partition[i])
                self.averaged_gradients[i] = None
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
                self.stop_timers([OPTIMIZER_GRADIENTS])
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)
                self.single_partition_of_fp32_groups[i].grad = None
                del single_grad_partition
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.stop_timers([OPTIMIZER_STEP])
        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            self.reset_cpu_buffers()
        self.start_timers([OPTIMIZER_ALLGATHER])
        all_gather_dp_groups(partitioned_param_groups=self.parallel_partitioned_bit16_groups, dp_process_group=self.real_dp_process_group, start_alignment_factor=self.nccl_start_alignment_factor, allgather_bucket_size=self.allgather_bucket_size)
        self.stop_timers([OPTIMIZER_ALLGATHER])
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)
        self.log_timers(timer_names)
        see_memory_usage('After zero_optimizer step')
        return

    @torch.no_grad()
    def update_lp_params(self):
        for i, (bit16_partitions, fp32_partition) in enumerate(zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            bit16_partitions[partition_id].data.copy_(fp32_partition.data)
        all_gather_dp_groups(partitioned_param_groups=self.parallel_partitioned_bit16_groups, dp_process_group=self.real_dp_process_group, start_alignment_factor=self.nccl_start_alignment_factor, allgather_bucket_size=self.allgather_bucket_size)

    def _average_expert_grad_norms(self, norm_groups):
        for i, norm in enumerate(norm_groups):
            if self.is_moe_param_group[i]:
                scaled_norm = norm * 1.0 / float(dist.get_world_size(group=self.real_dp_process_group[i]))
                scaled_norm_tensor = torch.tensor(scaled_norm, device='cuda', dtype=torch.float)
                dist.all_reduce(scaled_norm_tensor, group=self.real_dp_process_group[i])
                norm_groups[i] = scaled_norm_tensor.item()

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        combined_scale = self.loss_scale
        if self.clip_grad > 0.0:
            clip = (total_norm / self.loss_scale + 1e-06) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale
        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1.0 / combined_scale)
            else:
                grad.data.mul_(1.0 / combined_scale)

    def _check_overflow(self, partition_gradients=True):
        self.overflow = self.has_overflow(partition_gradients)

    def has_overflow_serial(self, params, is_grad_list=False):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True
        return False

    def has_overflow_partitioned_grads_serial(self):
        for i in range(len(self.bit16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial()
            overflow_gpu = torch.ByteTensor([overflow])
            """This will capture overflow across all data parallel and expert parallel process
            Since expert parallel process are a subset of data parallel process"""
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)
        else:
            params = []
            for group in self.bit16_groups:
                for param in group:
                    params.append(param)
            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = torch.ByteTensor([overflow])
        self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    @staticmethod
    def _has_inf_or_nan(x, j=None):
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

    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        self.micro_step_id += 1
        if self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(int(self.reduce_bucket_size), dtype=self.dtype, device=torch.cuda.current_device())
            self.ipg_buffer.append(buf_0)
            if self.overlap_comm:
                buf_1 = torch.empty(int(self.reduce_bucket_size), dtype=self.dtype, device=torch.cuda.current_device())
                self.ipg_buffer.append(buf_1)
            self.ipg_index = 0
        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    def check_overflow(self, partition_gradients=True):
        self._check_overflow(partition_gradients)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value
    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value
    param_groups = property(_get_param_groups, _set_param_groups)

    def _get_loss_scale(self):
        if self.custom_loss_scaler:
            return self.external_loss_scale
        else:
            return self.loss_scaler.cur_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value
    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    def _get_groups_without_padding(self, groups_with_padding):
        groups_without_padding = []
        for i, group in enumerate(groups_with_padding):
            lean_length = group.numel() - self.groups_padding[i]
            groups_without_padding.append(group[:lean_length])
        return groups_without_padding

    def _get_state_without_padding(self, state_with_padding, padding):
        lean_state = {}
        for key, value in state_with_padding.items():
            if torch.is_tensor(value):
                lean_length = value.numel() - padding
                lean_state[key] = value[:lean_length]
            else:
                lean_state[key] = value
        return lean_state

    def _get_base_optimizer_state(self):
        optimizer_groups_state = []
        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            lean_optimizer_state = self._get_state_without_padding(self.optimizer.state[p], self.groups_padding[i])
            optimizer_groups_state.append(lean_optimizer_state)
        return optimizer_groups_state

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict[CLIP_GRAD] = self.clip_grad
        if self.elastic_checkpoint:
            state_dict[BASE_OPTIMIZER_STATE] = self._get_base_optimizer_state()
        else:
            state_dict[BASE_OPTIMIZER_STATE] = self.optimizer.state_dict()
        fp32_groups_without_padding = self._get_groups_without_padding(self.single_partition_of_fp32_groups)
        state_dict[SINGLE_PARTITION_OF_FP32_GROUPS] = fp32_groups_without_padding
        state_dict[ZERO_STAGE] = ZeroStageEnum.gradients if self.partition_gradients else ZeroStageEnum.optimizer_states
        state_dict[GROUP_PADDINGS] = self.groups_padding
        state_dict[PARTITION_COUNT] = self.partition_count
        state_dict[DS_VERSION] = version
        state_dict[PARAM_SLICE_MAPPINGS] = self._param_slice_mappings
        return state_dict

    def _restore_from_elastic_fp32_weights(self, all_state_dict):
        merged_single_partition_of_fp32_groups = []
        for i in range(len(self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            merged_partitions = [sd[SINGLE_PARTITION_OF_FP32_GROUPS][i] for sd in all_state_dict]
            if self.is_moe_group(self.optimizer.param_groups[i]):
                ranks = self.get_ep_ranks(group_name=self.optimizer.param_groups[i]['name'])
                merged_partitions = [merged_partitions[i] for i in ranks]
            flat_merged_partitions = self.flatten_dense_tensors_aligned(merged_partitions, self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i]))
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions, i)
            merged_single_partition_of_fp32_groups.append(dp_partitions[partition_id])
        for current, saved in zip(self.single_partition_of_fp32_groups, merged_single_partition_of_fp32_groups):
            current.data.copy_(saved.data)

    def _restore_from_bit16_weights(self):
        for group_id, (bit16_partitions, fp32_partition) in enumerate(zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])
            fp32_partition.data.copy_(bit16_partitions[partition_id].data)

    def refresh_fp32_params(self):
        self._restore_from_bit16_weights()

    def _partition_base_optimizer_state(self, state_key, all_partition_states, group_id):
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])
        alignment = dist.get_world_size(group=self.real_dp_process_group[group_id])
        if torch.is_tensor(all_partition_states[0]):
            flat_merged_partitions = self.flatten_dense_tensors_aligned(all_partition_states, alignment)
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions, group_id)
            return dp_partitions[partition_id]
        else:
            return all_partition_states[0]

    def _restore_base_optimizer_state(self, base_optimizer_group_states):
        if type(base_optimizer_group_states) == dict:
            base_optimizer_group_states = base_optimizer_group_states['state']
        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            for key, saved in base_optimizer_group_states[i].items():
                if torch.is_tensor(self.optimizer.state[p][key]):
                    dst_tensor = self.optimizer.state[p][key]
                    src_tensor = _get_padded_tensor(saved, dst_tensor.numel())
                    self.optimizer.state[p][key].data.copy_(src_tensor.data)
                else:
                    self.optimizer.state[p][key] = saved

    def get_ep_ranks(self, rank=0, group_name=None):
        expert_parallel_size_ = groups._get_expert_parallel_world_size(group_name)
        world_size = groups._get_data_parallel_world_size()
        rank = groups._get_expert_parallel_rank(group_name)
        ranks = range(rank, world_size, expert_parallel_size_)
        return list(ranks)

    def _restore_elastic_base_optimizer_state(self, all_state_dict):
        base_optimizer_group_states = []
        for i in range(len(self.optimizer.param_groups)):
            partition_states = {}
            all_partition_group_states = [sd[BASE_OPTIMIZER_STATE][i] for sd in all_state_dict]
            if self.is_moe_group(self.optimizer.param_groups[i]):
                ranks = self.get_ep_ranks(group_name=self.optimizer.param_groups[i]['name'])
                all_partition_group_states = [all_partition_group_states[i] for i in ranks]
            for key in all_partition_group_states[0].keys():
                all_partition_states = [all_states[key] for all_states in all_partition_group_states]
                partition_states[key] = self._partition_base_optimizer_state(key, all_partition_states, i)
            base_optimizer_group_states.append(partition_states)
        self._restore_base_optimizer_state(base_optimizer_group_states)

    def load_state_dict(self, state_dict_list, load_optimizer_states=True, load_from_fp32_weights=False, checkpoint_folder=None):
        if checkpoint_folder:
            self._load_universal_checkpoint(checkpoint_folder, load_optimizer_states, load_from_fp32_weights)
        else:
            self._load_legacy_checkpoint(state_dict_list, load_optimizer_states, load_from_fp32_weights)

    def _load_universal_checkpoint(self, checkpoint_folder, load_optimizer_states, load_from_fp32_weights):
        self._load_hp_checkpoint_state(checkpoint_folder)

    @property
    def param_groups(self):
        """Forward the wrapped optimizer's parameters."""
        return self.optimizer.param_groups

    def _load_hp_checkpoint_state(self, checkpoint_dir):
        checkpoint_dir = os.path.join(checkpoint_dir, 'zero')
        tp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        tp_world_size = self.mpu.get_slice_parallel_world_size()
        for i, _ in enumerate(self.optimizer.param_groups):
            for lp in self.bit16_groups[i]:
                if lp._hp_mapping is not None:
                    lp.load_hp_checkpoint_state(os.path.join(checkpoint_dir, self.param_names[lp]), tp_rank, tp_world_size)

    def _load_legacy_checkpoint(self, state_dict_list, load_optimizer_states=True, load_from_fp32_weights=False):
        """Loading ZeRO checkpoint

        Arguments:
            state_dict_list: List of all saved ZeRO checkpoints, one for each saved partition.
                Note that the number of saved partitions may differ from number of loading partitions to support
                changing GPU count, specifically DP world size, between saving and loading checkpoints.
            load_optimizer_states: Boolean indicating whether or not to load base optimizer states
            load_from_fp32_weights: Boolean indicating whether to initialize fp32 master weights from fp32
            copies in checkpoints (no precision loss) or from model's fp16 copies (with precision loss).
        """
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        dp_rank = dist.get_rank(group=self.dp_process_group)
        current_rank_sd = state_dict_list[dp_rank]
        self.loss_scaler = current_rank_sd.get('loss_scaler', self.loss_scaler)
        self.dynamic_loss_scale = current_rank_sd.get('dynamic_loss_scale', self.dynamic_loss_scale)
        self.overflow = current_rank_sd.get('overflow', self.overflow)
        self.clip_grad = current_rank_sd.get(CLIP_GRAD, self.clip_grad)
        ckpt_version = current_rank_sd.get(DS_VERSION, False)
        assert ckpt_version, f'Empty ds_version in checkpoint, not clear how to proceed'
        ckpt_version = pkg_version.parse(ckpt_version)
        if not self.partition_gradients:
            required_version = pkg_version.parse('0.3.17')
            error_str = f"ZeRO stage 1 changed in {required_version} and is not backwards compatible with older stage 1 checkpoints. If you'd like to load an old ZeRO-1 checkpoint please use an older version of DeepSpeed (<= 0.5.8) and set 'legacy_stage1': true in your zero config json."
            assert required_version <= ckpt_version, f'Old version: {ckpt_version} {error_str}'
        ckpt_is_rigid = isinstance(current_rank_sd[BASE_OPTIMIZER_STATE], dict)
        if load_optimizer_states:
            if ckpt_is_rigid:
                self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])
            elif self.elastic_checkpoint:
                self._restore_elastic_base_optimizer_state(state_dict_list)
            else:
                self._restore_base_optimizer_state(current_rank_sd[BASE_OPTIMIZER_STATE])
        if load_from_fp32_weights:
            if self.elastic_checkpoint and not ckpt_is_rigid:
                self._restore_from_elastic_fp32_weights(state_dict_list)
            else:
                for current, saved in zip(self.single_partition_of_fp32_groups, current_rank_sd[SINGLE_PARTITION_OF_FP32_GROUPS]):
                    src_tensor = _get_padded_tensor(saved, current.numel())
                    current.data.copy_(src_tensor.data)
        else:
            self._restore_from_bit16_weights()
        if load_optimizer_states:
            self._link_all_hp_params()


class Eigenvalue(object):

    def __init__(self, verbose=False, max_iter=100, tol=0.01, stability=0, gas_boundary_resolution=1, layer_name='', layer_num=0):
        super().__init__()
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.stability = stability
        self.gas_boundary_resolution = gas_boundary_resolution
        self.layer_name = layer_name
        self.layer_num = layer_num
        assert len(self.layer_name) > 0 and layer_num > 0
        log_dist(f'enabled eigenvalue with verbose={verbose}, max_iter={max_iter}, tol={tol}, stability={stability}, gas_boundary_resolution={gas_boundary_resolution}, layer_name={layer_name}, layer_num={layer_num}', ranks=[0])

    def nan_to_num(self, x):
        device = x.device
        x = x.cpu().numpy()
        x = np.nan_to_num(x=x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x)

    def normalize(self, v):
        norm_squared = self.inner_product(v, v)
        norm = norm_squared ** 0.5 + self.stability
        normalized_vectors = [(vector / norm) for vector in v]
        normalized_vectors = [self.nan_to_num(vector) for vector in normalized_vectors]
        return normalized_vectors

    def inner_product(self, xs, ys):
        return sum([torch.sum(x * y) for x, y in zip(xs, ys)])

    def get_layers(self, module):
        scope_names = self.layer_name.split('.')
        assert len(scope_names) > 0
        m = module
        for name in scope_names:
            assert hasattr(m, name), 'layer_name configuration is invalid.'
            m = getattr(m, name)
        return m

    def compute_eigenvalue(self, module, device=None, scale=1.0):
        block_eigenvalue = []
        param_keys = []
        layers = self.get_layers(module)
        for block in range(self.layer_num):
            model_block = layers[block]
            rng_state = torch.random.get_rng_state()
            if device is None:
                v = [torch.randn(p.size()) for p in model_block.parameters() if p.grad is not None and p.grad.grad_fn is not None]
            else:
                v = [torch.randn(p.size(), device=device) for p in model_block.parameters() if p.grad is not None and p.grad.grad_fn is not None]
            torch.random.set_rng_state(rng_state)
            grads = [param.grad for param in model_block.parameters() if param.grad is not None and param.grad.grad_fn is not None]
            params = [param for param in model_block.parameters() if param.grad is not None and param.grad.grad_fn is not None]
            layer_keys = [id(p) for p in model_block.parameters()]
            param_keys.append(layer_keys)
            v = self.normalize(v)
            if len(grads) == 0 or len(params) == 0:
                log_dist(f'The model does NOT support eigenvalue computation.', ranks=[0], level=logging.WARNING)
                return []
            i = 0
            eigenvalue_current, eigenvalue_previous = 1.0, 0.0
            while i < self.max_iter and abs(eigenvalue_current) > 0 and abs((eigenvalue_current - eigenvalue_previous) / eigenvalue_current) >= self.tol:
                eigenvalue_previous = eigenvalue_current
                Hv = torch.autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)
                Hv = [self.nan_to_num(hv).float() for hv in Hv]
                eigenvalue_current = self.inner_product(Hv, v).item()
                v = self.normalize(Hv)
                v = [(x / scale) for x in v]
                i += 1
            eigenvalue_current *= scale
            block_eigenvalue.append(eigenvalue_current)
            if self.verbose:
                log_dist(f'block: {block}, power iteration: {i}, eigenvalue: {eigenvalue_current}', ranks=[0])
        block_eigenvalue = self.post_process(block_eigenvalue)
        if self.verbose:
            log_dist(f'post processed block_eigenvalue: {block_eigenvalue}', ranks=[0])
        ev_dict = {}
        for i, (layer_keys, value) in enumerate(zip(param_keys, block_eigenvalue)):
            ev_dict.update(dict.fromkeys(layer_keys, (value, i)))
        return ev_dict

    def post_process(self, value_list):
        max_value = abs(max(value_list, key=abs))
        return [(abs(v) / max_value if v != 0.0 else 1.0) for v in value_list]


BACKWARD_INNER_MICRO_TIMER = 'backward_inner_microstep'


BACKWARD_MICRO_TIMER = 'backward_microstep'


BACKWARD_REDUCE_MICRO_TIMER = 'backward_allreduce_microstep'


FORWARD_GLOBAL_TIMER = 'forward'


FORWARD_MICRO_TIMER = 'forward_microstep'


STEP_GLOBAL_TIMER = 'step'


STEP_MICRO_TIMER = 'step_microstep'


class EngineTimers(object):
    """Wallclock timers for DeepSpeedEngine"""

    def __init__(self, enable_micro_timers, enable_global_timers):
        self.forward_timers = []
        self.backward_timers = []
        self.backward_inner_timers = []
        self.backward_reduce_timers = []
        self.step_timers = []
        self.global_timers = []
        self.micro_timers = []
        if enable_micro_timers:
            self.forward_timers += [FORWARD_MICRO_TIMER]
            self.backward_timers += [BACKWARD_MICRO_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_MICRO_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_MICRO_TIMER]
            self.step_timers += [STEP_MICRO_TIMER]
            self.micro_timers += [FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER, STEP_MICRO_TIMER]
        if enable_global_timers:
            self.forward_timers += [FORWARD_GLOBAL_TIMER]
            self.backward_timers += [BACKWARD_GLOBAL_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_GLOBAL_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_GLOBAL_TIMER]
            self.step_timers += [STEP_GLOBAL_TIMER]
            self.global_timers += [FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER, STEP_GLOBAL_TIMER]


class CheckOverflow(object):
    """Checks for overflow in gradient across parallel process"""

    def __init__(self, param_groups=None, mpu=None, zero_reduce_scatter=False, deepspeed=None):
        self.mpu = mpu
        self.params = [] if param_groups else None
        self.zero_reduce_scatter = zero_reduce_scatter
        self.deepspeed = deepspeed
        self.has_moe_params = False
        if param_groups:
            for group in param_groups:
                for param in group:
                    self.params.append(param)
                    if is_moe_param(param):
                        self.has_moe_params = True

    def check_using_norm(self, norm_group, reduce_overflow=True):
        overflow = -1 in norm_group
        overflow_gpu = torch.FloatTensor([overflow])
        if self.has_moe_params:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=groups._get_max_expert_parallel_group())
        if self.mpu is not None:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
        elif reduce_overflow:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX)
            dist.barrier()
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    def check(self, param_groups=None):
        params = []
        has_moe_params = False
        if param_groups is None:
            params = self.params
            has_moe_params = self.has_moe_params
        else:
            assert param_groups is not None, 'self.params and param_groups both cannot be none'
            for group in param_groups:
                for param in group:
                    params.append(param)
                    if is_moe_param(param):
                        has_moe_params = True
        return self.has_overflow(params, has_moe_params=has_moe_params)

    def has_overflow_serial(self, params):
        for i, p in enumerate(params):
            if p.grad is not None and self._has_inf_or_nan(p.grad.data, i):
                return True
        return False

    def has_overflow(self, params, has_moe_params=None):
        if has_moe_params is None:
            has_moe_params = self.has_moe_params
        overflow = self.has_overflow_serial(params)
        overflow_gpu = torch.ByteTensor([overflow])
        if has_moe_params:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=groups._get_max_expert_parallel_group())
        if self.zero_reduce_scatter:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        elif self.mpu is not None:
            if self.deepspeed is not None:
                using_pipeline = hasattr(self.deepspeed, 'pipeline_enable_backward_allreduce')
                if using_pipeline and self.deepspeed.pipeline_enable_backward_allreduce is False or not using_pipeline and self.deepspeed.enable_backward_allreduce is False:
                    dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.mpu.get_data_parallel_group())
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
        elif self.deepspeed is not None and self.deepspeed.enable_backward_allreduce is False:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    @staticmethod
    def _has_inf_or_nan(x, i):
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


OPTIMIZER_STATE_DICT = 'optimizer_state_dict'


def get_grad_norm(parameters, norm_type=2, mpu=None):
    """Get grad norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.0
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue
            if tensor_mp_rank > 0 and not is_model_parallel_parameter(p):
                continue
            param_norm = p.grad.data.float().norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)
    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1
    return total_norm


def get_weight_norm(parameters, norm_type=2, mpu=None):
    """Get norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.0
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue
            if tensor_mp_rank > 0 and not is_model_parallel_parameter(p):
                continue
            param_norm = p.data.float().norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)
    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1
    return total_norm


class FP16_Optimizer(DeepSpeedOptimizer):
    """
   FP16 Optimizer for training fp16 models. Handles loss scaling.

   For usage example please see, TODO:  DeepSpeed V2 Tutorial
    """

    def __init__(self, init_optimizer, deepspeed=None, static_loss_scale=1.0, dynamic_loss_scale=False, initial_dynamic_scale=2 ** 32, dynamic_loss_args=None, verbose=True, mpu=None, clip_grad=0.0, fused_adam_legacy=False, has_moe_layers=False, timers=None):
        self.fused_adam_legacy = fused_adam_legacy
        self.timers = timers
        self.deepspeed = deepspeed
        self.has_moe_layers = has_moe_layers
        self.using_pipeline = self.deepspeed.pipeline_parallelism
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA.')
        self.optimizer = init_optimizer
        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.fp32_groups_flat = []
        self._global_grad_norm = 0.0
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.fp16_groups.append(param_group['params'])
            self.fp16_groups_flat.append(_flatten_dense_tensors([p.clone().detach() for p in self.fp16_groups[i]]))
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
            self.fp32_groups_flat.append(self.fp16_groups_flat[i].clone().float().detach())
            self.fp32_groups_flat[i].requires_grad = True
            param_group['params'] = [self.fp32_groups_flat[i]]
        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2
            if dynamic_loss_args is None:
                self.cur_scale = initial_dynamic_scale
                self.scale_window = 1000
                self.min_loss_scale = 1
            else:
                self.cur_scale = dynamic_loss_args[INITIAL_LOSS_SCALE]
                self.scale_window = dynamic_loss_args[SCALE_WINDOW]
                self.min_loss_scale = dynamic_loss_args[MIN_LOSS_SCALE]
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale
        self.verbose = verbose
        self.custom_loss_scaler = False
        self.external_loss_scale = None
        self.clip_grad = clip_grad
        self.norm_type = 2
        self.step_count = 0
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        if TORCH_MAJOR == 0 and TORCH_MINOR <= 4:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm
        else:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
        self.mpu = mpu
        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups, mpu=self.mpu, deepspeed=deepspeed)
        self.initialize_optimizer_states()

    def initialize_optimizer_states(self):
        for i, group in enumerate(self.fp16_groups):
            self.fp32_groups_flat[i].grad = torch.zeros(self.fp32_groups_flat[i].size(), device=self.fp32_groups_flat[i].device)
        self.optimizer.step()
        for i, group in enumerate(self.fp16_groups):
            self.fp32_groups_flat[i].grad = None
        return

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step_fused_adam(self, closure=None):
        """
        Not supporting closure.
        """
        grads_groups_flat = []
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            grads_groups_flat.append(_flatten_dense_tensors([(torch.zeros(p.size(), dtype=p.dtype, device=p.device) if p.grad is None else p.grad) for p in group]))
            norm_groups.append(get_weight_norm(grads_groups_flat[i], mpu=self.mpu))
        self.overflow = self.overflow_checker.check_using_norm(norm_groups)
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                logger.info('[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: {}, reducing to {}'.format(prev_scale, self.cur_scale))
            return self.overflow
        scaled_grad_norm = get_global_norm(norm_list=norm_groups)
        combined_scale = self.unscale_and_clip_grads(grads_groups_flat, scaled_grad_norm, apply_scale=False)
        self._global_grad_norm = scaled_grad_norm / self.cur_scale
        self.optimizer.step(grads=[[g] for g in grads_groups_flat], output_params=[[p] for p in self.fp16_groups_flat], scale=combined_scale, grad_norms=norm_groups)
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
        return self.overflow

    def start_timers(self, name_list):
        if self.timers is not None:
            for name in name_list:
                self.timers(name).start()

    def stop_timers(self, name_list):
        if self.timers is not None:
            for name in name_list:
                self.timers(name).stop()

    def log_timers(self, name_list):
        if self.timers is not None:
            self.timers.log(name_list)

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def override_loss_scale(self, loss_scale):
        if loss_scale != self.external_loss_scale:
            logger.info(f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        if self.fused_adam_legacy:
            return self.step_fused_adam()
        COMPUTE_NORM = 'compute_norm'
        OVERFLOW_CHECK = 'overflow_check'
        OVERFLOW_TIMERS = [COMPUTE_NORM, OVERFLOW_CHECK]
        UNSCALE_AND_CLIP = 'unscale_and_clip'
        BASIC_STEP = 'basic_step'
        UPDATE_FP16 = 'update_fp16'
        STEP_TIMERS = OVERFLOW_TIMERS + [UNSCALE_AND_CLIP, BASIC_STEP, UPDATE_FP16]
        self.start_timers([OVERFLOW_CHECK])
        fp16_params = []
        for i, group in enumerate(self.fp16_groups):
            fp16_params.extend([p for p in group if p.grad is not None])
        self.overflow = self.overflow_checker.has_overflow(fp16_params)
        self.stop_timers([OVERFLOW_CHECK])
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                log_dist(f'Overflow detected. Skipping step. Attempted loss scale: {prev_scale}, reducing to {self.cur_scale}', ranks=[0])
            for i, group in enumerate(self.fp16_groups):
                for p in group:
                    p.grad = None
            self.log_timers(OVERFLOW_TIMERS)
            return self.overflow
        grads_groups_flat = []
        for i, group in enumerate(self.fp16_groups):
            data_type = self.fp32_groups_flat[i].dtype
            grads_groups_flat.append(_flatten_dense_tensors([(torch.zeros(p.size(), dtype=data_type, device=p.device) if p.grad is None else p.grad) for p in group]))
            for p in group:
                p.grad = None
            self.fp32_groups_flat[i].grad = grads_groups_flat[i]
        self.start_timers([COMPUTE_NORM])
        all_groups_norm = get_grad_norm(self.fp32_groups_flat, mpu=self.mpu)
        self.stop_timers([COMPUTE_NORM])
        if self.has_moe_layers:
            all_groups_norm = self._get_norm_with_moe_layers(all_groups_norm)
        scaled_global_grad_norm = get_global_norm(norm_list=[all_groups_norm])
        self._global_grad_norm = scaled_global_grad_norm / self.cur_scale
        self.start_timers([UNSCALE_AND_CLIP])
        self.unscale_and_clip_grads(grads_groups_flat, scaled_global_grad_norm)
        self.stop_timers([UNSCALE_AND_CLIP])
        self.start_timers([BASIC_STEP])
        self.optimizer.step()
        self.stop_timers([BASIC_STEP])
        for group in self.fp32_groups_flat:
            group.grad = None
        self.start_timers([UPDATE_FP16])
        for i in range(len(self.fp16_groups)):
            updated_params = _unflatten_dense_tensors(self.fp32_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data.copy_(q.data)
        self.stop_timers([UPDATE_FP16])
        self.log_timers(STEP_TIMERS)
        self.step_count += 1
        return self.overflow

    def _get_norm_with_moe_layers(self, all_groups_norm):
        if self.using_pipeline:
            pg = self.deepspeed.mpu.get_data_parallel_group()
        else:
            pg = groups._get_data_parallel_group()
        scaled_norm = all_groups_norm * 1.0 / float(dist.get_world_size(group=pg))
        scaled_norm_tensor = torch.tensor(scaled_norm, device=self.fp32_groups_flat[0].device, dtype=torch.float)
        dist.all_reduce(scaled_norm_tensor, group=pg)
        all_groups_norm = scaled_norm_tensor.item()
        return all_groups_norm

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm, apply_scale=True):
        combined_scale = self.cur_scale
        if self.clip_grad > 0.0:
            clip = (total_norm / self.cur_scale + 1e-06) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.cur_scale
        if apply_scale:
            for grad in grad_groups_flat:
                grad.data.mul_(1.0 / combined_scale)
        return combined_scale

    def backward(self, loss, create_graph=False, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            scaled_loss = loss.float() * self.cur_scale
            scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            prev_scale = self.cur_scale
            if skip:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_loss_scale)
                self.last_overflow_iter = self.cur_iter
                if self.verbose:
                    logger.info(f'\nGrad overflow on iteration {self.cur_iter}')
                    logger.info(f'Reducing dynamic loss scale from {prev_scale} to {self.cur_scale}')
            else:
                stable_interval = self.cur_iter - self.last_overflow_iter - 1
                if stable_interval > 0 and stable_interval % self.scale_window == 0:
                    self.cur_scale *= self.scale_factor
                    if self.verbose:
                        logger.info(f'No Grad overflow for {self.scale_window} iterations')
                        logger.info(f'Increasing dynamic loss scale from {prev_scale} to {self.cur_scale}')
        elif skip:
            logger.info('Grad overflow on iteration: %s', self.cur_iter)
            logger.info('Using static loss scale of: %s', self.cur_scale)
        self.cur_iter += 1
        return

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value
    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value
    param_groups = property(_get_param_groups, _set_param_groups)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict[OPTIMIZER_STATE_DICT] = self.optimizer.state_dict()
        state_dict['fp32_groups_flat'] = self.fp32_groups_flat
        state_dict[CLIP_GRAD] = self.clip_grad
        return state_dict

    def refresh_fp32_params(self):
        for current, saved in zip(self.fp32_groups_flat, self.fp16_groups_flat):
            current.data.copy_(saved.data)

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict[OPTIMIZER_STATE_DICT])
        self.clip_grad = state_dict[CLIP_GRAD]
        for current, saved in zip(self.fp32_groups_flat, state_dict['fp32_groups_flat']):
            current.data.copy_(saved.data)

    def __repr__(self):
        return repr(self.optimizer)

    def _get_loss_scale(self):
        if self.custom_loss_scaler:
            return self.external_loss_scale
        else:
            return self.cur_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value
    loss_scale = property(_get_loss_scale, _set_loss_scale)


def split_params_grads_into_shared_and_expert_params(group: List[torch.nn.Parameter]) ->Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Split grad of parameters into grads of non-expert params
    and grads of expert params. This is useful while computing
    grad-norms for clipping and overflow detection

        group (List[torch.nn.Parameter]):
    Args:
            The group of parameters to split

    Returns:
        Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        list of gradients for non MoE params, list of gradients of MoE params
    """
    expert_grads = []
    shared_grads = []
    for p in group:
        if p.grad is not None:
            if is_moe_param(p):
                expert_grads.append(p.grad)
            else:
                shared_grads.append(p.grad)
    return shared_grads, expert_grads


class FP16_UnfusedOptimizer(DeepSpeedOptimizer):
    """
    FP16 Optimizer without weight fusion to support LAMB optimizer

    For usage example please see, TODO:  DeepSpeed V2 Tutorial
    """

    def __init__(self, init_optimizer, deepspeed=None, static_loss_scale=1.0, dynamic_loss_scale=False, dynamic_loss_args=None, verbose=True, mpu=None, clip_grad=0.0, fused_lamb_legacy=False):
        self.fused_lamb_legacy = fused_lamb_legacy
        self._global_grad_norm = 0.0
        if dist.get_rank() == 0:
            logger.info(f'Fused Lamb Legacy : {self.fused_lamb_legacy} ')
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA.')
        self.optimizer = init_optimizer
        self.fp16_groups = []
        self.fp32_groups = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.fp16_groups.append(param_group['params'])
            fp32_group = [p.clone().float().detach() for p in param_group['params']]
            for p in fp32_group:
                p.requires_grad = True
            self.fp32_groups.append(fp32_group)
            param_group['params'] = self.fp32_groups[i]
        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2.0
            if dynamic_loss_args is None:
                self.cur_scale = 1.0 * 2 ** 16
                self.scale_window = 1000
                self.min_loss_scale = 0.25
            else:
                self.cur_scale = dynamic_loss_args[INITIAL_LOSS_SCALE]
                self.scale_window = dynamic_loss_args[SCALE_WINDOW]
                self.min_loss_scale = dynamic_loss_args[MIN_LOSS_SCALE]
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale
        self.custom_loss_scaler = False
        self.external_loss_scale = None
        self.verbose = verbose
        self.clip_grad = clip_grad
        self.norm_type = 2
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        if TORCH_MAJOR == 0 and TORCH_MINOR <= 4:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm
        else:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
        self.mpu = mpu
        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups, mpu=self.mpu, deepspeed=deepspeed)
        self.initialize_optimizer_states()

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step_fused_lamb(self, closure=None):
        """
        Not supporting closure.
        """
        grads_groups_flat = []
        grads_groups = []
        norm_groups = []
        expert_norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            grads = [(torch.zeros(p.size(), dtype=p.dtype, device=p.device) if p.grad is None else p.grad) for p in group]
            grads_groups.append(grads)
            grads_groups_flat.append(_flatten_dense_tensors(grads))
            grads_for_norm, expert_grads_for_norm = split_params_grads_into_shared_and_expert_params(group)
            norm_group_value = 0.0
            if len(grads_for_norm) > 0:
                norm_group_value = get_weight_norm(_flatten_dense_tensors(grads_for_norm), mpu=self.mpu)
            norm_groups.append(norm_group_value)
            expert_norm_group_value = 0.0
            if len(expert_grads_for_norm) > 0:
                expert_norm_group_value = get_weight_norm(_flatten_dense_tensors(expert_grads_for_norm), mpu=self.mpu)
            expert_norm_groups.append(expert_norm_group_value)
        self.overflow = self.overflow_checker.check_using_norm(norm_groups + expert_norm_groups)
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                logger.info('[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: {}, reducing to {}'.format(prev_scale, self.cur_scale))
            return self.overflow
        self._global_grad_norm = get_global_norm(norm_list=norm_groups)
        combined_scale = self.unscale_and_clip_grads(self._global_grad_norm, apply_scale=False)
        self.optimizer.step(grads=grads_groups, output_params=self.fp16_groups, scale=combined_scale)
        for fp32_group, fp16_group in zip(self.fp32_groups, self.fp16_groups):
            for idx, (fp32_param, fp16_param) in enumerate(zip(fp32_group, fp16_group)):
                fp32_param.grad = None
                fp16_param.data.copy_(fp32_param.data)
        return self.overflow

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def override_loss_scale(self, loss_scale):
        if loss_scale != self.external_loss_scale:
            logger.info(f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        if self.fused_lamb_legacy:
            return self.step_fused_lamb()
        self.overflow = self.overflow_checker.check()
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                logger.info('[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: {}, reducing to {}'.format(prev_scale, self.cur_scale))
            return self.overflow
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            grads_for_norm, _ = split_params_grads_into_shared_and_expert_params(group)
            norm_group_value = 0.0
            if len(grads_for_norm) > 0:
                norm_group_value = get_weight_norm(grads_for_norm, mpu=self.mpu)
            norm_groups.append(norm_group_value)
            for fp32_param, fp16_param in zip(self.fp32_groups[i], self.fp16_groups[i]):
                if fp16_param.grad is None:
                    fp32_param.grad = torch.zeros(fp16_param.size(), dtype=fp32_param.dtype, device=fp32_param.device)
                else:
                    fp32_param.grad = fp16_param.grad
        self._global_grad_norm = get_global_norm(norm_list=norm_groups)
        self.unscale_and_clip_grads(self._global_grad_norm)
        self.optimizer.step()
        for fp32_group, fp16_group in zip(self.fp32_groups, self.fp16_groups):
            for idx, (fp32_param, fp16_param) in enumerate(zip(fp32_group, fp16_group)):
                fp32_param.grad = None
                fp16_param.data.copy_(fp32_param.data)
        return self.overflow

    def unscale_and_clip_grads(self, total_norm, apply_scale=True):
        combined_scale = self.cur_scale
        if self.clip_grad > 0.0:
            clip = (total_norm / self.cur_scale + 1e-06) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.cur_scale
        if apply_scale:
            for group in self.fp32_groups:
                for param in group:
                    if param.grad is not None:
                        param.grad.data.mul_(1.0 / combined_scale)
        return combined_scale

    def backward(self, loss, create_graph=False, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            scaled_loss = loss.float() * self.cur_scale
            scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            prev_scale = self.cur_scale
            if skip:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_loss_scale)
                self.last_overflow_iter = self.cur_iter
                if self.verbose:
                    logger.info('Grad overflow on iteration: %s', self.cur_iter)
                    logger.info(f'Reducing dynamic loss scale from {prev_scale} to {self.cur_scale}')
            else:
                stable_interval = self.cur_iter - self.last_overflow_iter - 1
                if stable_interval > 0 and stable_interval % self.scale_window == 0:
                    self.cur_scale *= self.scale_factor
                    if self.verbose:
                        logger.info(f'No Grad overflow for {self.scale_window} iterations')
                        logger.info(f'Increasing dynamic loss scale from {prev_scale} to {self.cur_scale}')
        elif skip:
            logger.info('Grad overflow on iteration %s', self.cur_iter)
            logger.info('Using static loss scale of %s', self.cur_scale)
        self.cur_iter += 1
        return

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value
    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value
    param_groups = property(_get_param_groups, _set_param_groups)

    def _get_loss_scale(self):
        if self.custom_loss_scaler:
            return self.external_loss_scale
        else:
            return self.cur_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value
    loss_scale = property(_get_loss_scale, _set_loss_scale)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict[OPTIMIZER_STATE_DICT] = self.optimizer.state_dict()
        state_dict['fp32_groups'] = self.fp32_groups
        return state_dict

    def refresh_fp32_params(self):
        for current_group, saved_group in zip(self.fp32_groups, self.fp16_groups):
            for current, saved in zip(current_group, saved_group):
                current.data.copy_(saved.data)

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict[OPTIMIZER_STATE_DICT])
        for current_group, saved_group in zip(self.fp32_groups, state_dict['fp32_groups']):
            for current, saved in zip(current_group, saved_group):
                current.data.copy_(saved.data)

    def __repr__(self):
        return repr(self.optimizer)

    def initialize_optimizer_states(self):
        for i, group in enumerate(self.fp16_groups):
            for param in group:
                param.grad = torch.zeros(param.size(), dtype=param.dtype, device=torch.cuda.current_device())
        for i, group in enumerate(self.fp32_groups):
            for param in group:
                param.grad = torch.zeros(param.size(), dtype=param.dtype, device=torch.cuda.current_device())
        self.optimizer.step()
        for i, group in enumerate(self.fp16_groups):
            for param in group:
                param.grad = None
        for i, group in enumerate(self.fp32_groups):
            for param in group:
                param.grad = None


def _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    flops += w_ih.shape[0] * w_ih.shape[1]
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        flops += rnn_module.hidden_size
        flops += rnn_module.hidden_size * 3
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        flops += rnn_module.hidden_size * 4
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def _rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]
    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


def _rnn_forward_hook(rnn_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers
    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]
    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {nn.RNN: _rnn_forward_hook, nn.GRU: _rnn_forward_hook, nn.LSTM: _rnn_forward_hook, nn.RNNCell: _rnn_cell_forward_hook, nn.LSTMCell: _rnn_cell_forward_hook, nn.GRUCell: _rnn_cell_forward_hook}


def _batch_norm_flops_compute(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    has_affine = weight is not None
    if training:
        return input.numel() * (5 if has_affine else 4), 0
    flops = input.numel() * (2 if has_affine else 1)
    return flops, 0


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _conv_flops_compute(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert weight.shape[1] * groups == input.shape[1]
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])
    length = len(input_dims)
    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length
    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)
    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs
    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count
    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _conv_trans_flops_compute(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])
    length = len(input_dims)
    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length
    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)
    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)
    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs
    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))
    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _elu_flops_compute(input: Tensor, alpha: float=1.0, inplace: bool=False):
    return input.numel(), 0


def _embedding_flops_compute(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return 0, 0


def _gelu_flops_compute(input):
    return input.numel(), 0


def _group_norm_flops_compute(input: Tensor, num_groups: int, weight: Optional[Tensor]=None, bias: Optional[Tensor]=None, eps: float=1e-05):
    has_affine = weight is not None
    return input.numel() * (5 if has_affine else 4), 0


def _instance_norm_flops_compute(input: Tensor, running_mean: Optional[Tensor]=None, running_var: Optional[Tensor]=None, weight: Optional[Tensor]=None, bias: Optional[Tensor]=None, use_input_stats: bool=True, momentum: float=0.1, eps: float=1e-05):
    has_affine = weight is not None
    return input.numel() * (5 if has_affine else 4), 0


def _layer_norm_flops_compute(input: Tensor, normalized_shape: List[int], weight: Optional[Tensor]=None, bias: Optional[Tensor]=None, eps: float=1e-05):
    has_affine = weight is not None
    return input.numel() * (5 if has_affine else 4), 0


def _leaky_relu_flops_compute(input: Tensor, negative_slope: float=0.01, inplace: bool=False):
    return input.numel(), 0


def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = input.numel() * out_features
    return 2 * macs, macs


def _pool_flops_compute(input, kernel_size, stride=None, padding=0, dilation=None, ceil_mode=False, count_include_pad=True, divisor_override=None, return_indices=None):
    return input.numel(), 0


def _prelu_flops_compute(input: Tensor, weight: Tensor):
    return input.numel(), 0


def _relu6_flops_compute(input: Tensor, inplace: bool=False):
    return input.numel(), 0


def _relu_flops_compute(input, inplace=False):
    return input.numel(), 0


def _silu_flops_compute(input: Tensor, inplace: bool=False):
    return input.numel(), 0


def _softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return input.numel(), 0


def _upsample_flops_compute(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    if size is not None:
        if isinstance(size, tuple):
            return int(_prod(size)), 0
        else:
            return int(size), 0
    assert scale_factor is not None, 'either size or scale_factor should be defined'
    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        flops * scale_factor ** len(input)
    return flops, 0


module_flop_count = []


module_mac_count = []


old_functions = {}


def wrapFunc(func, funcFlopCompute):
    oldFunc = func
    name = func.__name__
    old_functions[name] = oldFunc

    def newFunc(*args, **kwds):
        flops, macs = funcFlopCompute(*args, **kwds)
        if module_flop_count:
            module_flop_count[-1].append((name, flops))
        if module_mac_count and macs:
            module_mac_count[-1].append((name, macs))
        return oldFunc(*args, **kwds)
    newFunc.__name__ = func.__name__
    return newFunc


def _patch_functionals():
    F.linear = wrapFunc(F.linear, _linear_flops_compute)
    F.conv1d = wrapFunc(F.conv1d, _conv_flops_compute)
    F.conv2d = wrapFunc(F.conv2d, _conv_flops_compute)
    F.conv3d = wrapFunc(F.conv3d, _conv_flops_compute)
    F.conv_transpose1d = wrapFunc(F.conv_transpose1d, _conv_trans_flops_compute)
    F.conv_transpose2d = wrapFunc(F.conv_transpose2d, _conv_trans_flops_compute)
    F.conv_transpose3d = wrapFunc(F.conv_transpose3d, _conv_trans_flops_compute)
    F.relu = wrapFunc(F.relu, _relu_flops_compute)
    F.prelu = wrapFunc(F.prelu, _prelu_flops_compute)
    F.elu = wrapFunc(F.elu, _elu_flops_compute)
    F.leaky_relu = wrapFunc(F.leaky_relu, _leaky_relu_flops_compute)
    F.relu6 = wrapFunc(F.relu6, _relu6_flops_compute)
    if hasattr(F, 'silu'):
        F.silu = wrapFunc(F.silu, _silu_flops_compute)
    F.gelu = wrapFunc(F.gelu, _gelu_flops_compute)
    F.batch_norm = wrapFunc(F.batch_norm, _batch_norm_flops_compute)
    F.layer_norm = wrapFunc(F.layer_norm, _layer_norm_flops_compute)
    F.instance_norm = wrapFunc(F.instance_norm, _instance_norm_flops_compute)
    F.group_norm = wrapFunc(F.group_norm, _group_norm_flops_compute)
    F.avg_pool1d = wrapFunc(F.avg_pool1d, _pool_flops_compute)
    F.avg_pool2d = wrapFunc(F.avg_pool2d, _pool_flops_compute)
    F.avg_pool3d = wrapFunc(F.avg_pool3d, _pool_flops_compute)
    F.max_pool1d = wrapFunc(F.max_pool1d, _pool_flops_compute)
    F.max_pool2d = wrapFunc(F.max_pool2d, _pool_flops_compute)
    F.max_pool3d = wrapFunc(F.max_pool3d, _pool_flops_compute)
    F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, _pool_flops_compute)
    F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, _pool_flops_compute)
    F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, _pool_flops_compute)
    F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, _pool_flops_compute)
    F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, _pool_flops_compute)
    F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, _pool_flops_compute)
    F.upsample = wrapFunc(F.upsample, _upsample_flops_compute)
    F.interpolate = wrapFunc(F.interpolate, _upsample_flops_compute)
    F.softmax = wrapFunc(F.softmax, _softmax_flops_compute)
    F.embedding = wrapFunc(F.embedding, _embedding_flops_compute)


def _elementwise_flops_compute(input, other):
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape), 0
        else:
            return 1, 0
    elif not torch.is_tensor(other):
        return _prod(input.shape), 0
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)
        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = _prod(final_shape)
        return flops, 0


def _add_flops_compute(input, other, *, alpha=1, out=None):
    return _elementwise_flops_compute(input, other)


def _addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(input.shape), macs


def _einsum_flops_compute(equation, *operands):
    """
    Count flops for the einsum operation.
    """
    equation = equation.replace(' ', '')
    input_shapes = [o.shape for o in operands]
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): (97 + i) for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)
    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize='optimal')[1]
    for line in optim.split('\n'):
        if 'optimized flop' in line.lower():
            flop = int(float(line.split(':')[-1]))
            return flop, 0
    raise NotImplementedError('Unsupported einsum operation.')


def _matmul_flops_compute(input, other, *, out=None):
    """
    Count flops for the matmul operation.
    """
    macs = _prod(input.shape) * other.shape[-1]
    return 2 * macs, macs


def _mul_flops_compute(input, other, *, out=None):
    return _elementwise_flops_compute(input, other)


def _tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the tensor addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(self.shape), macs


def _patch_tensor_methods():
    torch.matmul = wrapFunc(torch.matmul, _matmul_flops_compute)
    torch.Tensor.matmul = wrapFunc(torch.Tensor.matmul, _matmul_flops_compute)
    torch.mm = wrapFunc(torch.mm, _matmul_flops_compute)
    torch.Tensor.mm = wrapFunc(torch.Tensor.mm, _matmul_flops_compute)
    torch.bmm = wrapFunc(torch.bmm, _matmul_flops_compute)
    torch.Tensor.bmm = wrapFunc(torch.bmm, _matmul_flops_compute)
    torch.addmm = wrapFunc(torch.addmm, _addmm_flops_compute)
    torch.Tensor.addmm = wrapFunc(torch.Tensor.addmm, _tensor_addmm_flops_compute)
    torch.mul = wrapFunc(torch.mul, _mul_flops_compute)
    torch.Tensor.mul = wrapFunc(torch.Tensor.mul, _mul_flops_compute)
    torch.add = wrapFunc(torch.add, _add_flops_compute)
    torch.Tensor.add = wrapFunc(torch.Tensor.add, _add_flops_compute)
    torch.einsum = wrapFunc(torch.einsum, _einsum_flops_compute)


def _reload_functionals():
    F.linear = old_functions[F.linear.__name__]
    F.conv1d = old_functions[F.conv1d.__name__]
    F.conv2d = old_functions[F.conv2d.__name__]
    F.conv3d = old_functions[F.conv3d.__name__]
    F.conv_transpose1d = old_functions[F.conv_transpose1d.__name__]
    F.conv_transpose2d = old_functions[F.conv_transpose2d.__name__]
    F.conv_transpose3d = old_functions[F.conv_transpose3d.__name__]
    F.relu = old_functions[F.relu.__name__]
    F.prelu = old_functions[F.prelu.__name__]
    F.elu = old_functions[F.elu.__name__]
    F.leaky_relu = old_functions[F.leaky_relu.__name__]
    F.relu6 = old_functions[F.relu6.__name__]
    F.batch_norm = old_functions[F.batch_norm.__name__]
    F.avg_pool1d = old_functions[F.avg_pool1d.__name__]
    F.avg_pool2d = old_functions[F.avg_pool2d.__name__]
    F.avg_pool3d = old_functions[F.avg_pool3d.__name__]
    F.max_pool1d = old_functions[F.max_pool1d.__name__]
    F.max_pool2d = old_functions[F.max_pool2d.__name__]
    F.max_pool3d = old_functions[F.max_pool3d.__name__]
    F.adaptive_avg_pool1d = old_functions[F.adaptive_avg_pool1d.__name__]
    F.adaptive_avg_pool2d = old_functions[F.adaptive_avg_pool2d.__name__]
    F.adaptive_avg_pool3d = old_functions[F.adaptive_avg_pool3d.__name__]
    F.adaptive_max_pool1d = old_functions[F.adaptive_max_pool1d.__name__]
    F.adaptive_max_pool2d = old_functions[F.adaptive_max_pool2d.__name__]
    F.adaptive_max_pool3d = old_functions[F.adaptive_max_pool3d.__name__]
    F.upsample = old_functions[F.upsample.__name__]
    F.interpolate = old_functions[F.interpolate.__name__]
    F.softmax = old_functions[F.softmax.__name__]
    F.embedding = old_functions[F.embedding.__name__]


def _reload_tensor_methods():
    torch.matmul = old_functions[torch.matmul.__name__]


def duration_to_string(duration, units=None, precision=2):
    if units is None:
        if duration > 1:
            return str(round(duration, precision)) + ' s'
        elif duration * 10 ** 3 > 1:
            return str(round(duration * 10 ** 3, precision)) + ' ms'
        elif duration * 10 ** 6 > 1:
            return str(round(duration * 10 ** 6, precision)) + ' us'
        else:
            return str(duration)
    elif units == 'us':
        return str(round(duration * 10.0 ** 6, precision)) + ' ' + units
    elif units == 'ms':
        return str(round(duration * 10.0 ** 3, precision)) + ' ' + units
    else:
        return str(round(duration, precision)) + ' s'


def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10 ** 12 > 0:
            return str(round(flops / 10.0 ** 12, precision)) + ' TFLOPS'
        if flops // 10 ** 9 > 0:
            return str(round(flops / 10.0 ** 9, precision)) + ' GFLOPS'
        elif flops // 10 ** 6 > 0:
            return str(round(flops / 10.0 ** 6, precision)) + ' MFLOPS'
        elif flops // 10 ** 3 > 0:
            return str(round(flops / 10.0 ** 3, precision)) + ' KFLOPS'
        else:
            return str(flops) + ' FLOPS'
    else:
        if units == 'TFLOPS':
            return str(round(flops / 10.0 ** 12, precision)) + ' ' + units
        if units == 'GFLOPS':
            return str(round(flops / 10.0 ** 9, precision)) + ' ' + units
        elif units == 'MFLOPS':
            return str(round(flops / 10.0 ** 6, precision)) + ' ' + units
        elif units == 'KFLOPS':
            return str(round(flops / 10.0 ** 3, precision)) + ' ' + units
        else:
            return str(flops) + ' FLOPS'


def get_module_duration(module):
    duration = module.__duration__
    if duration == 0:
        for m in module.children():
            duration += m.__duration__
    return duration


def get_module_flops(module):
    sum = module.__flops__
    for child in module.children():
        sum += get_module_flops(child)
    return sum


def get_module_macs(module):
    sum = module.__macs__
    for child in module.children():
        sum += get_module_macs(child)
    return sum


def macs_to_string(macs, units=None, precision=2):
    if units is None:
        if macs // 10 ** 9 > 0:
            return str(round(macs / 10.0 ** 9, precision)) + ' GMACs'
        elif macs // 10 ** 6 > 0:
            return str(round(macs / 10.0 ** 6, precision)) + ' MMACs'
        elif macs // 10 ** 3 > 0:
            return str(round(macs / 10.0 ** 3, precision)) + ' KMACs'
        else:
            return str(macs) + ' MACs'
    elif units == 'GMACs':
        return str(round(macs / 10.0 ** 9, precision)) + ' ' + units
    elif units == 'MMACs':
        return str(round(macs / 10.0 ** 6, precision)) + ' ' + units
    elif units == 'KMACs':
        return str(round(macs / 10.0 ** 3, precision)) + ' ' + units
    else:
        return str(macs) + ' MACs'


def num_to_string(num, precision=2):
    if num // 10 ** 9 > 0:
        return str(round(num / 10.0 ** 9, precision)) + ' G'
    elif num // 10 ** 6 > 0:
        return str(round(num / 10.0 ** 6, precision)) + ' M'
    elif num // 10 ** 3 > 0:
        return str(round(num / 10.0 ** 3, precision)) + ' K'
    else:
        return str(num)


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    elif units == 'M':
        return str(round(params_num / 10.0 ** 6, precision)) + ' ' + units
    elif units == 'K':
        return str(round(params_num / 10.0 ** 3, precision)) + ' ' + units
    else:
        return str(params_num)


class FlopsProfiler(object):
    """Measures the latency, number of estimated floating-point operations and parameters of each module in a PyTorch model.

    The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how latency, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top k modules in terms of aggregated latency, flops, and parameters at depth l with k and l specified by the user. The output profile is computed for each batch of input.
    The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package.
    When using DeepSpeed for model training, the flops profiler can be configured in the deepspeed_config file and no user code change is required.

    If using the profiler as a standalone package, one imports the flops_profiler package and use the APIs.

    Here is an example for usage in a typical training workflow:

        .. code-block:: python

            model = Model()
            prof = FlopsProfiler(model)

            for step, batch in enumerate(data_loader):
                if step == profile_step:
                    prof.start_profile()

                loss = model(batch)

                if step == profile_step:
                    flops = prof.get_total_flops(as_string=True)
                    params = prof.get_total_params(as_string=True)
                    prof.print_model_profile(profile_step=profile_step)
                    prof.end_profile()

                loss.backward()
                optimizer.step()

    To profile a trained model in inference, use the `get_model_profile` API.

    Args:
        object (torch.nn.Module): The PyTorch model to profile.
    """

    def __init__(self, model, ds_engine=None):
        self.model = model
        self.ds_engine = ds_engine
        self.started = False
        self.func_patched = False

    def start_profile(self, ignore_list=None):
        """Starts profiling.

        Extra attributes are added recursively to all the modules and the profiled torch.nn.functionals are monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to ignore while profiling. Defaults to None.
        """
        self.reset_profile()
        _patch_functionals()
        _patch_tensor_methods()

        def register_module_hooks(module, ignore_list):
            if ignore_list and type(module) in ignore_list:
                return
            if type(module) in MODULE_HOOK_MAPPING:
                if not hasattr(module, '__flops_handle__'):
                    module.__flops_handle__ = module.register_forward_hook(MODULE_HOOK_MAPPING[type(module)])
                return

            def pre_hook(module, input):
                module_flop_count.append([])
                module_mac_count.append([])
            if not hasattr(module, '__pre_hook_handle__'):
                module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)

            def post_hook(module, input, output):
                if module_flop_count:
                    module.__flops__ += sum([elem[1] for elem in module_flop_count[-1]])
                    module_flop_count.pop()
                    module.__macs__ += sum([elem[1] for elem in module_mac_count[-1]])
                    module_mac_count.pop()
            if not hasattr(module, '__post_hook_handle__'):
                module.__post_hook_handle__ = module.register_forward_hook(post_hook)

            def start_time_hook(module, input):
                torch.cuda.synchronize()
                module.__start_time__ = time.time()
            if not hasattr(module, '__start_time_hook_handle'):
                module.__start_time_hook_handle__ = module.register_forward_pre_hook(start_time_hook)

            def end_time_hook(module, input, output):
                torch.cuda.synchronize()
                module.__duration__ += time.time() - module.__start_time__
            if not hasattr(module, '__end_time_hook_handle__'):
                module.__end_time_hook_handle__ = module.register_forward_hook(end_time_hook)
        self.model.apply(partial(register_module_hooks, ignore_list=ignore_list))
        self.started = True
        self.func_patched = True

    def stop_profile(self):
        """Stop profiling.

        All torch.nn.functionals are restored to their originals.
        """
        if self.started and self.func_patched:
            _reload_functionals()
            _reload_tensor_methods()
            self.func_patched = False

        def remove_profile_attrs(module):
            if hasattr(module, '__pre_hook_handle__'):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, '__post_hook_handle__'):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, '__flops_handle__'):
                module.__flops_handle__.remove()
                del module.__flops_handle__
            if hasattr(module, '__start_time_hook_handle__'):
                module.__start_time_hook_handle__.remove()
                del module.__start_time_hook_handle__
            if hasattr(module, '__end_time_hook_handle__'):
                module.__end_time_hook_handle__.remove()
                del module.__end_time_hook_handle__
        self.model.apply(remove_profile_attrs)

    def reset_profile(self):
        """Resets the profiling.

        Adds or resets the extra attributes.
        """

        def add_or_reset_attrs(module):
            module.__flops__ = 0
            module.__macs__ = 0
            module.__params__ = sum(p.numel() for p in module.parameters())
            module.__start_time__ = 0
            module.__duration__ = 0
        self.model.apply(add_or_reset_attrs)

    def end_profile(self):
        """Ends profiling.

        The added attributes and handles are removed recursively on all the modules.
        """
        if not self.started:
            return
        self.stop_profile()
        self.started = False

        def remove_profile_attrs(module):
            if hasattr(module, '__flops__'):
                del module.__flops__
            if hasattr(module, '__macs__'):
                del module.__macs__
            if hasattr(module, '__params__'):
                del module.__params__
            if hasattr(module, '__start_time__'):
                del module.__start_time__
            if hasattr(module, '__duration__'):
                del module.__duration__
        self.model.apply(remove_profile_attrs)

    def get_total_flops(self, as_string=False):
        """Returns the total flops of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_flops = get_module_flops(self.model)
        return num_to_string(total_flops) if as_string else total_flops

    def get_total_macs(self, as_string=False):
        """Returns the total MACs of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_macs = get_module_macs(self.model)
        return macs_to_string(total_macs) if as_string else total_macs

    def get_total_duration(self, as_string=False):
        """Returns the total duration of the model forward pass.

        Args:
            as_string (bool, optional): whether to output the duration as string. Defaults to False.

        Returns:
            The latency of the model forward pass.
        """
        total_duration = get_module_duration(self.model)
        return duration_to_string(total_duration) if as_string else total_duration

    def get_total_params(self, as_string=False):
        """Returns the total parameters of the model.

        Args:
            as_string (bool, optional): whether to output the parameters as string. Defaults to False.

        Returns:
            The number of parameters in the model.
        """
        return params_to_string(self.model.__params__) if as_string else self.model.__params__

    def print_model_profile(self, profile_step=1, module_depth=-1, top_modules=1, detailed=True, output_file=None):
        """Prints the model graph with the measured profile attached to each module.

        Args:
            profile_step (int, optional): The global training step at which to profile. Note that warm up steps are needed for accurate time measurement.
            module_depth (int, optional): The depth of the model to which to print the aggregated module information. When set to -1, it prints information from the top to the innermost modules (the maximum depth).
            top_modules (int, optional): Limits the aggregated profile output to the number of top modules specified.
            detailed (bool, optional): Whether to print the detailed model profile.
            output_file (str, optional): Path to the output file. If None, the profiler prints to stdout.
        """
        if not self.started:
            return
        original_stdout = None
        f = None
        if output_file and output_file != '':
            dir_path = os.path.dirname(os.path.abspath(output_file))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            original_stdout = sys.stdout
            f = open(output_file, 'w')
            sys.stdout = f
        total_flops = self.get_total_flops()
        total_macs = self.get_total_macs()
        total_duration = self.get_total_duration()
        total_params = self.get_total_params()
        self.flops = total_flops
        self.macs = total_macs
        self.params = total_params
        None
        None
        None
        if self.ds_engine:
            None
            None
            None
            None
        None
        None
        None
        None
        None
        fwd_latency = self.get_total_duration()
        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            fwd_latency = self.ds_engine.timers('forward').elapsed(False) / 1000.0
        None
        None
        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            bwd_latency = self.ds_engine.timers('backward').elapsed(False) / 1000.0
            step_latency = self.ds_engine.timers('step').elapsed(False) / 1000.0
            None
            None
            None
            None
            iter_latency = fwd_latency + bwd_latency + step_latency
            None
            None
            samples_per_iter = self.ds_engine.train_micro_batch_size_per_gpu() * self.ds_engine.world_size
            None

        def flops_repr(module):
            params = module.__params__
            flops = get_module_flops(module)
            macs = get_module_macs(module)
            items = [params_to_string(params), '{:.2%} Params'.format(params / total_params if total_params else 0), macs_to_string(macs), '{:.2%} MACs'.format(0.0 if total_macs == 0 else macs / total_macs)]
            duration = get_module_duration(module)
            items.append(duration_to_string(duration))
            items.append('{:.2%} latency'.format(0.0 if total_duration == 0 else duration / total_duration))
            items.append(flops_to_string(0.0 if duration == 0 else flops / duration))
            items.append(module.original_extra_repr())
            return ', '.join(items)

        def add_extra_repr(module):
            flops_extra_repr = flops_repr.__get__(module)
            if module.extra_repr != flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = flops_extra_repr
                assert module.extra_repr != module.original_extra_repr

        def del_extra_repr(module):
            if hasattr(module, 'original_extra_repr'):
                module.extra_repr = module.original_extra_repr
                del module.original_extra_repr
        self.model.apply(add_extra_repr)
        None
        self.print_model_aggregated_profile(module_depth=module_depth, top_modules=top_modules)
        if detailed:
            None
            None
            None
            None
        self.model.apply(del_extra_repr)
        None
        if output_file:
            sys.stdout = original_stdout
            f.close()

    def print_model_aggregated_profile(self, module_depth=-1, top_modules=1):
        """Prints the names of the top top_modules modules in terms of aggregated time, flops, and parameters at depth module_depth.

        Args:
            module_depth (int, optional): the depth of the modules to show. Defaults to -1 (the innermost modules).
            top_modules (int, optional): the number of top modules to show. Defaults to 1.
        """
        info = {}
        if not hasattr(self.model, '__flops__'):
            None
            return

        def walk_module(module, curr_depth, info):
            if curr_depth not in info:
                info[curr_depth] = {}
            if module.__class__.__name__ not in info[curr_depth]:
                info[curr_depth][module.__class__.__name__] = [0, 0, 0]
            info[curr_depth][module.__class__.__name__][0] += get_module_macs(module)
            info[curr_depth][module.__class__.__name__][1] += module.__params__
            info[curr_depth][module.__class__.__name__][2] += get_module_duration(module)
            has_children = len(module._modules.items()) != 0
            if has_children:
                for child in module.children():
                    walk_module(child, curr_depth + 1, info)
        walk_module(self.model, 0, info)
        depth = module_depth
        if module_depth == -1:
            depth = len(info) - 1
        None
        for d in range(depth):
            num_items = min(top_modules, len(info[d]))
            sort_macs = {k: macs_to_string(v[0]) for k, v in sorted(info[d].items(), key=lambda item: item[1][0], reverse=True)[:num_items]}
            sort_params = {k: params_to_string(v[1]) for k, v in sorted(info[d].items(), key=lambda item: item[1][1], reverse=True)[:num_items]}
            sort_time = {k: duration_to_string(v[2]) for k, v in sorted(info[d].items(), key=lambda item: item[1][2], reverse=True)[:num_items]}
            None
            None
            None
            None


class MultiTensorApply(object):

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


multi_tensor_applier = MultiTensorApply(2048 * 32)


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Currently GPU-only.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=0.001, bias_correction=True, betas=(0.9, 0.999), eps=1e-08, adam_w_mode=True, weight_decay=0.0, amsgrad=False, set_grad_none=True):
        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        fused_adam_cuda = FusedAdamBuilder().load()
        self._dummy_overflow_buf = torch.IntTensor([0])
        self.multi_tensor_adam = fused_adam_cuda.multi_tensor_adam

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedAdam, self).zero_grad()

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError('FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.')
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            if 'step' not in group:
                group['step'] = 0
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = group.get('step', 0)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedAdam only support fp16 and fp32.')
            if len(g_16) > 0:
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_16, p_16, m_16, v_16], group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode, bias_correction, group['weight_decay'])
            if len(g_32) > 0:
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_32, p_32, m_32, v_32], group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode, bias_correction, group['weight_decay'])
        return loss


MEMORY_OPT_ALLREDUCE_SIZE = 500000000


class Monitor(ABC):

    @abstractmethod
    def __init__(self, monitor_config):
        self.monitor_config = monitor_config

    @abstractmethod
    def write_events(self, event_list):
        pass


class WandbMonitor(Monitor):

    def __init__(self, monitor_config):
        super().__init__(monitor_config)
        check_wandb_availability()
        self.enabled = monitor_config.wandb_config.enabled
        self.group = monitor_config.wandb_config.group
        self.team = monitor_config.wandb_config.team
        self.project = monitor_config.wandb_config.project
        if self.enabled and dist.get_rank() == 0:
            wandb.init(project=self.project, group=self.group, entity=self.team)

    def log(self, data, step=None, commit=None, sync=None):
        if self.enabled and dist.get_rank() == 0:
            return wandb.log(data, step=step, commit=commit, sync=sync)

    def write_events(self, event_list):
        if self.enabled and dist.get_rank() == 0:
            for event in event_list:
                label = event[0]
                value = event[1]
                step = event[2]
                self.log({label: value}, step=step)


class MonitorMaster(Monitor):

    def __init__(self, monitor_config):
        super().__init__(monitor_config)
        self.tb_monitor = None
        self.wandb_monitor = None
        self.csv_monitor = None
        self.enabled = monitor_config.tensorboard_enabled or monitor_config.csv_monitor_enabled or monitor_config.wandb_enabled
        if dist.get_rank() == 0:
            if monitor_config.tensorboard_enabled:
                self.tb_monitor = TensorBoardMonitor(monitor_config)
            if monitor_config.wandb_enabled:
                self.wandb_monitor = WandbMonitor(monitor_config)
            if monitor_config.csv_monitor_enabled:
                self.csv_monitor = csvMonitor(monitor_config)

    def write_events(self, event_list):
        if dist.get_rank() == 0:
            if self.tb_monitor is not None:
                self.tb_monitor.write_events(event_list)
            if self.wandb_monitor is not None:
                self.wandb_monitor.write_events(event_list)
            if self.csv_monitor is not None:
                self.csv_monitor.write_events(event_list)


PLD_GAMMA = 'gamma'


PLD_THETA = 'theta'


class ProgressiveLayerDrop(object):
    """ Progressive Layer Dropping (PLD) for model training.
        This implements the PLD technique for compressed model training
        from this paper: https://arxiv.org/pdf/2010.13369.pdf
    Args:
        theta (float): a hyper-parameter that controls the trade-off between training time and robustness.
        The lower the theta value, the faster the training speed. Default value: 0.5.
        gamma (float): a hyper-parameter that controls how fast the drop ratio increases. Default value: 0.001.
    """

    def __init__(self, theta=0.5, gamma=0.001):
        super().__init__()
        self.theta = theta
        self.gamma = gamma
        self.current_theta = 1.0
        log_dist(f'Enabled progressive layer dropping (theta = {self.theta})', ranks=[0])

    def get_state(self):
        kwargs = {'progressive_layer_drop': True, 'pld_theta': self.get_theta()}
        return kwargs

    def get_theta(self):
        return self.current_theta

    def update_state(self, global_step):

        def _prob(x, gamma, p):
            return (1.0 - p) * np.exp(-gamma * x) + p
        self.current_theta = _prob(global_step, self.gamma, self.theta)


RANDOM_LTD_GLOBAL_BATCH_SIZE = 'global_batch_size'


RANDOM_LTD_LAYER_ID = 'random_ltd_layer_id'


RANDOM_LTD_LAYER_NUM = 'random_ltd_layer_num'


ROUTE_EVAL = 'eval'


ROUTE_PREDICT = 'predict'


ROUTE_TRAIN = 'train'


RANDOM_LTD_INCREASE_STEP = 'seq_per_step'


RANDOM_LTD_MIN_VALUE = 'min_value'


RANDOM_LTD_REQUIRE_STEP = 'require_steps'


RANDOM_LTD_SCHEDULER_TYPE = 'schedule_type'


RANDOM_LTD_SCHEDULE_CONFIG = 'schedule_config'


class BaseScheduler(object):

    def __init__(self):
        self.state = {}

    def __fixed_root_get_value(self, global_steps, root_degree=None):
        s_state = self.state[RANDOM_LTD_SCHEDULE_CONFIG]
        if root_degree is None:
            root_degree = s_state['root_degree']
        next_seq = (float(global_steps) / s_state[RANDOM_LTD_REQUIRE_STEP]) ** (1.0 / root_degree)
        next_seq = math.floor(next_seq * (self.state[RANDOM_LTD_MAX_VALUE] - self.state[RANDOM_LTD_MIN_VALUE]) + self.state[RANDOM_LTD_MIN_VALUE])
        next_seq -= next_seq % s_state[RANDOM_LTD_INCREASE_STEP]
        next_seq = min(next_seq, self.state[RANDOM_LTD_MAX_VALUE])
        return next_seq

    def get_value(self, global_steps):
        if self.state[RANDOM_LTD_SCHEDULER_TYPE] == 'fixed_linear':
            return self.__fixed_root_get_value(global_steps, 1)
        else:
            raise RuntimeError('Unsupported random LTD schedule type')


RANDOM_LTD_CONSUMED_LAYER_TOKENS = 'consumed_layer_tokens'


RANDOM_LTD_CURRENT_VALUE = 'current_value'


RANDOM_LTD_CURR_STEP = 'current_steps'


RANDOM_LTD_SCHEDULER = 'random_ltd_schedule'


RANDOM_LTD_TOTAL_LAYER_NUM = 'total_layer_num'


class RandomLTDScheduler(BaseScheduler):

    def __init__(self, config):
        super().__init__()
        self.model_layer_num = config[RANDOM_LTD_TOTAL_LAYER_NUM]
        self.random_ltd_layer_num = config[RANDOM_LTD_LAYER_NUM]
        self.config_schedule = config[RANDOM_LTD_SCHEDULER]
        self.global_batch_size = config[RANDOM_LTD_GLOBAL_BATCH_SIZE]
        self.reset_to_init()
        if config[RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE][RANDOM_LTD_LAYER_TOKEN_LR_ENABLED]:
            logger.warning('**********Work In Progress************')
            raise NotImplementedError
        self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS] = 0

    def get_total_layer_tokens(self, train_iters):
        for step in range(train_iters):
            self.update_seq(step)
        return self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS]

    def reset_to_init(self):
        if self.config_schedule is not None:
            self.state[RANDOM_LTD_MIN_VALUE] = self.config_schedule[RANDOM_LTD_MIN_VALUE]
            self.state[RANDOM_LTD_MAX_VALUE] = self.config_schedule[RANDOM_LTD_MAX_VALUE]
            self.state[RANDOM_LTD_CURRENT_VALUE] = self.config_schedule[RANDOM_LTD_MIN_VALUE]
            self.state[RANDOM_LTD_SCHEDULE_CONFIG] = self.config_schedule[RANDOM_LTD_SCHEDULE_CONFIG]
            self.state[RANDOM_LTD_SCHEDULER_TYPE] = self.config_schedule[RANDOM_LTD_SCHEDULER_TYPE]
        self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS] = 0
        self.state[RANDOM_LTD_CURR_STEP] = -1

    def get_current_seq(self):
        return self.state[RANDOM_LTD_CURRENT_VALUE]

    def set_current_seq(self, seq_length):
        self.state[RANDOM_LTD_CURRENT_VALUE] = seq_length

    def get_random_ltd_layer_num(self):
        return self.random_ltd_layer_num

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def update_seq(self, global_steps):
        if self.state[RANDOM_LTD_CURRENT_VALUE] < self.state[RANDOM_LTD_MAX_VALUE]:
            self.state[RANDOM_LTD_CURRENT_VALUE] = self.get_value(global_steps)
        if global_steps != self.state[RANDOM_LTD_CURR_STEP]:
            self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS] += self.global_batch_size * (self.state[RANDOM_LTD_CURRENT_VALUE] * self.random_ltd_layer_num + self.state[RANDOM_LTD_MAX_VALUE] * (self.model_layer_num - self.random_ltd_layer_num))
            self.state[RANDOM_LTD_CURR_STEP] = global_steps

    def state_dict(self):
        return {RANDOM_LTD_CONSUMED_LAYER_TOKENS: self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS], RANDOM_LTD_CURR_STEP: self.state[RANDOM_LTD_CURR_STEP], RANDOM_LTD_CURRENT_VALUE: self.state[RANDOM_LTD_CURRENT_VALUE], RANDOM_LTD_MIN_VALUE: self.state[RANDOM_LTD_MIN_VALUE], RANDOM_LTD_MAX_VALUE: self.state[RANDOM_LTD_MAX_VALUE]}

    def load_state_dict(self, state_dict):
        self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS] = state_dict[RANDOM_LTD_CONSUMED_LAYER_TOKENS]
        self.state[RANDOM_LTD_CURR_STEP] = state_dict[RANDOM_LTD_CURR_STEP]
        self.state[RANDOM_LTD_CURRENT_VALUE] = state_dict[RANDOM_LTD_CURRENT_VALUE]
        self.state[RANDOM_LTD_MIN_VALUE] = state_dict[RANDOM_LTD_MIN_VALUE]
        self.state[RANDOM_LTD_MAX_VALUE] = state_dict[RANDOM_LTD_MAX_VALUE]


class SparseTensor(object):
    """ Compressed Sparse Tensor """

    def __init__(self, dense_tensor=None):
        self.orig_dense_tensor = dense_tensor
        self.is_sparse = dense_tensor.is_sparse
        if dense_tensor is not None:
            if dense_tensor.is_sparse:
                dense_tensor = dense_tensor.coalesce()
                self.indices = dense_tensor.indices().flatten()
                self.values = dense_tensor.values()
            else:
                result = torch.sum(dense_tensor, dim=1)
                self.indices = result.nonzero().flatten()
                self.values = dense_tensor[self.indices]
            self.dense_size = list(dense_tensor.size())
        else:
            self.indices = None
            self.values = None
            self.dense_size = None

    def to_coo_tensor(self):
        return torch.sparse_coo_tensor(self.indices.unsqueeze(0), self.values, self.dense_size)

    @staticmethod
    def type():
        return 'deepspeed.SparseTensor'

    def to_dense(self):
        it = self.indices.unsqueeze(1)
        full_indices = torch.cat([it for _ in range(self.dense_size[1])], dim=1)
        return self.values.new_zeros(self.dense_size).scatter_add_(0, full_indices, self.values)

    def sparse_size(self):
        index_size = list(self.indices.size())
        index_size = index_size[0]
        value_size = list(self.values.size())
        value_size = value_size[0] * value_size[1]
        dense_size = self.dense_size[0] * self.dense_size[1]
        return index_size + value_size, dense_size

    def add(self, b):
        assert self.dense_size == b.dense_size
        self.indices = torch.cat([self.indices, b.indices])
        self.values = torch.cat([self.values, b.values])

    def __str__(self):
        sparse_size, dense_size = self.sparse_size()
        return 'DeepSpeed.SparseTensor(indices_size={}, values_size={}, dense_size={}, device={}, reduction_factor={})'.format(self.indices.size(), self.values.size(), self.dense_size, self.indices.get_device(), dense_size / sparse_size)

    def __repr__(self):
        return self.__str__()


TORCH_ADAM_PARAM = 'torch_adam'


class ThroughputTimer:

    def __init__(self, batch_size, start_step=2, steps_per_output=50, monitor_memory=False, logging_fn=None):
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.batch_size = 1 if batch_size is None else batch_size
        self.start_step = start_step
        self.epoch_count = 0
        self.micro_step_count = 0
        self.global_step_count = 0
        self.total_elapsed_time = 0
        self.step_elapsed_time = 0
        self.steps_per_output = steps_per_output
        self.monitor_memory = monitor_memory
        self.logging = logging_fn
        if self.logging is None:
            self.logging = logger.info
        self.initialized = False
        if self.monitor_memory and not PSUTILS_INSTALLED:
            raise ImportError("Unable to import 'psutils', please install package")

    def update_epoch_count(self):
        self.epoch_count += 1
        self.micro_step_count = 0

    def _init_timer(self):
        self.initialized = True

    def start(self):
        self._init_timer()
        self.started = True
        if self.global_step_count >= self.start_step:
            torch.cuda.synchronize()
            self.start_time = time.time()

    def stop(self, global_step=False, report_speed=True):
        if not self.started:
            return
        self.started = False
        self.micro_step_count += 1
        if global_step:
            self.global_step_count += 1
        if self.start_time > 0:
            torch.cuda.synchronize()
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.total_elapsed_time += duration
            self.step_elapsed_time += duration
            if global_step:
                if report_speed and self.global_step_count % self.steps_per_output == 0:
                    self.logging('epoch={}/micro_step={}/global_step={}, RunningAvgSamplesPerSec={}, CurrSamplesPerSec={}, MemAllocated={}GB, MaxMemAllocated={}GB'.format(self.epoch_count, self.micro_step_count, self.global_step_count, self.avg_samples_per_sec(), self.batch_size / self.step_elapsed_time, round(torch.cuda.memory_allocated() / 1024 ** 3, 2), round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)))
                    if self.monitor_memory:
                        virt_mem = psutil.virtual_memory()
                        swap = psutil.swap_memory()
                        self.logging('epoch={}/micro_step={}/global_step={}, vm %: {}, swap %: {}'.format(self.epoch_count, self.micro_step_count, self.global_step_count, virt_mem.percent, swap.percent))
                self.step_elapsed_time = 0

    def avg_samples_per_sec(self):
        if self.global_step_count > 0:
            total_step_offset = self.global_step_count - self.start_step
            avg_time_per_step = self.total_elapsed_time / total_step_offset
            return self.batch_size / avg_time_per_step
        return float('-inf')


class ZeRORuntimeException(Exception):
    pass


x = [torch.rand((512, 512)), torch.rand((512, 1024)), torch.rand((512, 30000))]


unflat_t = x * 30


flat_py = _flatten_dense_tensors(unflat_t)


flat_t = flat_py


def apex():
    for i in range(1000):
        unflat = unflatten_apex(flat_t, unflat_t)


def clip_grad_norm_(parameters, max_norm, norm_type=2, mpu=None):
    """Clips gradient norm of an iterable of parameters.

    This has been adapted from Nvidia megatron. We add norm averaging
    to consider MoE params when calculating norm as they will result
    in different norms across different ranks.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0
        for p in parameters:
            if mpu is not None:
                if mpu.get_model_parallel_rank() == 0 or is_model_parallel_parameter(p):
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
            else:
                param_norm = p.grad.data.float().norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm_cuda = torch.FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)
    pg = groups._get_data_parallel_group()
    scaled_norm = total_norm * 1.0 / float(dist.get_world_size(group=pg))
    scaled_norm_tensor = torch.FloatTensor([float(scaled_norm)])
    dist.all_reduce(scaled_norm_tensor, group=pg)
    total_norm = scaled_norm_tensor.item()
    clip_coef = max_norm / (total_norm + 1e-06)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


def is_module_compressible(module, mpu=None):
    ret = isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Embedding) or isinstance(module, torch.nn.BatchNorm2d)
    if mpu is not None:
        ret = ret or isinstance(module, mpu.RowParallelLinear) or isinstance(module, mpu.ColumnParallelLinear)
    return ret


def get_module_name(group_name, model, key_word, exist_module_name, mpu=None, verbose=True):
    """
    get the associated module name from the model based on the key_word provided by users
    """
    return_module_name = []
    for name, module in model.named_modules():
        module_check = is_module_compressible(module, mpu)
        if re.search(key_word, name) is not None and module_check:
            if name in exist_module_name and verbose:
                raise ValueError(f'{name} is already added to compression, please check your config file for {group_name}.')
            if name not in exist_module_name:
                exist_module_name.add(name)
                return_module_name.append(name)
    return return_module_name, exist_module_name


def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


class compression_scheduler:
    """
    Used to schedule different compression methods
    """

    def __init__(self, model, compression_config):
        self.model = model
        self.compression_config = compression_config
        self.make_init()
        self.training_steps = 0
        self.weight_quantization_enabled = False
        self.verbose = {WEIGHT_QUANTIZATION: False, ACTIVATION_QUANTIZATION: False, SPARSE_PRUNING: False, HEAD_PRUNING: False, ROW_PRUNING: False, CHANNEL_PRUNING: False}

    def make_init(self):
        self.different_compression_methods = {}
        for method, method_content in self.compression_config.items():
            if LAYER_REDUCTION in method:
                continue
            self.different_compression_methods[method] = {TECHNIQUE_ENABLED: False, SHARED_PARAMETERS: None, DIFFERENT_GROUPS: []}
            exist_module_name = set()
            shared_parameters = method_content[SHARED_PARAMETERS]
            self.different_compression_methods[method][TECHNIQUE_ENABLED] = shared_parameters[TECHNIQUE_ENABLED]
            self.different_compression_methods[method][SHARED_PARAMETERS] = shared_parameters
            for group_name, method_parameters in method_content[DIFFERENT_GROUPS].items():
                module_name_list = []
                for key_word in method_parameters[DIFFERENT_GROUPS_MODULE_SCOPE]:
                    module_name, exist_module_name = get_module_name(group_name, self.model, key_word, exist_module_name, verbose=False)
                    module_name_list.extend(module_name)
                if module_name_list:
                    self.different_compression_methods[method][DIFFERENT_GROUPS].append([group_name, module_name_list, method_parameters.copy().pop('params')])

    def check_weight_quantization(self):
        wq = self.different_compression_methods[WEIGHT_QUANTIZATION]
        if not wq[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = wq[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in wq[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.weight_quantization_enabled = True
                if not self.verbose[WEIGHT_QUANTIZATION]:
                    logger.info(f'Weight quantization is enabled at step {self.training_steps}')
                    self.weight_quantization_enabled = True
                    self.verbose[WEIGHT_QUANTIZATION] = True

    def check_activation_quantization(self):
        aq = self.different_compression_methods[ACTIVATION_QUANTIZATION]
        if not aq[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = aq[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in aq[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.activation_quantization_enabled = True
                if not self.verbose[ACTIVATION_QUANTIZATION]:
                    logger.info(f'Activation quantization is enabled at step {self.training_steps}')
                    self.verbose[ACTIVATION_QUANTIZATION] = True

    def check_sparse_pruning(self):
        sp = self.different_compression_methods[SPARSE_PRUNING]
        if not sp[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = sp[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in sp[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.sparse_pruning_enabled = True
                if not self.verbose[SPARSE_PRUNING]:
                    logger.info(f'Sparse pruning is enabled at step {self.training_steps}')
                    self.verbose[SPARSE_PRUNING] = True

    def check_head_pruning(self):
        hp = self.different_compression_methods[HEAD_PRUNING]
        if not hp[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = hp[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in hp[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.head_pruning_enabled = True
                if not self.verbose[HEAD_PRUNING]:
                    logger.info(f'Head pruning is enabled at step {self.training_steps}')
                    self.verbose[HEAD_PRUNING] = True

    def check_row_pruning(self):
        rp = self.different_compression_methods[ROW_PRUNING]
        if not rp[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = rp[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in rp[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.row_pruning_enabled = True
                if not self.verbose[ROW_PRUNING]:
                    logger.info(f'Row pruning is enabled at step {self.training_steps}')
                    self.verbose[ROW_PRUNING] = True

    def check_channel_pruning(self):
        cp = self.different_compression_methods[CHANNEL_PRUNING]
        if not cp[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = cp[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in cp[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.channel_pruning_enabled = True
                if not self.verbose[CHANNEL_PRUNING]:
                    logger.info(f'Channel pruning is enabled at step {self.training_steps}')
                    self.verbose[CHANNEL_PRUNING] = True

    def check_all_modules(self):
        self.check_weight_quantization()
        self.check_activation_quantization()
        self.check_sparse_pruning()
        self.check_head_pruning()
        self.check_row_pruning()
        self.check_channel_pruning()

    def step(self, step_zero_check=False):
        if not step_zero_check:
            self.training_steps += 1
        self.check_all_modules()


def debug_extract_module_and_param_names(model):
    global module_names
    global param_names
    module_names = {module: name for name, module in model.named_modules()}
    param_names = {param: name for name, param in model.named_parameters()}


def ensure_directory_exists(filename):
    """Create the directory path to ``filename`` if it does not already exist.

    Args:
        filename (str): A file path.
    """
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def err(s: str) ->None:
    None


def get_ma_status():
    if dist.is_initialized() and not dist.get_rank() == 0:
        return 0
    return torch.cuda.memory_allocated()


ZERO_SUPPORTED_OPTIMIZERS = [torch.optim.Adam, torch.optim.AdamW, FusedAdam, DeepSpeedCPUAdam]


def is_zero_supported_optimizer(optimizer):
    if dist.get_rank() == 0:
        logger.info(f'Checking ZeRO support for optimizer={optimizer.__class__.__name__} type={type(optimizer)}')
    return type(optimizer) in ZERO_SUPPORTED_OPTIMIZERS


def print_configuration(args, name):
    logger.info('{}:'.format(name))
    for arg in sorted(vars(args)):
        dots = '.' * (29 - len(arg))
        logger.info('  {} {} {}'.format(arg, dots, getattr(args, arg)))


def print_json_dist(message, ranks=None, path=None):
    """Print message when one of following condition meets

    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    Args:
        message (str)
        ranks (list)
        path (str)

    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or my_rank in set(ranks)
    if should_log:
        message['rank'] = my_rank
        with open(path, 'w') as outfile:
            json.dump(message, outfile)
            os.fsync(outfile)


def remove_random_ltd_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if '.random_ltd_layer' in key:
            new_key = ''.join(key.split('.random_ltd_layer'))
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def split_half_float_double_sparse(tensors):
    supported_types = ['torch.cuda.HalfTensor', 'torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor', 'torch.cuda.BFloat16Tensor', SparseTensor.type()]
    for t in tensors:
        assert t.type() in supported_types, f'attempting to reduce an unsupported grad type: {t.type()}'
    buckets = []
    for i, dtype in enumerate(supported_types):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets


DATA_PARALLEL_ID = -2


LOG_STAGE = -2


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


class PartitionedTensor:

    def __init__(self, tensor, group, partition_meta=None):
        super().__init__()
        self.group = group
        self.num_parts = dist.get_world_size(group=self.group)
        self.rank = dist.get_rank(group=self.group)
        self.orig_size = list(tensor.size())
        self.orig_device = tensor.device
        self.local_data, self.partition = self._partition_tensor(tensor)

    @classmethod
    def from_meta(cls, meta, local_part, group, device='cuda'):
        assert meta.dtype == torch.long
        dummy = torch.ones(dist.get_world_size(group=group))
        part_obj = cls(tensor=dummy, group=group)
        meta = meta.tolist()
        part_obj.orig_size = meta[1:1 + meta[0]]
        meta = meta[1 + meta[0]:]
        part_obj.orig_device = device
        part_obj.local_data = local_part.detach()
        part_obj.group = group
        assert part_obj.num_parts == meta[0]
        assert part_obj.rank == meta[1]
        part_obj.partition = meta[2:]
        return part_obj

    def _partition_tensor(self, tensor):
        partition = partition_uniform(num_items=tensor.numel(), num_parts=self.num_parts)
        start = partition[self.rank]
        length = partition[self.rank + 1] - start
        tensor_part = tensor.detach().contiguous().view(-1).narrow(0, start=start, length=length).clone()
        return tensor_part, partition

    def full(self, device=None):
        if device is None:
            device = self.orig_device
        full_numel = prod(self.full_size())
        flat_tensor = torch.zeros([full_numel], dtype=self.local_data.dtype, device=device)
        partition_tensors = []
        for part_id in range(self.num_parts):
            part_size = self.partition[part_id + 1] - self.partition[part_id]
            buf = flat_tensor.narrow(0, start=self.partition[part_id], length=part_size)
            if part_id == self.rank:
                buf.copy_(self.local_data)
            partition_tensors.append(buf)
        dist.all_gather(partition_tensors, partition_tensors[self.rank], group=self.group)
        for i in range(len(partition_tensors)):
            partition_tensors[i].data = torch.zeros(1)
            partition_tensors[i] = None
        return flat_tensor.view(self.full_size()).clone().detach()

    def to_meta(self):
        """Returns a torch.LongTensor that encodes partitioning information.

        Can be used along with ``data()`` to serialize a ``PartitionedTensor`` for
        communication.

        Returns:
            torch.LongTensor: a tensor encoding the meta-information for the partitioning
        """
        meta = []
        meta.append(len(self.orig_size))
        meta += list(self.orig_size)
        meta.append(self.num_parts)
        meta.append(self.rank)
        meta += self.partition
        return torch.LongTensor(data=meta)

    def data(self):
        return self.local_data

    def local_size(self):
        return self.local_data.size()

    def full_size(self):
        return self.orig_size


class PipelineError(Exception):
    """Errors related to the use of deepspeed.PipelineModule """


class RepeatingLoader:

    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def is_even(number):
    return number % 2 == 0


class LinearModuleForZeroStage3(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    The weights are pre-transposed and stored as A^T instead of transposing during each
    forward. Memory savings proportional to the parameter size.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \\text{in\\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \\text{out\\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\\text{out\\_features}, \\text{in\\_features})`. The values are
            initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
            :math:`k = \\frac{1}{\\text{in\\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\\text{out\\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
                :math:`k = \\frac{1}{\\text{in\\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool=True) ->None:
        super(LinearModuleForZeroStage3, self).__init__()
        None
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) ->Tensor:
        return LinearFunctionForZeroStage3.apply(input, self.weight, self.bias)

    def extra_repr(self) ->str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


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
        self.in_parts = partition(num_items=in_features, num_parts=in_splits)
        self.out_parts = partition(num_items=out_features, num_parts=out_splits)
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

    def forward(self, input_):
        if self.in_splits > 1 and not self.input_is_already_split:
            input_parts = partition(input_.shape[-1], self.in_splits)
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
                with deepspeed.zero.GatheredParameters(local.weight, modifier_rank=0):
                    local.weight.copy_(global_weight)
            if local.bias is not None:
                with deepspeed.zero.GatheredParameters(local.bias, modifier_rank=0):
                    local.bias.data.copy_(other.bias[rstart:rstop].data)


class TiledLinearReturnBias(TiledLinear):
    """Wrapper for a Linear class that returns its own bias parameter, such as
    used by Megatron-LM.
    """

    def _reduce_local_output(self, in_id, out_id, current_out, new_out):
        """Reduces output tensors, but not the returned bias. """
        if current_out is not None:
            old_tensor, old_bias = current_out
        else:
            old_tensor, old_bias = None, None
        assert isinstance(new_out, tuple)
        assert len(new_out) == 2
        tensor, bias = new_out
        assert tensor is not None
        tensor = super()._reduce_local_output(in_id=in_id, out_id=out_id, current_out=old_tensor, new_out=tensor)
        if bias is None:
            bias = old_bias
        return tensor, bias

    def _combine_output_splits(self, outputs):
        tensors = [o[0] for o in outputs]
        tensor = super()._combine_output_splits(tensors)
        biases = [o[1] for o in outputs if o[1] is not None]
        if len(biases) > 0:
            bias = super()._combine_output_splits(biases)
        else:
            bias = None
        return tensor, bias


class VerboseLinear(torch.nn.Linear):

    def __init__(self, **kwargs):
        None
        super().__init__(**kwargs)
        None


class LinearStack(torch.nn.Module):

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_layer = torch.nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False) for x in range(num_layers)])
        self.output_layer = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False, nlayers=1):
        super(SimpleModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for i in range(nlayers)])
        if empty_grad:
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.empty_grad = empty_grad

    def forward(self, x, y):
        if len(self.linears) == 1:
            x = self.linears[0](x)
        else:
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
        return self.cross_entropy_loss(x, y)


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.loss_fn(x, y)


class AlexNetPipe(AlexNet):

    def to_layers(self):
        layers = [*self.features, lambda x: x.view(x.size(0), -1), self.classifier]
        return layers


class AlexNetPipeSpec(PipelineModule):

    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        specs = [LayerSpec(nn.Conv2d, 3, 64, kernel_size=11, stride=4, padding=5), LayerSpec(nn.ReLU, inplace=True), LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2), LayerSpec(nn.Conv2d, 64, 192, kernel_size=5, padding=2), F.relu, LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2), LayerSpec(nn.Conv2d, 192, 384, kernel_size=3, padding=1), F.relu, LayerSpec(nn.Conv2d, 384, 256, kernel_size=3, padding=1), F.relu, LayerSpec(nn.Conv2d, 256, 256, kernel_size=3, padding=1), F.relu, LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2), lambda x: x.view(x.size(0), -1), LayerSpec(nn.Linear, 256, self.num_classes)]
        super().__init__(layers=specs, loss_fn=nn.CrossEntropyLoss(), **kwargs)


class Conv1D(torch.nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        self.weight = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def get_test_path(filename):
    curr_path = Path(__file__).parent
    return str(curr_path.joinpath(filename))


class MockGPT2ModelPipe(PipelineModule):

    def __init__(self, num_layers, mp_size, args_others, topo, **kwargs):
        args_defaults = {'vocab_file': get_test_path('gpt2-vocab.json'), 'merge_file': get_test_path('gpt2-merges.txt'), 'tokenizer_type': 'GPT2BPETokenizer'}
        args_defaults.update(args_others)
        sys.argv.extend(['--model-parallel-size', str(mp_size), '--make-vocab-size-divisible-by', str(1)])
        initialize_megatron(args_defaults=args_defaults, ignore_unknown_args=True)


        class ParallelTransformerLayerPipe(ParallelTransformerLayer):

            def forward(self, args):
                attention_mask = torch.tensor([[True]], device=torch.cuda.current_device())
                return super().forward(args, attention_mask)
        layers = []
        for x in range(num_layers):
            layers.append(LayerSpec(ParallelTransformerLayerPipe, self.gpt2_attention_mask_func, self.init_method_normal(0.02), self.scaled_init_method_normal(0.02, num_layers), x))
        super().__init__(layers=layers, loss_fn=torch.nn.CrossEntropyLoss(), topology=topo, **kwargs)

    def gpt2_attention_mask_func(self, attention_scores, ltor_mask):
        attention_scores.masked_fill_(ltor_mask, -10000.0)
        return attention_scores

    def init_method_normal(self, sigma):
        """Init method based on N(0, sigma)."""

        def init_(tensor):
            return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
        return init_

    def scaled_init_method_normal(self, sigma, num_layers):
        """Init method based on N(0, sigma/sqrt(2*num_layers)."""
        std = sigma / math.sqrt(2.0 * num_layers)

        def init_(tensor):
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)
        return init_


def f_gelu(x):
    x_type = x.dtype
    x = x.float()
    x = x * 0.5 * (1.0 + torch.erf(x / 1.41421))
    return x


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return f_gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


def bias_gelu(bias, y):
    y_type = y.dtype
    x = bias.float() + y.float()
    x = x * 0.5 * (1.0 + torch.erf(x / 1.41421))
    return x


def bias_tanh(bias, y):
    y_type = y.dtype
    x = bias.float() + y.float()
    x = torch.tanh(x)
    return x


class LinearActivation(Module):
    """Fused Linear and activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, weights, biases, act='gelu', bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused_gelu = False
        self.fused_tanh = False
        if isinstance(act, str):
            if bias and act == 'gelu':
                self.fused_gelu = True
            elif bias and act == 'tanh':
                self.fused_tanh = True
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = weights[5]
        self.bias = biases[5]

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.fused_gelu:
            y = F.linear(input, self.weight, None)
            bg = bias_gelu(self.bias, y)
            return bg
        elif self.fused_tanh:
            return bias_tanh(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, i, config, weights, biases):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.query.weight = weights[0]
        self.query.bias = biases[0]
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.key.weight = weights[1]
        self.key.bias = biases[1]
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.value.weight = weights[2]
        self.value.bias = biases[2]
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask, grads=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer1 = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape)
        if grads is not None:
            query_layer.register_hook(lambda x, self=self: grads.append([x, 'Query']))
            key_layer.register_hook(lambda x, self=self: grads.append([x, 'Key']))
            value_layer.register_hook(lambda x, self=self: grads.append([x, 'Value']))
        return context_layer1


class BertSelfOutput(nn.Module):

    def __init__(self, config, weights, biases):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense.weight = weights[3]
        self.dense.bias = biases[3]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    def get_w(self):
        return self.dense.weight


class BertAttention(nn.Module):

    def __init__(self, i, config, weights, biases):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(i, config, weights, biases)
        self.output = BertSelfOutput(config, weights, biases)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

    def get_w(self):
        return self.output.get_w()


class BertIntermediate(nn.Module):

    def __init__(self, config, weights, biases):
        super(BertIntermediate, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size, config.intermediate_size, weights, biases, act=config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config, weights, biases):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense.weight = weights[6]
        self.dense.bias = biases[6]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, i, config, weights, biases):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(i, config, weights, biases)
        self.PreAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.PostAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.intermediate = BertIntermediate(config, weights, biases)
        self.output = BertOutput(config, weights, biases)
        self.weight = weights
        self.biases = biases

    def forward(self, hidden_states, attention_mask, grads, collect_all_grads=False):
        input_layer_norm = self.PreAttentionLayerNorm(hidden_states)
        attention_output = self.attention(input_layer_norm, attention_mask)
        intermediate_input = hidden_states + attention_output
        intermediate_layer_norm = self.PostAttentionLayerNorm(intermediate_input)
        intermediate_output = self.intermediate(intermediate_layer_norm)
        layer_output = self.output(intermediate_output, attention_output)
        if collect_all_grads:
            self.weight[2].register_hook(lambda x, self=self: grads.append([x, 'V_W']))
            self.biases[2].register_hook(lambda x, self=self: grads.append([x, 'V_B']))
            self.weight[3].register_hook(lambda x, self=self: grads.append([x, 'O_W']))
            self.biases[3].register_hook(lambda x, self=self: grads.append([x, 'O_B']))
            self.PostAttentionLayerNorm.weight.register_hook(lambda x, self=self: grads.append([x, 'N2_W']))
            self.PostAttentionLayerNorm.bias.register_hook(lambda x, self=self: grads.append([x, 'N2_B']))
            self.weight[5].register_hook(lambda x, self=self: grads.append([x, 'int_W']))
            self.biases[5].register_hook(lambda x, self=self: grads.append([x, 'int_B']))
            self.weight[6].register_hook(lambda x, self=self: grads.append([x, 'out_W']))
            self.biases[6].register_hook(lambda x, self=self: grads.append([x, 'out_B']))
            self.PreAttentionLayerNorm.weight.register_hook(lambda x, self=self: grads.append([x, 'norm_W']))
            self.PreAttentionLayerNorm.bias.register_hook(lambda x, self=self: grads.append([x, 'norm_B']))
        return layer_output + intermediate_input

    def get_w(self):
        return self.attention.get_w()


class BertEncoder(nn.Module):

    def __init__(self, config, weights, biases):
        super(BertEncoder, self).__init__()
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.layer = nn.ModuleList([copy.deepcopy(BertLayer(i, config, weights, biases)) for i in range(config.num_hidden_layers)])
        self.grads = []
        self.graph = []

    def get_grads(self):
        return self.grads

    def get_modules(self, big_node, input):
        for mdl in big_node.named_children():
            self.graph.append(mdl)
            self.get_modules(self, mdl, input)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, checkpoint_activations=False):
        all_encoder_layers = []

        def custom(start, end):

            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward
        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(custom(l, l + chunk_length), hidden_states, attention_mask * 1)
                l += chunk_length
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask, self.grads, collect_all_grads=True)
                hidden_states.register_hook(lambda x, i=i, self=self: self.grads.append([x, 'hidden_state']))
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers or checkpoint_activations:
            hidden_states = self.FinalLayerNorm(hidden_states)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size, config.hidden_size, act='tanh')

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size, config.hidden_size, act=config.hidden_act)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        torch.cuda.nvtx.range_push('decoder input.size() = {}, weight.size() = {}'.format(hidden_states.size(), self.decoder.weight.size()))
        hidden_states = self.decoder(hidden_states) + self.bias
        torch.cuda.nvtx.range_pop()
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size_or_config_json_file, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, batch_size=8, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, fp16=False):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.batch_size = batch_size
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.fp16 = fp16
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


CONFIG_NAME = 'bert_config.json'


PRETRAINED_MODEL_ARCHIVE_MAP = {'bert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz', 'bert-large-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz', 'bert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz', 'bert-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz', 'bert-base-multilingual-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz', 'bert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz', 'bert-base-chinese': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz'}


TF_WEIGHTS_NAME = 'model.ckpt'


WEIGHTS_NAME = 'pytorch_model.bin'


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        None
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    None
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        None
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m'] for n in name):
            None
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                l = re.split('_(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
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
        None
        pointer.data = torch.from_numpy(array)
    return model


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `BertConfig`. To create a model from a Google pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None, from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        if resolved_archive_file == archive_file:
            logger.info('loading archive file {}'.format(archive_file))
        else:
            logger.info('loading archive file {} from cache at {}'.format(archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info('extracting archive file {} to temp dir {}'.format(resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info('Model config {}'.format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            shutil.rmtree(tempdir)
        if from_tf:
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
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
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controlled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, checkpoint_activations=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers, checkpoint_activations=checkpoint_activations)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args):
        super(BertForPreTraining, self).__init__(config)
        self.summary_writer = None
        if dist.get_rank() == 0:
            self.summary_writer = args.summary_writer
        self.samples_per_step = dist.get_world_size() * args.train_batch_size
        self.sample_count = self.samples_per_step
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def log_summary_writer(self, logs: dict, base='Train'):
        if dist.get_rank() == 0:
            module_name = 'Samples'
            for key, log in logs.items():
                self.summary_writer.add_scalar(f'{base}/{module_name}/{key}', log, self.sample_count)
            self.sample_count += self.samples_per_step

    def forward(self, batch, log=True):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[5]
        next_sentence_label = batch[4]
        checkpoint_activations = False
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, checkpoint_activations=False):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None, checkpoint_activations=False):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, checkpoint_activations=False):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForMultipleChoice(BertPreTrainedModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_choices):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, checkpoint_activations=False):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, checkpoint_activations=False):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None, checkpoint_activations=False):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class MultiOutputModel(torch.nn.Module):

    def __init__(self, hidden_dim, weight_value):
        super(MultiOutputModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear.weight.data.fill_(weight_value)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        losses = []
        for x, y in zip(inputs, targets):
            hidden_dim = self.linear(x)
            loss = self.cross_entropy_loss(hidden_dim, y)
            losses.append(loss)
        return tuple(losses)


class DSEncoder(nn.Module):

    def __init__(self, config, weights, biases):
        super(DSEncoder, self).__init__()
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.layer = nn.ModuleList([copy.deepcopy(DeepSpeedTransformerLayer(config, weights, biases)) for _ in range(config.num_hidden_layers)])
        self.grads = []
        self.pre_or_post = config.pre_layer_norm

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, checkpoint_activations=False):
        all_encoder_layers = []

        def custom(start, end):

            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward
        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(custom(l, l + chunk_length), hidden_states, attention_mask * 1)
                l += chunk_length
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers or checkpoint_activations:
            if self.pre_or_post:
                hidden_states = self.FinalLayerNorm(hidden_states)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class LeNet5(torch.nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1), torch.nn.Tanh(), torch.nn.AvgPool2d(kernel_size=2), torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), torch.nn.Tanh(), torch.nn.AvgPool2d(kernel_size=2), torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1), torch.nn.Tanh())
        self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=120, out_features=84), torch.nn.Tanh(), torch.nn.Linear(in_features=84, out_features=n_classes))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return logits, probs


class MaskedLinear(torch.nn.Linear):

    def forward(self, x, mask):
        out = super().forward(x)
        if mask.is_floating_point():
            out = out * mask
        else:
            out = out * mask.type_as(out)
        return out


class MaskedLinearSeq(MaskedLinear):
    """Tests pipeline modules by also returning the mask."""

    def forward(self, x, mask):
        return super().forward(x, mask), mask


class MaskedLinearSeqDup(MaskedLinearSeq):
    """MaskedLinearSeq, but with more outputs than inputs and in a different order."""

    def forward(self, x, mask):
        dup = x.clone().detach() * 1.38
        x, mask = super().forward(x, mask)
        return dup, x, mask


class DropMaskLinear(torch.nn.Linear):

    def forward(self, x, mask):
        return super().forward(x)


class LinearNonTensorInput(torch.nn.Linear):

    def forward(self, x, non_tensor_input):
        return super().forward(x)


HIDDEN_DIM = 20


class LinearNonTensorOutput(torch.nn.Linear):

    def __init__(self, non_tensor_output):
        super().__init__(HIDDEN_DIM, HIDDEN_DIM)
        self.non_tensor_output = non_tensor_output

    def forward(self, x):
        out = super().forward(x)
        return out, self.non_tensor_output


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(10, 3, mode='sum', sparse=True)
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x, offsets):
        return self.linear(self.emb(x, offsets))


def _assert_fully_available(model: Module) ->None:
    for _, param in model.named_parameters():
        assert param.ds_status == ZeroParamStatus.AVAILABLE


class EltwiseMultiplicationModule(Module):

    def __init__(self, weight: Parameter) ->None:
        super().__init__()
        self.weight = weight

    def forward(self, x: Tensor) ->Tensor:
        _assert_fully_available(self)
        result = self.weight * x
        return result


def _assert_partition_status(model: Module, valid_statuses: Set[ZeroParamStatus]) ->None:
    for _, param in model.named_parameters():
        assert param.ds_status in valid_statuses, param.ds_summary()


class EltwiseMultiplicationTestNetwork(Module):
    """used for testing purposes"""

    def __init__(self, weight1: Parameter, weight2: Parameter, weight3: Parameter) ->None:
        super().__init__()
        self.__layer1 = EltwiseMultiplicationModule(weight1)
        self.__layer2 = EltwiseMultiplicationModule(weight2)
        self.__layer3 = EltwiseMultiplicationModule(weight3)
        self.loss = L1Loss(reduction='none')

    def forward(self, x: Tensor, y: Tensor, use_module_trace: bool, param_prefetching: bool) ->Dict[str, Tensor]:
        _assert_partition_status(self, {ZeroParamStatus.NOT_AVAILABLE, ZeroParamStatus.INFLIGHT, ZeroParamStatus.AVAILABLE} if use_module_trace else {ZeroParamStatus.NOT_AVAILABLE})
        pre_layer_expected_states = {ZeroParamStatus.INFLIGHT if param_prefetching else ZeroParamStatus.NOT_AVAILABLE, ZeroParamStatus.AVAILABLE}
        post_layer_expected_states = {ZeroParamStatus.AVAILABLE if param_prefetching else ZeroParamStatus.NOT_AVAILABLE}
        _assert_partition_status(self.__layer1, pre_layer_expected_states)
        hidden1 = self.__layer1(x)
        _assert_partition_status(self.__layer1, post_layer_expected_states)
        _assert_partition_status(self.__layer2, pre_layer_expected_states)
        hidden2 = self.__layer2(hidden1)
        _assert_partition_status(self.__layer2, post_layer_expected_states)
        _assert_partition_status(self.__layer3, pre_layer_expected_states)
        y_hat = self.__layer3(hidden2)
        _assert_partition_status(self.__layer3, post_layer_expected_states)
        loss = self.loss(y_hat, y)
        _assert_partition_status(self, {ZeroParamStatus.NOT_AVAILABLE, ZeroParamStatus.INFLIGHT, ZeroParamStatus.AVAILABLE} if use_module_trace else {ZeroParamStatus.NOT_AVAILABLE})
        return {'hidden1': hidden1, 'hidden2': hidden2, 'y_hat': y_hat, 'loss': loss}


class DanglingBias(torch.nn.Linear):

    def forward(self, *inputs):
        out = super().forward(*inputs)
        return out, self.bias


class DataClass:
    """Just wraps data in an object. """

    def __init__(self, out=None, bias=None):
        self.out = out
        self.bias = bias


class DanglingBiasClass(DanglingBias):

    def forward(self, *inputs):
        out, bias = super().forward(*inputs)
        return DataClass(out=out, bias=bias)


class DanglingAttention(torch.nn.Linear):

    def __init__(self, dim=16, return_obj=False):
        super().__init__(dim, dim)
        self.dim = dim
        self.return_obj = return_obj
        if return_obj:
            self.d_linear = DanglingBiasClass(dim, dim)
        else:
            self.d_linear = DanglingBias(dim, dim)

    def forward(self, input):
        out = super().forward(input)
        if self.return_obj:
            out_obj = self.d_linear(out)
            assert out_obj.bias.ds_status == ZeroParamStatus.AVAILABLE
            return out_obj.out, out_obj.bias
        else:
            out, bias = self.d_linear(out)
            assert hasattr(bias, 'ds_status') or hasattr(bias, 'ds_param_alias')
            z3_bias = bias if hasattr(bias, 'ds_status') else bias.ds_param_alias
            assert z3_bias.ds_status == ZeroParamStatus.AVAILABLE
            return out, bias


class ModelContainer(torch.nn.Module):

    def __init__(self, dim=16, return_obj=False):
        super().__init__()
        self.dim = dim
        self.linear1 = torch.nn.Linear(dim, dim)
        self.dangler = DanglingAttention(dim, return_obj=return_obj)

    def forward(self, input):
        act1 = self.linear1(input)
        act2, bias = self.dangler(act1)
        return (act2 + bias).sum()


class DanglingExt(torch.nn.Module):

    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        self.container = ModelContainer(dim)

    def forward(self, input):
        out = self.container(input)
        assert len(self._external_params) == 0
        assert len(self.container._external_params) == 1
        assert len(self.container.dangler._external_params) == 0
        assert id(self.container.dangler.d_linear.bias) in self.container._external_params.keys()
        return out


class ModelContainerVariableOutputType(ModelContainer):

    def __init__(self, dim=16, output_type=dict):
        super().__init__()
        self.output_type = output_type
        self.dim = dim
        self.linear1 = torch.nn.Linear(dim, dim)

    def forward(self, input):
        act1 = self.linear1(input)
        if self.output_type is dict:
            return {'loss': act1.sum()}
        if self.output_type is torch.tensor:
            return act1.sum()


class ConvX(torch.nn.Conv1d):

    def __init__(self, *args):
        super().__init__(*args)
        self.param_in = torch.nn.Parameter(torch.FloatTensor(5).uniform_())

    def forward(self, x):
        return x


class ConvNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = ConvX(1, 3, 4)
        self.param = torch.nn.Parameter(torch.FloatTensor(5).uniform_())

    def forward(self, x):
        return x


class GrandPa(torch.nn.Module):

    def __init__(self, *args):
        super().__init__(*args)
        self.param_grandpa = torch.nn.Parameter(torch.ones(5))
        self.param_grandpa.data = (self.param_grandpa.data + 1).data


class Pa(GrandPa):

    def __init__(self, *args):
        super().__init__(*args)
        self.param_pa = torch.nn.Parameter(torch.ones(5))
        self.param_pa.data = (self.param_pa.data + 1).data
        self.param_grandpa.data = (self.param_grandpa.data + 1).data


class Son(Pa):

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(5))
        self.param.data = (self.param.data + 1).data
        self.param_pa.data = (self.param_pa.data + 1).data
        self.param_grandpa.data = (self.param_grandpa.data + 1).data


class LinearWrapper(torch.nn.Linear):
    """Returns its own bias to simulate Megatron-LM's behavior.

    Megatron-LM optionally delays the bias addition to fuse with a proceeding kernel.
    """

    def forward(self, input):
        out = super().forward(input)
        return out, self.bias


class Curriculum_SimpleModel(SimpleModel):

    def __init__(self, hidden_dim, empty_grad=False):
        super(Curriculum_SimpleModel, self).__init__(hidden_dim, empty_grad)

    def forward(self, x, y, **kwargs):
        seqlen = kwargs.get('curriculum_seqlen', None)
        loss = super(Curriculum_SimpleModel, self).forward(x, y)
        return loss, seqlen


class SimpleMoEModel(torch.nn.Module):

    def __init__(self, hidden_dim, num_experts=4, ep_size=1, use_residual=False):
        super(SimpleMoEModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        expert = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = MoE(hidden_size=hidden_dim, expert=expert, ep_size=ep_size, use_residual=use_residual, num_experts=num_experts, k=1)
        self.linear3 = MoE(hidden_size=hidden_dim, expert=expert, ep_size=ep_size, use_residual=use_residual, num_experts=num_experts, k=1)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = self.linear(x)
        output, _, _ = self.linear2(hidden_dim)
        output, _, _ = self.linear3(output)
        hidden_dim = hidden_dim + output
        sentence_embed = hidden_dim.mean(1)
        return self.cross_entropy_loss(sentence_embed, y)


class SimplePRMoEModel(torch.nn.Module):

    def __init__(self, hidden_dim, num_experts=2, ep_size=1, use_residual=False):
        super(SimplePRMoEModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = MoE(hidden_size=hidden_dim, expert=linear2, ep_size=ep_size, use_residual=use_residual, num_experts=num_experts, k=1)
        linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = MoE(hidden_size=hidden_dim, expert=linear3, ep_size=ep_size, use_residual=use_residual, num_experts=int(2 * num_experts), k=1)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = x
        hidden_dim = self.linear(hidden_dim)
        output, _, _ = self.linear2(hidden_dim)
        output, _, _ = self.linear3(output)
        hidden_dim = hidden_dim + output
        sentence_embed = hidden_dim.mean(1)
        return self.cross_entropy_loss(sentence_embed, y)


class UnusedParametersModel(SimpleModel):

    def __init__(self, hidden_dim, empty_grad=False):
        super().__init__(hidden_dim, empty_grad)
        self.unused_linear = torch.nn.Linear(hidden_dim, hidden_dim)


class LinearStackPipe(PipelineModule):

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, num_layers=4, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        layers = []
        layers.append(LayerSpec(torch.nn.Linear, self.input_dim, self.hidden_dim))
        for x in range(self.num_layers):
            layers.append(LayerSpec(torch.nn.Linear, self.hidden_dim, self.hidden_dim, bias=False))
            layers.append(lambda x: x)
        layers.append(LayerSpec(torch.nn.Linear, self.hidden_dim, self.output_dim))
        super().__init__(layers=layers, loss_fn=torch.nn.CrossEntropyLoss(), **kwargs)


class PLD_SimpleModel(SimpleModel):

    def __init__(self, hidden_dim, empty_grad=False):
        super(PLD_SimpleModel, self).__init__(hidden_dim, empty_grad)

    def forward(self, x, y, **kwargs):
        pld = kwargs.get('progressive_layer_drop', False)
        theta = kwargs.get('pld_theta', 1.0)
        hidden_dim = super(PLD_SimpleModel, self).forward(x, y)
        return hidden_dim

