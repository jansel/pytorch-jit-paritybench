import sys
_module = sys.modules[__name__]
del sys
conf = _module
bert = _module
imagenet_autotune = _module
resnet50 = _module
intel_extension_for_pytorch = _module
cpu = _module
_auto_kernel_selection = _module
auto_ipex = _module
autocast = _module
_autocast_mode = _module
_grad_scaler = _module
hypertune = _module
config = _module
dotdict = _module
resnet50 = _module
objective = _module
strategy = _module
grid = _module
random = _module
launch = _module
runtime = _module
cpupool = _module
multi_stream = _module
runtime_utils = _module
task = _module
frontend = _module
jit = _module
_trace = _module
nn = _module
functional = _module
_embeddingbag = _module
_roi_align = _module
_tensor_method = _module
interaction = _module
modules = _module
_roi_align = _module
frozen_batch_norm = _module
linear_fuse_eltwise = _module
merged_embeddingbag = _module
utils = _module
_model_convert = _module
_weight_cast = _module
_weight_prepack = _module
optim = _module
_functional = _module
_lamb = _module
_optimizer_utils = _module
quantization = _module
_autotune = _module
_module_swap_utils = _module
_qconfig = _module
_quantization_state = _module
_quantization_state_utils = _module
_quantize = _module
_quantize_utils = _module
_recipe = _module
_utils = _module
_cpu_isa = _module
_custom_fx_tracer = _module
channels_last_1d = _module
linear_bn_folding = _module
verbose = _module
xpu = _module
_utils = _module
amp = _module
autocast_mode = _module
cpp_extension = _module
intrinsic = _module
intrinsic = _module
lazy_init = _module
memory = _module
random = _module
single_card = _module
streams = _module
utils = _module
setup = _module
autocast_test_lists = _module
interaction = _module
merged_embeddingbag = _module
optimizer = _module
code_free_optimization = _module
common_device_type = _module
common_ipex_conf = _module
common_methods_invocations = _module
common_nn = _module
common_utils = _module
network1 = _module
network2 = _module
expecttest = _module
fpmath_mode = _module
itensor_size1_test = _module
linear_prepack = _module
linear_reorder = _module
override = _module
profile_ipex_op = _module
runtime = _module
test_ao_jit_ipex_quantization = _module
test_ao_jit_llga_quantization_fuser = _module
test_ao_jit_llga_throughput_benchmark = _module
test_ao_jit_llga_utils = _module
test_auto_channels_last = _module
test_autocast = _module
test_check = _module
test_code_free_optimization = _module
test_conv_reorder = _module
test_cpu_ops = _module
test_cumsum = _module
test_dropout = _module
test_dyndisp = _module
test_emb = _module
test_fpmath_mode = _module
test_frozen_batch_norm = _module
test_graph_capture = _module
test_interaction = _module
test_ipex_custom_op = _module
test_ipex_optimize = _module
test_jit = _module
test_jit_llga_fuser = _module
test_launcher = _module
test_layer_norm = _module
test_linear_fuse_eltwise = _module
test_linear_reorder = _module
test_lru_cache = _module
test_merged_embeddingbag = _module
test_mha = _module
test_nms = _module
test_optimizer = _module
test_profile = _module
test_quantization_default_recipe = _module
test_rnnt_custom_kernel = _module
test_roialign = _module
test_runtime_api = _module
test_runtime_api_jit = _module
test_softmax = _module
test_tensor_method = _module
test_tensorexpr = _module
test_toolkit = _module
test_transfree_bmm = _module
test_verbose = _module
test_weight_cast = _module
test_weight_prepack = _module
utils = _module
verbose = _module
linter = _module
clang_format_all = _module
clang_format_utils = _module
clang_tidy = _module
generate_build_files = _module
max_tokens_pragma = _module
run = _module
flake8_hook = _module
install = _module
download_bin = _module
mypy_wrapper = _module
trailing_newlines = _module
translate_annotations = _module

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


import torchvision.models as models


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import re


import logging


import uuid


import warnings


from typing import Any


from typing import Optional


from torch.types import _dtype


from collections import defaultdict


from collections import abc


from enum import Enum


from typing import Dict


from typing import List


from typing import Tuple


import functools


import numpy as np


import torch.nn as nn


from typing import Union


import copy


import torch._dynamo


import torch.fx.experimental.optimization as optimization


from torch.jit._trace import TracerWarning


from enum import IntEnum


from functools import wraps


from torch import nn


from torch import Tensor


from torch.nn.modules.utils import _pair


from torch.jit.annotations import BroadcastingList2


from torch.autograd import Function


import enum


from typing import NamedTuple


from itertools import accumulate


from torch.nn.utils.rnn import PackedSequence


import types


import time


from typing import Callable


from torch.ao.quantization import swap_module


import torch.nn.quantized.dynamic as nnqd


from torch.quantization.qconfig import QConfig


from torch.ao.quantization import PlaceholderObserver


from torch.ao.quantization import PerChannelMinMaxObserver


from torch.ao.quantization import HistogramObserver


from torch.ao.quantization import QConfig


import torch.nn.functional as F


from torch.fx.node import map_aggregate


from collections import OrderedDict


import inspect


import numbers


from torch import _VF


import torch.fx as fx


from torch.nn.utils.fusion import fuse_linear_bn_eval


from torch import device as _device


from torch._utils import _get_device_index


from torch.storage import _StorageBase


from torch import serialization


from torch.utils.cpp_extension import _TORCH_PATH


from torch.utils.file_baton import FileBaton


from torch.utils._cpp_extension_versioner import ExtensionVersioner


from torch.utils.hipify.hipify_python import GeneratedFileCleaner


from torch.torch_version import TorchVersion


import collections


from torch.types import Device


from typing import cast


from typing import Iterable


from typing import Generator


import torch.distributed as dist


from torch.utils import ThroughputBenchmark


import math


from torch._six import inf


from functools import reduce


from torch.autograd import Variable


from copy import deepcopy


from itertools import product


from math import pi


import torch.cuda


from torch.nn.functional import _Reduction


from torch.autograd.gradcheck import get_numerical_jacobian


import torch.backends.cudnn


import random


from numbers import Number


from torch._utils_internal import get_writable_path


from torch._six import string_classes


import torch.backends.mkl


from torch.autograd import gradcheck


from torch.autograd.gradcheck import gradgradcheck


import itertools


from torch.testing import FileCheck


from torch.testing._internal.common_utils import TEST_SCIPY


from torch.testing._internal.common_utils import TemporaryFileName


from torch.testing._internal.jit_utils import freeze_rng_state


from torch.ao.quantization import MinMaxObserver


from torch.quantization.quantize_fx import prepare_fx


from torch.quantization.quantize_fx import convert_fx


from torch.ao.quantization.quantize_fx import convert_to_reference_fx


from torch.ao.quantization.quantize_fx import prepare_qat_fx


from torch.testing import assert_allclose


from torch.testing._internal.common_utils import run_tests


from torch.testing._internal.common_utils import TestCase


from torch.testing._internal.jit_utils import JitTestCase


from torch.testing._internal.jit_utils import warmup_backward


from torch.testing._internal.jit_utils import get_execution_plan


from torch.testing._internal.common_utils import freeze_rng_state


from torch.testing._internal.common_utils import get_function_arglist


from torch.testing._internal.common_utils import load_tests


from torch.jit._recursive import wrap_cpp_module


from torch.testing._internal.common_utils import TestCase as TorchTestCase


import torch.autograd.functional as autogradF


from torch.fx import GraphModule


from torch.optim import Adadelta


from torch.optim import Adagrad


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim import Adamax


from torch.optim import ASGD


from torch.optim import RMSprop


from torch.optim import Rprop


from torch.optim import SGD


from torch.nn.modules.linear import Linear


import sklearn.metrics


def get_num_cores_per_node():
    return int(subprocess.check_output("LANG=C; lscpu | grep Core'.* per socket' | awk '{print $4}'", shell=True))


def get_num_nodes():
    return int(subprocess.check_output("LANG=C; lscpu | grep Socket | awk '{print $2}'", shell=True))


def get_core_list_of_node_id(node_id):
    """
    Helper function to get the CPU cores' ids of the input numa node.

    Args:
        node_id (int): Input numa node id.

    Returns:
        list: List of CPU cores' ids on this numa node.
    """
    num_of_nodes = get_num_nodes()
    assert node_id < num_of_nodes, 'input node_id:{0} must less than system number of nodes:{1}'.format(node_id, num_of_nodes)
    num_cores_per_node = get_num_cores_per_node()
    return list(range(num_cores_per_node * node_id, num_cores_per_node * (node_id + 1)))


class CPUPool(object):
    """
    An abstraction of a pool of CPU cores used for intra-op parallelism.

    Args:
        core_ids (list): A list of CPU cores' ids used for intra-op parallelism.
        node_id (int): A numa node id with all CPU cores on the numa node.
            ``node_id`` doesn't work if ``core_ids`` is set.

    Returns:
        intel_extension_for_pytorch.cpu.runtime.CPUPool: Generated
        intel_extension_for_pytorch.cpu.runtime.CPUPool object.
    """

    def __init__(self, core_ids: list=None, node_id: int=None):
        if core_ids is not None:
            if node_id is not None:
                warnings.warn('Both of core_ids and node_id are inputed. core_ids will be used with priority.')
            if type(core_ids) is range:
                core_ids = list(core_ids)
            assert type(core_ids) is list, 'Input of core_ids must be the type of list[Int]'
            self.core_ids = core_ids
        elif node_id is not None:
            self.core_ids = get_core_list_of_node_id(node_id)
        else:
            self.core_ids = ipex._C.get_process_available_cores()
        self.cpu_pool = ipex._C.CPUPool(self.core_ids)
        self.core_ids = self.cpu_pool.get_core_list()


class MultiStreamModuleHint(object):
    """
    MultiStreamModuleHint is a hint to MultiStreamModule about how to split the inputs
    or concat the output. Each argument should be None, with type of int or a container
    which containes int or None such as: (0, None, ...) or [0, None, ...]. If the argument
    is None, it means this argument will not be split or concat. If the argument is with
    type int, its value means along which dim this argument will be split or concat.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        intel_extension_for_pytorch.cpu.runtime.MultiStreamModuleHint: Generated
        intel_extension_for_pytorch.cpu.runtime.MultiStreamModuleHint object.

    :meta public:
    """

    def __init__(self, *args, **kwargs):
        self.args = list(args)
        self.kwargs = kwargs
        self.args_len = args.__len__()
        self.kwargs_len = kwargs.__len__()


class Task(object):
    """
    An abstraction of computation based on PyTorch module and is scheduled
    asynchronously.

    Args:
        model (torch.jit.ScriptModule or torch.nn.Module): The input module.
        cpu_pool (intel_extension_for_pytorch.cpu.runtime.CPUPool): An
            intel_extension_for_pytorch.cpu.runtime.CPUPool object, contains
            all CPU cores used to run Task asynchronously.

    Returns:
        intel_extension_for_pytorch.cpu.runtime.Task: Generated
        intel_extension_for_pytorch.cpu.runtime.Task object.
    """

    def __init__(self, module, cpu_pool: CPUPool):
        self.cpu_pool = cpu_pool
        assert type(self.cpu_pool) is CPUPool
        if isinstance(module, torch.jit.ScriptModule):
            self._task = ipex._C.TaskModule(module._c, self.cpu_pool.cpu_pool, True)
        else:
            self._task = ipex._C.TaskModule(module, self.cpu_pool.cpu_pool)

    def __call__(self, *args, **kwargs):
        return self._task.run_async(*args, **kwargs)

    def run_sync(self, *args, **kwargs):
        return self._task.run_sync(*args, **kwargs)


default_multi_stream_module_concat_hint = MultiStreamModuleHint(0)


default_multi_stream_module_split_hint = MultiStreamModuleHint(0)


def get_default_num_streams(cpu_pool):
    return cpu_pool.core_ids.__len__()


class RoIAlign(nn.Module):
    """
    See :func:`roi_align`.
    """

    def __init__(self, output_size: BroadcastingList2[int], spatial_scale: float, sampling_ratio: int, aligned: bool=False):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: Tensor, rois: Tensor) ->Tensor:
        return F._roi_align.roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)

    def __repr__(self) ->str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ', aligned=' + str(self.aligned)
        tmpstr += ')'
        return tmpstr


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`

    Shape
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float=1e-05):
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        return torch.ops.torch_ipex.frozen_batch_norm(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps)


class EltwiseType(enum.IntEnum):
    NotFused = 0
    ReLU = 1
    Sigmoid = 2


class _IPEXLinear(torch.nn.Module):

    def __init__(self, dense_module, use_dnnl):
        super(_IPEXLinear, self).__init__()
        self.use_dnnl = use_dnnl
        self.batch_size_collapsed = None
        if hasattr(dense_module, 'input_shape'):
            self.batch_size_collapsed = 1
            for i in range(len(dense_module.input_shape) - 1):
                self.batch_size_collapsed *= dense_module.input_shape[i]
        if dense_module.bias is not None:
            self.bias = nn.Parameter(dense_module.bias.detach().clone(), requires_grad=dense_module.bias.requires_grad)
            if hasattr(dense_module, 'master_bias'):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, 'bias_trail'):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter('bias', None)
        if self.use_dnnl:
            self.ctx = torch.ops.ipex_prepack.linear_prepack(dense_module.weight, self.bias, self.batch_size_collapsed)
        else:
            self.ctx = torch.ops.ipex_prepack.mkl_sgemm_prepack(dense_module.weight, self.bias, self.batch_size_collapsed)
        self.weight = nn.Parameter(self.ctx.get_weight(), requires_grad=dense_module.weight.requires_grad)
        if hasattr(dense_module, 'master_weight'):
            self.master_weight = self.ctx.pack(dense_module.master_weight.detach().clone())
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = self.ctx.pack(dense_module.weight_trail.detach().clone())

    def forward(self, x):
        if self.use_dnnl:
            return torch.ops.torch_ipex.ipex_linear(x, self.weight, self.bias, self.ctx.get_data_handle())
        else:
            return torch.ops.torch_ipex.ipex_MKLSGEMM(x, self.weight, self.bias, self.ctx.get_data_handle())

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert not keep_vars, "can not using keep_vars true when to save _IPEXLinear's parameters"
        if self.bias is not None:
            if hasattr(self, 'master_bias'):
                bias = self.master_bias
            elif hasattr(self, 'bias_trail'):
                bias = torch.ops.torch_ipex.cat_bfloat16_float(self.bias, self.bias_trail)
            else:
                bias = self.bias
            destination[prefix + 'bias'] = bias.detach()
        if hasattr(self, 'master_weight'):
            weight = self.master_weight
        elif hasattr(self, 'weight_trail'):
            weight = torch.ops.torch_ipex.cat_bfloat16_float(self.weight, self.weight_trail)
        else:
            weight = self.weight
        destination[prefix + 'weight'] = self.ctx.to_public(weight.detach())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        assert False, '_IPEXLinear does not support _load_from_state_dict method'


class IPEXLinearEltwise(torch.nn.Module):

    def __init__(self, ipex_linear_module, eltwise='relu'):
        super(IPEXLinearEltwise, self).__init__()
        assert isinstance(ipex_linear_module, _IPEXLinear)
        self.m = ipex_linear_module
        if eltwise == 'relu':
            self.eltwise = EltwiseType.ReLU
        else:
            assert eltwise == 'sigmoid'
            self.eltwise = EltwiseType.Sigmoid

    def forward(self, x):
        return torch.ops.torch_ipex.ipex_linear_eltwise(x, self.m.weight, self.m.bias, self.eltwise, self.m.ctx.get_data_handle())


class EmbeddingSpec(NamedTuple):
    num_of_features: int
    feature_size: int
    pooling_modes: str
    dtype: torch.dtype
    weight: Optional[torch.Tensor]
    sparse: bool


class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1


class MergedEmbeddingBagFunc(Function):

    @staticmethod
    def unpack(*args):
        return args

    @staticmethod
    def forward(ctx, indices, offsets, indices_with_row_offsets, row_offsets, pooling_modes, *weights):
        output = torch.ops.torch_ipex.merged_embeddingbag_forward(indices, offsets, weights, pooling_modes)
        ctx.offsets = offsets
        ctx.weights = weights
        ctx.indices_with_row_offsets = indices_with_row_offsets
        ctx.row_offsets = row_offsets
        ctx.pooling_modes = pooling_modes
        return MergedEmbeddingBagFunc.unpack(*output)

    @staticmethod
    def backward(ctx, *grad_out):
        offsets = ctx.offsets
        weights = ctx.weights
        indices_with_row_offsets = ctx.indices_with_row_offsets
        row_offsets = ctx.row_offsets
        pooling_modes = ctx.pooling_modes
        grad_list = torch.ops.torch_ipex.merged_embeddingbag_backward_cpu(grad_out, offsets, weights, indices_with_row_offsets, row_offsets, pooling_modes)
        n_tables = len(weights)
        output = [None for i in range(5)]
        for grad in grad_list:
            output.append(grad)
        return MergedEmbeddingBagFunc.unpack(*output)


def merged_embeddingbag(indices, offsets, indices_with_row_offsets, row_offsets, pooling_modes, *weights):
    if torch.is_grad_enabled():
        return MergedEmbeddingBagFunc.apply(indices, offsets, indices_with_row_offsets, row_offsets, pooling_modes, *weights)
    return torch.ops.torch_ipex.merged_embeddingbag_forward(indices, offsets, weights, pooling_modes)


class MergedEmbeddingBag(nn.Module):
    """
    Merge multiple Pytorch EmbeddingBag (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py#L221) 
    as one torch.nn.Module.
    At the current stage:
    MergedEmbeddingBag assumes constructed by nn.EmbeddingBag with sparse=False, and will return dense grad.
    MergedEmbeddingBagWithSGD does not return any grad, backward and update are fused.
    Native usage for multiple EmbeddingBag is:
        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> inputs = [in1, in2, in3, ..., in_m]
        >>> outputs = []
        >>> for i in range(len(EmbLists)):
        >>>     outputs.append(Emb[in_i])
    Our optimized path will be:
        >>> merged_emb = MergedEmbeddingBagWithSGD(args)
        >>> outputs = MergedEmbeddingBagWithSGD(input)
    We will have benefits for our optimized path:
      1). We will save Pytorch OP dispatch overhead, if the EmbeddingBag operations are not
      heavy, this dispatch overhead will have big impact.
    We introduce "linearize_indices_and_offsets" step to merged indices/offsets together. But consider EmbeddingBags
    are usually the first layer of a model. So "linearize_indices_and_offsets" can be considered as "data prepocess" and
    can be done offline.
    This Module can not be used alone, we suggest to use MergedEmbeddingBagWith[Optimizer] instead.
    Now we can only choose MergedEmbeddingBagWithSGD and we plan to add more optimizer support
    in the future.
    For the introduction of MergedEmbeddingBagWith[Optimizer], please find the comments at
    MergedEmbeddingBagWithSGD.
    """
    embedding_specs: List[EmbeddingSpec]

    def __init__(self, embedding_specs: List[EmbeddingSpec]):
        super(MergedEmbeddingBag, self).__init__()
        self.n_tables = len(embedding_specs)
        self.weights = []
        row_offsets = []
        feature_sizes = []
        self.pooling_modes = []
        self.dtypes = []
        dtype = None
        self.alldense = True
        self.weights = torch.nn.ParameterList([nn.Parameter(torch.Tensor()) for i in range(len(embedding_specs))])
        for i, emb in enumerate(embedding_specs):
            num_of_features, feature_size, mode, dtype, weight, sparse = emb
            row_offsets.append(num_of_features)
            if mode == 'sum':
                self.pooling_modes.append(PoolingMode.SUM)
            elif mode == 'mean':
                self.pooling_modes.append(PoolingMode.MEAN)
            else:
                assert False, 'MergedEmbeddingBag only support EmbeddingBag with model sum or mean'
            if weight is None:
                weight = torch.empty((num_of_features, feature_size), dtype=dtype)
            self.weights[i] = nn.Parameter(weight)
            if sparse:
                self.alldense = False
        self.register_buffer('row_offsets', torch.tensor([0] + list(accumulate(row_offsets)), dtype=torch.int64))

    @classmethod
    def from_embeddingbag_list(cls, tables: List[torch.nn.EmbeddingBag]):
        embedding_specs = []
        for emb in tables:
            emb_shape = emb.weight.shape
            assert not emb.sparse, 'MergedEmbeddingBag can only be used for dense gradient EmebddingBag. Please use MergedEmbeddingBagWith[Optimizer] for sparse gradient.'
            embedding_specs.append(EmbeddingSpec(num_of_features=emb_shape[0], feature_size=emb_shape[1], pooling_modes=emb.mode, dtype=emb.weight.dtype, weight=emb.weight.detach(), sparse=emb.sparse))
        return cls(embedding_specs)

    def extra_repr(self) ->str:
        s = 'number of tables={}\n'.format(self.n_tables)
        for i in range(self.n_tables):
            s += 'table{}: {}, {}, {}, {}'.format(i, self.weights[i].shape[0], self.weights[i].shape[1], self.pooling_modes[i], self.weights[i].dtype)
            if i != self.n_tables - 1:
                s += '\n'
        return s

    def linearize_indices_and_offsets(self, indices: List[Tensor], offsets: List[Optional[Tensor]], include_last_offsets: List[bool]):
        """
        To make backward/update more balance, we only have 1 logical table in MergedEmbedingBag and
        use unified indices for access the whole logical table.
        We need to re-mark the indice from different tables to distinguish them.
        For example, we have  2 tables with shape [200, 128] and [100, 128].
        The indice 50 for table1 is still 50 and the indice 50 for table2 should be set to 50 + 200 = 250.
        We assume the original indice and offset will follow the usage for Pytorch EmbeddingBag:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py#L355-L382
        """

        def get_batch_size(indice, offset, include_last_offset):
            if indice.dim() == 2:
                assert offset is None, 'offset should be None if indice is 2-D tensor, https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py#L355-L382'
                batch_size = indice.shape[0]
            else:
                batch_size = offset.numel()
                if include_last_offset:
                    batch_size -= 1
            return batch_size
        assert self.n_tables == len(indices), 'expected {} but got {} indices'.format(self.n_tables, len(indices))
        assert self.n_tables == len(offsets), 'expected {} but got {} offsets'.format(self.n_tables, len(offsets))
        assert self.n_tables == len(include_last_offsets), 'expected {} but got {} include_last_offsets'.format(self.n_tables, len(include_last_offsets))
        batch_size = get_batch_size(indices[0], offsets[0], include_last_offsets[0])
        assert all(batch_size == get_batch_size(idx, offset, include_last) for idx, offset, include_last in zip(indices, offsets, include_last_offsets)), 'MergedEmbeddingBag only support input with same batch size'
        n_indices = sum([t.numel() for t in indices])
        n_offsets = batch_size * self.n_tables + 1
        merged_indices = torch.empty(n_indices, dtype=torch.int64)
        merged_indices_with_row_offsets = torch.empty(n_indices, dtype=torch.int64)
        merged_offsets = torch.empty(n_offsets, dtype=torch.int64)
        idx_start = 0
        offset_start = 0
        for i in range(self.n_tables):
            n_indice = indices[i].numel()
            merged_indices[idx_start:idx_start + n_indice].copy_(indices[i].view(-1))
            merged_indices_with_row_offsets[idx_start:idx_start + n_indice].copy_(indices[i].view(-1) + self.row_offsets[i])
            if indices[i].dim() == 2:
                bag_size = indices[i].shape[1]
                offset = torch.arange(0, indices[i].numel(), bag_size)
            else:
                offset = offsets[i][:-1] if include_last_offsets[i] else offsets[i]
            assert offset.numel() == batch_size
            merged_offsets[offset_start:offset_start + batch_size].copy_(offset + idx_start)
            idx_start += n_indice
            offset_start += batch_size
        assert idx_start == n_indices
        assert offset_start == n_offsets - 1
        merged_offsets[-1] = n_indices
        return merged_indices, merged_offsets, merged_indices_with_row_offsets

    def forward(self, input, need_linearize_indices_and_offsets=torch.BoolTensor([True])):
        """
        Args:
            input (Tuple[Tensor]): a tuple of (indices, offsets, include_last_offsets(if not merged)/indices_with_row_offsets(if merged))
            need_linearize_indices_and_offsets: indicate whether input need to be linearized
        Returns:
            List[Tensor] output shape of `(batch_size, feature_size)` which length = num of tables.
        """
        assert self.alldense, 'MergedEmbeddingBag only support EmbeddingBag List with all dense gradient, please use MergedEmbeddingBagWith[Optimizer] for sparse gridient EmbeddingBag'
        if need_linearize_indices_and_offsets.item():
            indices, offsets, include_last_offsets = input
            indices, offsets, indices_with_row_offsets = self.linearize_indices_and_offsets(indices, offsets, include_last_offsets)
        else:
            indices, offsets, indices_with_row_offsets = input
        return merged_embeddingbag(indices, offsets, indices_with_row_offsets, self.row_offsets, self.pooling_modes, *self.weights)


class SGDArgs(NamedTuple):
    bf16_trail: List[Optional[torch.Tensor]]
    weight_decay: float
    lr: float


class MergedEmbeddingBagSGDFunc(Function):

    @staticmethod
    def unpack(*args):
        return args

    @staticmethod
    def forward(ctx, indices, offsets, indices_with_row_offsets, row_offsets, pooling_modes, sgd_args, *weights):
        output = torch.ops.torch_ipex.merged_embeddingbag_forward(indices, offsets, weights, pooling_modes)
        ctx.indices = indices
        ctx.offsets = offsets
        ctx.weights = weights
        ctx.indices_with_row_offsets = indices_with_row_offsets
        ctx.row_offsets = row_offsets
        ctx.pooling_modes = pooling_modes
        ctx.sgd_args = sgd_args
        return MergedEmbeddingBagSGDFunc.unpack(*output)

    @staticmethod
    def backward(ctx, *grad_out):
        indices = ctx.indices
        offsets = ctx.offsets
        weights = ctx.weights
        indices_with_row_offsets = ctx.indices_with_row_offsets
        row_offsets = ctx.row_offsets
        pooling_modes = ctx.pooling_modes
        sgd_args = ctx.sgd_args
        bf16_trail = sgd_args.bf16_trail
        weight_decay = sgd_args.weight_decay
        lr = sgd_args.lr
        torch.ops.torch_ipex.merged_embeddingbag_backward_sgd(grad_out, indices, offsets, weights, indices_with_row_offsets, row_offsets, pooling_modes, bf16_trail, weight_decay, lr)
        n_tables = len(weights)
        output = [None for i in range(n_tables + 6)]
        return MergedEmbeddingBagSGDFunc.unpack(*output)


def merged_embeddingbag_sgd(indices, offsets, indices_with_row_offsets, row_offsets, pooling_modes, sgd_args, *weights):
    if torch.is_grad_enabled():
        return MergedEmbeddingBagSGDFunc.apply(indices, offsets, indices_with_row_offsets, row_offsets, pooling_modes, sgd_args, *weights)
    return torch.ops.torch_ipex.merged_embeddingbag_forward(indices, offsets, weights, pooling_modes)


class MergedEmbeddingBagWithSGD(MergedEmbeddingBag):
    """
    To support training for MergedEmbeddingBag with good performance, we fused optimizer step
    with backward function.
    Native usage for multiple EmbeddingBag is:
        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> sgd = torch.optim.SGD(EmbLists.parameters(), lr=lr, weight_decay=weight_decay)
        >>> inputs = [in1, in2, in3, ..., in_m]
        >>> outputs = []
        >>> for i in range(len(EmbLists)):
        >>>     outputs.append(Emb[in_i])
        >>> sgd.zero_grad()
        >>> for i in range(len(outputs)):
        >>>     out.backward(grads[i]) 
        >>> sgd.step()
    Our optimized path will be:
        >>> # create MergedEmbeddingBagWithSGD module with optimizer args (lr and weight decay)
        >>> merged_emb = MergedEmbeddingBagWithSGD(args)
        >>> merged_input = merged_emb.linearize_indices_and_offsets(inputs)
        >>> outputs = MergedEmbeddingBagWithSGD(merged_input)
        >>> outputs.backward(grads)
    We will get further benefits in training:
      1). We will futher save Pytorch OP dispatch overhead in backward and weight update process.
      2). We will make thread loading more balance during backward/weight update. In real
      world scenario, Embedingbag are often used to represent categorical features and the 
      categorical features will often fit power law distribution. For example, if we use one
      Embeddingtable to represent the age range the users of a video game website. We might
      find most of users are from 10-19 or 20-29. So we may need update the row which represent
      10-19 or 20-29 frequently. Since update these rows need to write at the same memory address,
      we need to write it by 1 thread (or we will have write conflict or have overhead to solve the conflict).
      By merge multiple table together, we will have more friendly distribution to distribute
      backward/update tasks.
      3). We will fuse update with backward together. We can immediately update the weight after
      we get grad from backward thus the memory pattern will be more friendly. We will have 
      more chance to access data from cache. 
    """
    embedding_specs: List[EmbeddingSpec]

    def __init__(self, embedding_specs: List[EmbeddingSpec], lr: float=0.01, weight_decay: float=0):
        super(MergedEmbeddingBagWithSGD, self).__init__(embedding_specs)
        self.sgd_args = self.init_sgd_args(lr, weight_decay)
        for i in range(self.n_tables):
            weight = self.weights[i]
            if weight.dtype == torch.bfloat16:
                self.sgd_args.bf16_trail.append(torch.zeros_like(weight, dtype=torch.bfloat16))
            else:
                self.sgd_args.bf16_trail.append(torch.empty(0, dtype=torch.bfloat16))

    def init_sgd_args(self, lr, weight_decay, bf16_trail=[]):
        if lr < 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if weight_decay < 0.0:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        return SGDArgs(weight_decay=weight_decay, lr=lr, bf16_trail=bf16_trail)

    def to_bfloat16_train(self):
        """
        Cast weight to bf16 and it's trail part for training
        """
        trails = []
        for i in range(len(self.weights)):
            if self.weights[i].dtype == torch.float:
                bf16_w, trail = torch.ops.torch_ipex.split_float_bfloat16(self.weights[i])
            elif self.weights[i].dtype == torch.bfloat16:
                bf16_w = self.weights[i]
                trail = torch.zeros_like(bf16_w, dtype=torch.bfloat16)
            elif self.weights[i].dtype == torch.double:
                bf16_w, trail = torch.ops.torch_ipex.split_float_bfloat16(self.weights[i].float())
            else:
                assert False, 'MergedEmbeddingBag only support dtypes with bfloat, float and double'
            trails.append(trail)
            self.weights[i] = torch.nn.Parameter(bf16_w)
        self.sgd_args = self.sgd_args._replace(bf16_trail=trails)

    def forward(self, input, need_linearize_indices_and_offsets=torch.BoolTensor([True])):
        """
        Args:
            input (Tuple[Tensor]): a tuple of (indices, offsets, include_last_offsets(if not merged)/indices_with_row_offsets(if merged))
            need_linearize_indices_and_offsets: indicate whether input need to be linearized
        Returns:
            List[Tensor] output shape of `(batch_size, feature_size)` which length = num of tables.
        """
        if need_linearize_indices_and_offsets.item():
            indices, offsets, include_last_offsets = input
            indices, offsets, indices_with_row_offsets = self.linearize_indices_and_offsets(indices, offsets, include_last_offsets)
        else:
            indices, offsets, indices_with_row_offsets = input
        return merged_embeddingbag_sgd(indices, offsets, indices_with_row_offsets, self.row_offsets, self.pooling_modes, self.sgd_args, *self.weights)

    @classmethod
    def from_embeddingbag_list(cls, tables: List[torch.nn.EmbeddingBag], lr: float=0.01, weight_decay: float=0):
        embedding_specs = []
        for emb in tables:
            emb_shape = emb.weight.shape
            embedding_specs.append(EmbeddingSpec(num_of_features=emb_shape[0], feature_size=emb_shape[1], pooling_modes=emb.mode, dtype=emb.weight.dtype, weight=emb.weight.detach(), sparse=emb.sparse))
        return cls(embedding_specs, lr, weight_decay)


class _LSTM(torch.nn.LSTM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, hx=None):
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            return super(_LSTM, self).forward(input, hx)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.num_layers * num_directions, max_batch_size, real_hidden_size, dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.num_layers * num_directions, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            hx = h_zeros, c_zeros
        else:
            hx = self.permute_hidden(hx, sorted_indices)
        self.check_forward_args(input, hx, batch_sizes)
        result = torch.ops.torch_ipex.ipex_lstm(input, hx, self._flat_weights, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first)
        output = result[0]
        hidden = result[1:]
        return output, self.permute_hidden(hidden, unsorted_indices)


class _IPEXConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'out_channels', 'kernel_size']

    def __init__(self, dense_module):
        super(_IPEXConvNd, self).__init__()
        self.out_channels = dense_module.out_channels
        self.in_channels = dense_module.in_channels
        self.kernel_size = dense_module.kernel_size
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups
        self.prepack_input_shape = dense_module.input_shape if hasattr(dense_module, 'input_shape') else []
        self.weight_channels_last = dense_module.weight.is_contiguous(memory_format=torch.channels_last) or dense_module.weight.is_contiguous(memory_format=torch.channels_last_3d)
        if dense_module.bias is not None:
            self.bias = nn.Parameter(dense_module.bias.detach().clone(), requires_grad=dense_module.bias.requires_grad)
            if hasattr(dense_module, 'master_bias'):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, 'bias_trail'):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter('bias', None)
        self.ctx = torch.ops.ipex_prepack.convolution_prepack(dense_module.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.weight_channels_last, self.prepack_input_shape)
        self.weight = nn.Parameter(self.ctx.get_weight(), requires_grad=dense_module.weight.requires_grad)
        if hasattr(dense_module, 'master_weight'):
            self.master_weight = self.ctx.pack(dense_module.master_weight.detach().clone())
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = self.ctx.pack(dense_module.weight_trail.detach().clone())

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert not keep_vars, "can not using keep_vars true when to save _IPEXConvNd's parameters"
        if self.bias is not None:
            if hasattr(self, 'master_bias'):
                bias = self.master_bias
            elif hasattr(self, 'bias_trail'):
                bias = torch.ops.torch_ipex.cat_bfloat16_float(self.bias, self.bias_trail)
            else:
                bias = self.bias
            destination[prefix + 'bias'] = bias.detach()
        if hasattr(self, 'master_weight'):
            weight = self.master_weight
        elif hasattr(self, 'weight_trail'):
            weight = torch.ops.torch_ipex.cat_bfloat16_float(self.weight, self.weight_trail)
        else:
            weight = self.weight
        destination[prefix + 'weight'] = self.ctx.to_public(weight.detach())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        assert False, '_IPEXConvNd does not support _load_from_state_dict method'

    def forward(self, x):
        return torch.ops.torch_ipex.convolution_forward(x, self.weight, self.bias, self.ctx.get_data_handle())


class _IPEXConv1d(_IPEXConvNd):

    def __init__(self, dense_module):
        super(_IPEXConv1d, self).__init__(dense_module)


class _IPEXConv2d(_IPEXConvNd):

    def __init__(self, dense_module):
        super(_IPEXConv2d, self).__init__(dense_module)


class _IPEXConv3d(_IPEXConvNd):

    def __init__(self, dense_module):
        super(_IPEXConv3d, self).__init__(dense_module)


class _IPEXConvTransposeNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'out_channels', 'kernel_size', 'output_padding']

    def __init__(self, dense_module):
        super(_IPEXConvTransposeNd, self).__init__()
        self.out_channels = dense_module.out_channels
        self.in_channels = dense_module.in_channels
        self.kernel_size = dense_module.kernel_size
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.groups = dense_module.groups
        self.dilation = dense_module.dilation
        self.output_padding = dense_module.output_padding
        self.prepack_input_shape = dense_module.input_shape if hasattr(dense_module, 'input_shape') else []
        self.weight_channels_last = dense_module.weight.is_contiguous(memory_format=torch.channels_last) or dense_module.weight.is_contiguous(memory_format=torch.channels_last_3d)
        if dense_module.bias is not None:
            self.bias = nn.Parameter(dense_module.bias.detach().clone(), requires_grad=dense_module.bias.requires_grad)
            if hasattr(dense_module, 'master_bias'):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, 'bias_trail'):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter('bias', None)
        self.ctx = torch.ops.ipex_prepack.conv_transpose_prepack(dense_module.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation, self.weight_channels_last, self.prepack_input_shape)
        self.weight = nn.Parameter(self.ctx.get_weight(), requires_grad=dense_module.weight.requires_grad)
        if hasattr(dense_module, 'master_weight'):
            self.master_weight = self.ctx.pack(dense_module.master_weight.detach().clone())
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = self.ctx.pack(dense_module.weight_trail.detach().clone())

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert not keep_vars, "can not using keep_vars true when to save _IPEXConvTransposeNd's parameters"
        if self.bias is not None:
            if hasattr(self, 'master_bias'):
                bias = self.master_bias
            elif hasattr(self, 'bias_trail'):
                bias = torch.ops.torch_ipex.cat_bfloat16_float(self.bias, self.bias_trail)
            else:
                bias = self.bias
            destination[prefix + 'bias'] = bias.detach()
        if hasattr(self, 'master_weight'):
            weight = self.master_weight
        elif hasattr(self, 'weight_trail'):
            weight = torch.ops.torch_ipex.cat_bfloat16_float(self.weight, self.weight_trail)
        else:
            weight = self.weight
        destination[prefix + 'weight'] = self.ctx.to_public(weight.detach())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        assert False, '_IPEXConvTransposeNd does not support _load_from_state_dict method'

    def forward(self, x):
        return torch.ops.torch_ipex.conv_transpose(x, self.weight, self.bias, self.ctx.get_data_handle())


class _IPEXConvTranspose2d(_IPEXConvTransposeNd):

    def __init__(self, dense_module):
        super(_IPEXConvTranspose2d, self).__init__(dense_module)


class _IPEXConvTranspose3d(_IPEXConvTransposeNd):

    def __init__(self, dense_module):
        super(_IPEXConvTranspose3d, self).__init__(dense_module)


OpConvertInfo = Tuple[List[Optional[Tuple[float, int, torch.dtype]]], List[bool]]


class OpQuantizeabilityType(enum.Enum):
    QUANTIZEABLE = 0
    NOT_QUANTIZEABLE = 1


def _raise_obs_not_found_error(func):
    raise RuntimeError(f'Encountered arithmetic operation {torch.typename(func)} but we have encountered fewer arithmetic operations in previous calibration runs. This likely indicates that the program contains dynamic control flow.  Quantization is not defined over dynamic control flow!')


def _raise_obs_op_mismatch(func, prev_op):
    raise RuntimeError(f'Encountered arithmetic operation {torch.typename(func)} but previously recorded operation was {prev_op}!. This likely indicates that the program contains dynamic control flow. Quantization is not defined over dynamic control flow!')


conv_linear_ops = [str(F.conv2d), str(F.conv3d), str(torch.conv2d), str(torch.conv3d), str(F.conv_transpose2d), str(F.conv_transpose3d), str(torch.conv_transpose2d), str(torch.conv_transpose3d), str(F.linear), str(torch._C._nn.linear)]


embedding_op = [str(F.embedding_bag), str(torch.embedding_bag)]


def get_input_observed_arg_idxs(op_type: str, op_type_is_module: bool) ->Optional[List[int]]:
    if op_type_is_module and op_type != str(torch.nn.EmbeddingBag):
        return [0]
    elif op_type in conv_linear_ops:
        return [0, 1]
    elif op_type in embedding_op:
        return [1]
    return None


def get_weight_arg_idx(op: str) ->Optional[int]:
    if op in conv_linear_ops:
        return 1
    return None


class InteractionFunc(Function):

    @staticmethod
    def forward(ctx, *args):
        ctx.save_for_backward(*args)
        output = torch.ops.torch_ipex.interaction_forward(args)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        args = ctx.saved_tensors
        grad_in = torch.ops.torch_ipex.interaction_backward(grad_out.contiguous(), args)
        return tuple(grad_in)


def interaction(*args):
    """
    Get the interaction feature beyond different kinds of features (like gender
    or hobbies), used in DLRM model.

    For now, we only optimized "dot" interaction at `DLRM Github repo <https://github.com/facebookresearch/dlrm/blob/main/dlrm_s_pytorch.py#L475-L495>`_.
    Through this, we use the dot product to represent the interaction feature
    between two features.

    For example, if feature 1 is "Man" which is represented by [0.1, 0.2, 0.3],
    and feature 2 is "Like play football" which is represented by [-0.1, 0.3, 0.2].

    The dot interaction feature is
    ([0.1, 0.2, 0.3] * [-0.1, 0.3, 0.2]^T) =  -0.1 + 0.6 + 0.6 = 1.1

    Args:
        *args: Multiple tensors which represent different features

    Shape
        - Input: :math:`N * (B, D)`, where N is the number of different kinds of features, B is the batch size, D is feature size
        - Output: :math:`(B, D + N * ( N - 1 ) / 2)`
    """
    if torch.is_grad_enabled():
        return InteractionFunc.apply(*args)
    return torch.ops.torch_ipex.interaction_forward(args)


conv_linear_modules = [str(torch.nn.Conv2d), str(torch.nn.Conv3d), str(torch.nn.ConvTranspose2d), str(torch.nn.ConvTranspose3d), str(torch.nn.Linear)]


def iterate_and_apply_convert(args: Any, quant_infos: List[Optional[Tuple[float, int, torch.dtype]]], quant_or_dequant_needed: List[bool], op: Callable, flattened_tensor_infos_idx=None) ->Any:
    """
    Inputs:
      `args`: arguments to a function, may contain nested types, for example:
        ([torch.Tensor, torch.Tensor], int, (int, int))
      `quant_infos`: tensor information containers for each tensor
        in `args`, flattened, for example corresponding with above:
        ({...}, {...}, None, None, None)
       `quant_or_dequant_needed`: tensor information about whether do quantization
        containers for each tensorin `args`,
      `op`: cur quantizable op
    Returns `new_args`, where each tensor has been transformed by `func`.
    """
    if flattened_tensor_infos_idx is None:
        flattened_tensor_infos_idx = [0]
    if isinstance(args, tuple):
        new_args = []
        for arg in args:
            new_arg = iterate_and_apply_convert(arg, quant_infos, quant_or_dequant_needed, op, flattened_tensor_infos_idx)
            new_args.append(new_arg)
        return tuple(new_args)
    elif isinstance(args, list):
        new_args = []
        for arg in args:
            new_arg = iterate_and_apply_convert(arg, quant_infos, quant_or_dequant_needed, op, flattened_tensor_infos_idx)
            new_args.append(new_arg)
        return new_args
    else:
        cur_quant_infos = quant_infos[flattened_tensor_infos_idx[0]]
        cur_quant_or_dequant_needed = quant_or_dequant_needed[flattened_tensor_infos_idx[0]]
        if cur_quant_infos is not None and cur_quant_or_dequant_needed and isinstance(args, torch.Tensor):
            scale, zp, dtype = cur_quant_infos
            if str(op) in conv_linear_ops and get_weight_arg_idx(str(op)) == flattened_tensor_infos_idx[0] and isinstance(scale, torch.Tensor) and scale.numel() > 1:
                ch_axis = 0
                if str(op) in [str(F.conv_transpose2d), str(torch.conv_transpose2d), str(F.conv_transpose3d), str(torch.conv_transpose3d)]:
                    ch_axis = 1
                if torch.is_autocast_cpu_enabled() and core.get_autocast_dtype() == torch.bfloat16:
                    if args.dtype == torch.float32:
                        args = args
                    args = args
                    args = torch.quantize_per_channel(args, scale, zp, ch_axis, dtype)
                    args = args.dequantize()
                    args = args
                else:
                    args = torch.quantize_per_channel(args, scale, zp, ch_axis, dtype)
                    args = args.dequantize()
            elif str(op) in conv_linear_ops + [str(torch.matmul), str(torch.Tensor.matmul)] + embedding_op or str(type(op)) in conv_linear_modules:
                if torch.is_autocast_cpu_enabled() and core.get_autocast_dtype() == torch.bfloat16:
                    if args.dtype == torch.float32:
                        args = args
                    args = args
                    args = torch.quantize_per_tensor(args, scale.item(), zp.item(), dtype)
                    args = args.dequantize()
                    args = args
                else:
                    args = torch.quantize_per_tensor(args, scale.item(), zp.item(), dtype)
                    args = args.dequantize()
            else:
                args_is_bfloat16 = False
                if args.dtype == torch.bfloat16:
                    args_is_bfloat16 = True
                    args = args
                args = torch.quantize_per_tensor(args, scale.item(), zp.item(), dtype)
                args = args.dequantize()
                if args_is_bfloat16:
                    args = args
        flattened_tensor_infos_idx[0] += 1
        return args


functions_supported_by_quantization = set([torch.Tensor.add, torch.add, torch.Tensor.relu, torch.flatten, torch.Tensor.flatten, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d, F.avg_pool2d, F.avg_pool3d, F.max_pool2d, F.max_pool3d, F.conv2d, F.conv3d, torch.conv2d, torch.conv3d, F.conv_transpose2d, F.conv_transpose3d, torch.conv_transpose2d, torch.conv_transpose3d, torch.relu, F.relu, F.linear, torch._C._nn.linear, torch.matmul, torch.Tensor.matmul, F.embedding_bag, torch.embedding_bag])


may_inplace_module = set([torch.nn.ReLU])


module_types_supported_by_quantization = set([torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d, torch.nn.Linear, torch.nn.MaxPool2d, torch.nn.MaxPool3d, torch.nn.AvgPool2d, torch.nn.AvgPool3d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool3d, torch.nn.ReLU, torch.nn.EmbeddingBag, torch.nn.Flatten, torch.nn.LSTM, nnqd.Linear, nnqd.LSTM])


def op_needs_quantization(op: Callable) ->bool:
    if op in functions_supported_by_quantization or op in functions_supported_by_quantization_ipex:
        return True
    elif type(op) in module_types_supported_by_quantization:
        if op in may_inplace_module and op.inplace:
            return False
        return True
    else:
        return False


a_related_to_b = (str(torch.add), str(torch.Tensor.add)), (str(torch.Tensor.add), str(torch.add)), (str(torch.nn.Linear), str(nnqd.Linear)), (str(nnqd.Linear), str(torch.nn.Linear)), (str(torch.nn.LSTM), str(nnqd.LSTM)), (str(nnqd.LSTM), str(torch.nn.LSTM))


def ops_are_related(cur_op: Callable, expected_op_type: str, type_is_module: bool) ->bool:
    """
    This function is to check whether the cur_op is align with the saved op_type, which make sure
    the model doesn't have dynamic workflow.
    """
    if type_is_module:
        cur_op = type(cur_op)
    return str(cur_op) == expected_op_type or (str(cur_op), expected_op_type) in a_related_to_b


quantized_modules_has_weights = set([torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear, torch.nn.EmbeddingBag, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d, torch.nn.LSTM])


class AutoQuantizationStateModuleDict(torch.nn.ModuleDict):
    pass


class Interaction(torch.nn.Module):

    def __init__(self):
        super(Interaction, self).__init__()

    def forward(self, x):
        return ipex.nn.functional.interaction(*x)


class EmbeddingBagList(torch.nn.Module):

    def __init__(self, max_rows, vector_size):
        super(EmbeddingBagList, self).__init__()
        self.emb_list = torch.nn.ModuleList()
        for n_f in max_rows:
            self.emb_list.append(torch.nn.EmbeddingBag(n_f, vector_size, mode='sum', sparse=True))

    def forward(self, indices, offsets):
        ly = []
        for k, sparse_index_group_batch in enumerate(indices):
            sparse_offset_group_batch = offsets[k]
            E = self.emb_list[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)
            ly.append(V)
        return ly


class ConvBatchNorm(torch.nn.Module):

    def __init__(self):
        super(ConvBatchNorm, self).__init__()
        self.input1 = torch.randn(1, 3, 224, 224)
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBatchNormSoftmax(torch.nn.Module):

    def __init__(self):
        super(ConvBatchNormSoftmax, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        return nn.Softmax(dim=-1)(self.bn(self.conv(x)))


class TestModule(torch.nn.Module):

    def __init__(self):
        super(TestModule, self).__init__()
        self.linear = torch.nn.Linear(5, 10)
        self.conv1d = torch.nn.Conv1d(1, 10, 5, 1)
        self.conv2d = torch.nn.Conv2d(1, 10, 5, 1)
        self.conv3d = torch.nn.Conv3d(1, 10, 5, 1)
        self.transpose_conv1d = torch.nn.ConvTranspose1d(1, 10, 5, 1)
        self.transpose_conv2d = torch.nn.ConvTranspose2d(1, 10, 5, 1)
        self.transpose_conv3d = torch.nn.ConvTranspose3d(1, 10, 5, 1)
        self.bn = torch.nn.BatchNorm2d(num_features=10)
        self.embeddingbag = torch.nn.EmbeddingBag(10, 3, mode='sum')
        self.embedding = torch.nn.Embedding(10, 3)
        table0 = torch.nn.EmbeddingBag(100, 16, mode='mean', sparse=False)
        table1 = torch.nn.EmbeddingBag(50, 32, mode='sum', sparse=False)
        self.merged = MergedEmbeddingBag.from_embeddingbag_list([table0, table1])

    def forward(self, x):
        x = self.conv2d(x)
        return


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(x)


default_static_qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8), weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))


LLGA_FUSION_GROUP = 'ipex::LlgaFusionGroup'


def findFusionGroups(graph):
    result = []
    for n in graph.nodes():
        if n.kind() == LLGA_FUSION_GROUP:
            result.append(n.g('Subgraph'))
            continue
        for block in n.blocks():
            result += findFusionGroups(block)
    return result


def warmup_forward(f, *args, profiling_count=2):
    for i in range(profiling_count):
        results = f(*args)
    return results


class JitLlgaTestCase(JitTestCase):

    def checkScript(self, m, x, freeze=True):
        if isinstance(m, torch.nn.Module):
            m.eval()
        with torch.no_grad():
            ref = m(*x)
            scripted = torch.jit.script(m)
            if isinstance(scripted, torch.nn.Module) and freeze:
                scripted = torch.jit.freeze(scripted)
            warmup_forward(scripted, *x)
            graph = scripted.graph_for(*x)
            y = scripted(*x)
            self.assertEqual(y, ref)
        return graph, scripted

    def checkTrace(self, m, x, freeze=True, *args, **kwargs):
        if isinstance(m, torch.nn.Module):
            m.eval()
        with torch.no_grad(), torch._jit_internal._disable_emit_hooks():
            traced = torch.jit.trace(m, x)
            if isinstance(traced, torch.nn.Module) and freeze:
                traced = torch.jit.freeze(traced)
            warmup_forward(traced, *x)
            fwd_graph = traced.graph_for(*x)
            ref_o = m(*x)
            jit_o = traced(*x)
            self.assertEqual(jit_o, ref_o)
        return fwd_graph, traced

    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    def checkQuantizeTrace(self, model, x, atol=0.001, rtol=0.01, x_var=None, qconfig=default_static_qconfig, int8_bf16=False, freeze=True):
        graph, traced_model, fp32_model = self.prepareModel(model, x, qconfig, int8_bf16, freeze=freeze)
        with torch.no_grad():
            y = fp32_model(*x)
            y = y if int8_bf16 else y
            y_llga = traced_model(*x)
            self.assertEqual(y, y_llga, atol=atol, rtol=rtol)
            if x_var:
                y_var = fp32_model(*x_var)
                y_var = y_var if int8_bf16 else y_var
                y_var_llga = traced_model(*x_var)
                self.assertEqual(y_var, y_var_llga, atol=atol, rtol=rtol)
            return graph

    def prepareModel(self, model, x, qconfig=default_static_qconfig, int8_bf16=False, prepare_inplace=True, convert_inplace=True, freeze=True):
        model.eval()
        fp32_model = copy.deepcopy(model)
        with torch.no_grad(), torch._jit_internal._disable_emit_hooks():
            ipex.nn.utils._model_convert.replace_dropout_with_identity(model)
            model = ipex.quantization.prepare(model, qconfig, x, inplace=prepare_inplace)
            y = model(*x)
            if int8_bf16:
                with torch.cpu.amp.autocast():
                    convert_model = ipex.quantization.convert(model, inplace=convert_inplace)
                    traced_model = torch.jit.trace(convert_model, x)
            else:
                convert_model = ipex.quantization.convert(model, inplace=convert_inplace)
                traced_model = torch.jit.trace(convert_model, x)
            if freeze:
                traced_model = torch.jit.freeze(traced_model)
            y0 = traced_model(*x)
            graph = traced_model.graph_for(*x)
            return graph, traced_model, fp32_model

    def checkPatterns(self, graph, patterns):
        fusion_groups = findFusionGroups(graph)
        assert len(fusion_groups) == len(patterns), 'length of subgraphs not equal to length of given patterns'
        for i in range(len(fusion_groups)):
            for pattern in patterns[i]:
                self.assertGraphContains(fusion_groups[i], pattern)

    def checkAttr(self, graph, node, attr):

        def count(block, node, attr):
            for n in block.nodes():
                if n.kind() == node:
                    self.assertFalse(n.hasAttribute('qtype'))
                for block in n.blocks():
                    count(block, node, attr)
        count(graph, node, attr)


def llga_fp32_bf16_test_env(func):

    @wraps(func)
    def wrapTheFunction(*args):
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_profiling_executor(True)
        ipex._C.set_llga_fp32_bf16_enabled(True)
        func(*args)
        ipex._C.set_llga_fp32_bf16_enabled(False)
    return wrapTheFunction


class M(torch.nn.Module):

    def __init__(self):
        super(M, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(10)

    def forward(self, x):
        x = self.layer_norm(x)
        return x


def get_rand_seed():
    return int(time.time() * 1000000000)


class TestLSTM(TorchTestCase):

    def _lstm_params_list(self):
        params_dict = {'input_size': [1, 2], 'hidden_size': [5, 32], 'num_layers': [1, 3], 'bidirectional': [False, True], 'bias': [False, True], 'empty_state': [False, True], 'batch_first': [False, True], 'dropout': [0, 0.4, 0.7, 1], 'batch_size': [1, 2], 'seq_len': [1, 3]}
        params_list = []
        for key, value in params_dict.items():
            params_list.append(value)
        return params_list

    def _cast_dtype(self, input, bf16):
        if bf16:
            input = input
        return input

    def _test_lstm(self, training, bf16, rtol=1.3e-06, atol=1e-05):
        rand_seed = int(get_rand_seed())
        None
        torch.manual_seed(rand_seed)
        params_list = self._lstm_params_list()
        for input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, dropout, batch_size, seq_len in itertools.product(*params_list):
            if dropout > 0 and num_layers == 1:
                continue
            num_directions = 2 if bidirectional else 1
            if batch_first:
                input = torch.randn(batch_size, seq_len, input_size)
            else:
                input = torch.randn(seq_len, batch_size, input_size)
            h = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            c = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            input_cpu = input.clone().requires_grad_(training)
            h_cpu = h.clone().requires_grad_(training)
            c_cpu = c.clone().requires_grad_(training)
            model_cpu = M(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)
            model_cpu.train() if training else model_cpu.eval()
            input_ipex = input.clone().requires_grad_(training)
            h_ipex = h.clone().requires_grad_(training)
            c_ipex = c.clone().requires_grad_(training)
            model_ipex = copy.deepcopy(model_cpu)
            model_ipex.train() if training else model_ipex.eval()
            ipex.nn.utils._model_convert.replace_lstm_with_ipex_lstm(model_ipex, None)
            with torch.cpu.amp.autocast(enabled=bf16, dtype=torch.bfloat16):
                if empty_state:
                    torch.manual_seed(rand_seed)
                    y_cpu, hy_cpu = self._cast_dtype(model_cpu, bf16)(self._cast_dtype(input_cpu, bf16))
                    torch.manual_seed(rand_seed)
                    y_ipex, hy_ipex = model_ipex(input_ipex)
                else:
                    torch.manual_seed(rand_seed)
                    y_cpu, hy_cpu = self._cast_dtype(model_cpu, bf16)(self._cast_dtype(input_cpu, bf16), (self._cast_dtype(h_cpu, bf16), self._cast_dtype(c_cpu, bf16)))
                    torch.manual_seed(rand_seed)
                    y_ipex, hy_ipex = model_ipex(input_ipex, (h_ipex, c_ipex))
                self.assertEqual(y_cpu, y_ipex, rtol=rtol, atol=atol)
                self.assertEqual(hy_cpu[0], hy_ipex[0], rtol=rtol, atol=atol)
                self.assertEqual(hy_cpu[1], hy_ipex[1], rtol=rtol, atol=atol)
                if training:
                    y_cpu.sum().backward(retain_graph=True)
                    y_ipex.sum().backward(retain_graph=True)
                    self.assertEqual(input_ipex.grad, input_cpu.grad, rtol=rtol, atol=atol)
                    self.assertEqual(self._cast_dtype(model_ipex.lstm.weight_ih_l0.grad, bf16), model_cpu.lstm.weight_ih_l0.grad, rtol=rtol, atol=atol)
                    self.assertEqual(self._cast_dtype(model_ipex.lstm.weight_hh_l0.grad, bf16), model_cpu.lstm.weight_hh_l0.grad, rtol=rtol, atol=atol)
                    if bias:
                        self.assertEqual(self._cast_dtype(model_ipex.lstm.bias_ih_l0.grad, bf16), model_cpu.lstm.bias_ih_l0.grad, rtol=rtol, atol=atol)
                        self.assertEqual(self._cast_dtype(model_ipex.lstm.bias_hh_l0.grad, bf16), model_cpu.lstm.bias_hh_l0.grad, rtol=rtol, atol=atol)
                    if not empty_state:
                        hy_cpu[0].sum().backward(retain_graph=True)
                        hy_ipex[0].sum().backward(retain_graph=True)
                        self.assertEqual(h_ipex.grad, h_cpu.grad, rtol=rtol, atol=atol)
                        hy_cpu[1].sum().backward(retain_graph=True)
                        hy_ipex[1].sum().backward(retain_graph=True)
                        self.assertEqual(c_ipex.grad, c_cpu.grad, rtol=rtol, atol=atol)

    def _test_lstm_pack_padded_sequence(self):
        embedding_dim = 1024
        hidden_dim = 10
        batch_size = 24
        num_layers = 1
        bidirectional = True
        num_direc = 2
        max_lens = 96
        sent = torch.randn(batch_size, max_lens, embedding_dim)
        hid_0 = torch.rand(num_layers * num_direc, batch_size, hidden_dim)
        hid_1 = torch.randn(num_layers * num_direc, batch_size, hidden_dim)
        sentences = sent.clone().requires_grad_(False)
        sent_lens = torch.Tensor([1, 2, 3, 4, 5, 1, 3, 2, 96, 5, 3, 1, 1, 2, 1, 2, 3, 6, 1, 2, 4, 6, 2, 1])
        assert sent_lens.shape[0] == batch_size
        assert sent_lens.max().item() == max_lens
        hidden_0 = hid_0.clone().requires_grad_(False)
        hidden_1 = hid_1.clone().requires_grad_(False)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(sentences, sent_lens, batch_first=True, enforce_sorted=False)
        model = M(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, bias=True, dropout=0.2)
        model_ipex = copy.deepcopy(model)
        ipex.nn.utils._model_convert.replace_lstm_with_ipex_lstm(model_ipex, None)
        lstm_out, hidden_out = model(embeds, (hidden_0, hidden_1))
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out_ipex, hidden_out_ipex = model_ipex(embeds, (hidden_0, hidden_1))
        lstm_out_ipex, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_ipex, batch_first=True)
        self.assertEqual(lstm_out, lstm_out_ipex)
        self.assertEqual(hidden_out[0], hidden_out_ipex[0])
        self.assertEqual(hidden_out[1], hidden_out_ipex[1])

    def test_lstm_op(self):
        self._test_lstm(training=False, bf16=False)
        self._test_lstm(training=False, bf16=True, rtol=0.02, atol=0.02)
        self._test_lstm(training=True, bf16=False)
        self._test_lstm(training=True, bf16=True, rtol=0.02, atol=0.03)

    def test_lstm_pack_padded_sequence(self):
        self._test_lstm_pack_padded_sequence()


class Model(nn.Module):

    def __init__(self, ic, oc, bias):
        super(Model, self).__init__()
        self.linear = nn.Linear(ic, oc, bias=bias)

    def forward(self, input):
        return self.linear(input)


class inplace_softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = x + 1
        x2 = nn.Softmax(dim=-1)(x1)
        return x2


class LinearEltwise(nn.Module):

    def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
        super(LinearEltwise, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.eltwise = eltwise_fn
        self.kwargs = kwargs

    def forward(self, x):
        a = self.linear(x)
        a = a / 2
        b = self.eltwise(a, **self.kwargs)
        return b


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        return self.conv(x)


convtranspose_module = {(2): torch.nn.ConvTranspose2d, (3): torch.nn.ConvTranspose3d}


class ConvTranspose(nn.Module):

    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose, self).__init__()
        self.conv_transpose = convtranspose_module[dim](in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)

    def forward(self, x):
        x = self.conv_transpose(x)
        return x


class Linear(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        return self.linear(x)


bn_module = {(2): torch.nn.BatchNorm2d, (3): torch.nn.BatchNorm3d}


conv_module = {(1): torch.nn.Conv1d, (2): torch.nn.Conv2d, (3): torch.nn.Conv3d}


class Conv_Bn_Relu(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(Conv_Bn_Relu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](out_channels, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Conv_IF_Relu(nn.Module):

    def __init__(self):
        super(Conv_IF_Relu, self).__init__()
        self.conv = torch.nn.Conv2d(6, 3, 3)

    def forward(self, x):
        if x.sum().item() > 0:
            return F.relu(self.conv(x), inplace=True)
        else:
            return F.relu(self.conv(x))


class LinearBatchNormNd(torch.nn.Module):

    def __init__(self, dim):
        super(LinearBatchNormNd, self).__init__()
        self.linear = torch.nn.Linear(32, 32)
        if dim == 1:
            self.input1 = torch.randn(1, 32)
            self.bn = torch.nn.BatchNorm1d(32)
        elif dim == 2:
            self.input1 = torch.randn(1, 32, 32, 32)
            self.bn = torch.nn.BatchNorm2d(32)
        elif dim == 3:
            self.input1 = torch.randn(1, 32, 32, 32, 32)
            self.bn = torch.nn.BatchNorm3d(32)

    def forward(self, x):
        return self.bn(self.linear(x))


class ConvBatchNormLinearBatchNorm(torch.nn.Module):

    def __init__(self):
        super(ConvBatchNormLinearBatchNorm, self).__init__()
        self.input1 = torch.randn(1, 32, 32, 32)
        self.conv = torch.nn.Conv2d(32, 32, 1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.linear = torch.nn.Linear(32, 32)
        self.bn2 = torch.nn.BatchNorm2d(32)

    def forward(self, x):
        return self.bn2(self.linear(self.bn1(self.conv(x))))


class TwoLayerMLP(torch.nn.Module):

    def __init__(self):
        super(TwoLayerMLP, self).__init__()
        self.input1 = torch.randn(2, 2)
        self.input2 = torch.randn(3, 3)
        self.l1 = torch.nn.Linear(2, 2)
        self.l2 = torch.nn.Linear(3, 3)

    def forward(self, x1, x2):
        return self.l1(x1).sum() + self.l2(x2).sum()


class OneLayerMLP(torch.nn.Module):

    def __init__(self):
        super(OneLayerMLP, self).__init__()
        self.input1 = torch.randn(2, 2)
        self.l1 = torch.nn.Linear(2, 2)

    def forward(self, x1):
        return self.l1(x1)


class ConvTranspose2d(torch.nn.Module):

    def __init__(self):
        super(ConvTranspose2d, self).__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(5, 5, (3, 3))
        self.input1 = torch.randn(5, 5, 3, 3)

    def forward(self, x):
        x = self.conv_transpose2d(x)
        return x


class ConvEltwise(nn.Module):

    def __init__(self, eltwise_fn, dim, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvEltwise, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size)
        self.eltwise = eltwise_fn
        self.kwargs = kwargs

    def forward(self, x):
        a = self.conv(x)
        b = self.eltwise(a, **self.kwargs)
        return b


class ConvTransposeEltwise(nn.Module):

    def __init__(self, eltwise_fn, dim, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvTransposeEltwise, self).__init__()
        self.conv_transpose = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.eltwise = eltwise_fn
        self.kwargs = kwargs

    def forward(self, x):
        a = self.conv_transpose(x)
        b = self.eltwise(a, **self.kwargs)
        return b


class ConvTransposeSumAccumuOnRight(nn.Module):

    def __init__(self, dim, add_func, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvTransposeSumAccumuOnRight, self).__init__()
        self.convtranspose = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.convtranspose1 = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.add_func = add_func
        self.kwargs = kwargs

    def forward(self, x):
        a = self.convtranspose(x)
        b = F.relu(self.convtranspose1(x))
        return self.add_func(a, b, self.kwargs)


class ConvTransposeSumAccumuOnLeft(nn.Module):

    def __init__(self, dim, add_func, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvTransposeSumAccumuOnLeft, self).__init__()
        self.convtranspose = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.convtranspose1 = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.add_func = add_func
        self.kwargs = kwargs

    def forward(self, x):
        a = F.relu(self.convtranspose(x))
        b = self.convtranspose1(x)
        return self.add_func(a, b, self.kwargs)


class ConvTransposeSumBroadcast(nn.Module):

    def __init__(self, dim, add_func, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvTransposeSumBroadcast, self).__init__()
        self.convtranspose = convtranspose_module[dim](in_channels, 1, kernel_size, image_size)
        self.convtranspose1 = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.add_func = add_func
        self.kwargs = kwargs

    def forward(self, x):
        a = F.relu(self.convtranspose(x))
        b = self.convtranspose1(x)
        return self.add_func(a, b, self.kwargs)


class ConvTransposeAddRelu(nn.Module):

    def __init__(self, dim, in_channels, mid_channels, out_channels, kernel_size, inplace, **kwargs):
        super(ConvTransposeAddRelu, self).__init__()
        self.convtranspose = convtranspose_module[dim](in_channels, mid_channels, kernel_size, padding=1, bias=False, **kwargs)
        self.convtranspose1 = convtranspose_module[dim](mid_channels, out_channels, kernel_size, padding=1, bias=False, **kwargs)
        self.convtranspose2 = convtranspose_module[dim](in_channels, out_channels, kernel_size, padding=1, bias=False, **kwargs)
        self.inplace = inplace

    def forward(self, x):
        a = self.convtranspose(x)
        a = F.relu(a, inplace=self.inplace)
        a = self.convtranspose1(a)
        b = self.convtranspose2(x)
        return F.relu(a.add_(b), inplace=self.inplace)


class ConvBatchNorm_Fixed(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvBatchNorm_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](out_channels, eps=0.001)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBatchNorm_Fixed2(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvBatchNorm_Fixed2, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](out_channels, eps=0.001, track_running_stats=False)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBatchNorm_Fixed3(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvBatchNorm_Fixed3, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=True, **kwargs)
        self.bn = bn_module[dim](out_channels, eps=0.001, affine=False)

    def forward(self, x):
        return self.bn(self.conv(x))


class BatchNormConv_Fixed(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(BatchNormConv_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](in_channels, eps=0.001)

    def forward(self, x):
        return self.conv(self.bn(x))


class BatchNorm_Conv_BatchNorm(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(BatchNorm_Conv_BatchNorm, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn1 = bn_module[dim](in_channels, eps=0.001)
        self.bn2 = bn_module[dim](out_channels, eps=0.001)

    def forward(self, x):
        return self.bn2(self.conv(self.bn1(x)))


class ConvReshapeBatchNorm(nn.Module):

    def __init__(self, dim, in_channels, out_channels, dest_shape, **kwargs):
        super(ConvReshapeBatchNorm, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.dest_shape = dest_shape
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](dest_shape[1], eps=0.001)

    def forward(self, x):
        conv_output = self.conv(x)
        return self.bn(torch.reshape(conv_output, self.dest_shape))


class Conv_Conv_Concat(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(Conv_Conv_Concat, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return torch.cat((self.conv1(x), self.conv2(x)))


class ConvRelu_Fixed(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvRelu_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class Conv_Relu_Add(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(Conv_Relu_Add, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return torch.add(F.relu(self.conv1(x), inplace=True), self.conv2(x))


class Conv_Scalar_Binary(nn.Module):

    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Scalar_Binary, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op

    def forward(self, x):
        return self.op(self.conv(x), 2.0)


class Conv_Scalar_Binary_Add(nn.Module):

    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Scalar_Binary_Add, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv1 = conv_module[dim](in_channels, out_channels, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op

    def forward(self, x):
        return torch.add(self.op(self.conv1(x), 2.0), self.op(self.conv2(x), 2.0))


class Conv_Tensor_Binary(nn.Module):

    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Tensor_Binary, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op
        input_size = [1, out_channels, 1, 1]
        if dim == 3:
            input_size.append(1)
        self.tensor = torch.randn(input_size)

    def forward(self, x):
        return self.op(self.conv(x), self.tensor)


class Conv_Tensor_Binary2(nn.Module):

    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Tensor_Binary2, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op
        input_size = [1, out_channels, 1, 1]
        if dim == 3:
            input_size.append(1)
        self.tensor = torch.randn(input_size, dtype=torch.cfloat)

    def forward(self, x):
        return self.op(self.conv(x), self.tensor)


class Conv_Tensor_Binary_Add(nn.Module):

    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Tensor_Binary_Add, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv1 = conv_module[dim](in_channels, out_channels, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op
        input_size = [1, out_channels, 1, 1]
        if dim == 3:
            input_size.append(1)
        self.tensor = torch.randn(input_size)

    def forward(self, x):
        return torch.add(self.op(self.conv1(x), self.tensor), self.op(self.conv2(x), self.tensor))


class ConvReshapeRelu(nn.Module):

    def __init__(self, dim, in_channels, out_channels, dest_shape, **kwargs):
        super(ConvReshapeRelu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.dest_shape = dest_shape
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return F.relu(torch.reshape(self.conv(x), self.dest_shape), inplace=True)


class ConvSum(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvSum, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv(x)
        b = self.conv1(x)
        return a + b


class ConvSum_v2(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvSum_v2, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv(x)
        b = self.conv(x)
        a.add_(b)
        c = self.conv1(x)
        a.add_(c)
        return a


class ConvScalarSum(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvScalarSum, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        b = self.conv(x)
        return b + 2


class ConvBroadcastSum(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvBroadcastSum, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv(x)
        b = self.conv1(x)
        return a[1:2].clone() + b


class ConvReshapeSum(nn.Module):

    def __init__(self, dim, in_channels, out_channels, dest_shape, **kwargs):
        super(ConvReshapeSum, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.dest_shape = dest_shape
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = torch.reshape(self.conv1(x), self.dest_shape)
        b = torch.reshape(self.conv2(x), self.dest_shape)
        return a + b


class CascadedConvBnSumRelu(nn.Module):

    def __init__(self, dim, in_channels, mid_channels, out_channels, **kwargs):
        super(CascadedConvBnSumRelu, self).__init__()
        torch.manual_seed(2018)
        self.conv = conv_module[dim](in_channels, mid_channels, bias=False, **kwargs)
        self.conv1 = conv_module[dim](mid_channels, out_channels, bias=False, padding=1, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](mid_channels, eps=0.001)
        self.bn1 = bn_module[dim](out_channels, eps=0.001)
        self.bn2 = bn_module[dim](out_channels, eps=0.001)

    def forward(self, x):
        a = self.conv(x)
        a = self.bn(a)
        a = F.relu(a, inplace=True)
        a = self.conv1(a)
        a = self.bn1(a)
        b = self.conv2(x)
        b = self.bn2(b)
        return F.relu(a.add_(b), inplace=True)


class Linear_Scalar_Binary(nn.Module):

    def __init__(self, op, in_channels, out_channels, **kwargs):
        super(Linear_Scalar_Binary, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.op = op

    def forward(self, x):
        return self.op(self.linear(x), 2.0)


class Linear_Tensor_Binary(nn.Module):

    def __init__(self, op, in_channels, out_channels, **kwargs):
        super(Linear_Tensor_Binary, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.op = op
        self.tensor = torch.randn(out_channels)

    def forward(self, x):
        return self.op(self.linear(x), self.tensor)


class Linear_Tensor_Binary2(nn.Module):

    def __init__(self, op, in_channels, out_channels, **kwargs):
        super(Linear_Tensor_Binary2, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.op = op
        self.tensor = torch.tensor([2])

    def forward(self, x):
        return self.op(self.linear(x), self.tensor)


class Linear_Tensor_Binary3(nn.Module):

    def __init__(self, op, in_channels, out_channels, **kwargs):
        super(Linear_Tensor_Binary3, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.op = op
        self.tensor = torch.randn(out_channels, dtype=torch.cfloat)

    def forward(self, x):
        return self.op(self.linear(x), self.tensor)


class LinearRelu(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearRelu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return F.relu(self.linear(x), inplace=True)


class LinearSigmoidMul(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearSigmoidMul, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        linear_res = self.linear(x)
        return torch.mul(linear_res, F.sigmoid(linear_res))


class LinearAdd(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearAdd, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.linear1 = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        x1 = x.clone()
        return torch.add(self.linear(x), self.linear1(x1))


class LinearAddRelu(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, inplace, **kwargs):
        super(LinearAddRelu, self).__init__()
        self.linear = nn.Linear(in_channels, mid_channels, bias=False, **kwargs)
        self.linear1 = nn.Linear(mid_channels, out_channels, bias=False, **kwargs)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=False, **kwargs)
        self.inplace = inplace

    def forward(self, x):
        a = self.linear(x)
        a = F.relu(a, inplace=self.inplace)
        a = self.linear1(a)
        b = self.linear2(x)
        return F.relu(a.add_(b), inplace=self.inplace)


class Linear_Reshape_Relu(nn.Module):

    def __init__(self, in_channels, out_channels, dest_shape, **kwargs):
        super(Linear_Reshape_Relu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.dest_shape = dest_shape

    def forward(self, x):
        return F.relu(torch.reshape(self.linear(x), self.dest_shape), inplace=True)


class LinearBn(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(LinearBn, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.bn = bn_module[dim](1, eps=0.001)

    def forward(self, x):
        return self.bn(self.linear(x))


class Linear_Reshape_Bn(nn.Module):

    def __init__(self, dim, in_channels, out_channels, dest_shape, **kwargs):
        super(Linear_Reshape_Bn, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.bn = bn_module[dim](1, eps=0.001)
        self.dest_shape = dest_shape

    def forward(self, x):
        return self.bn(torch.reshape(self.linear(x), self.dest_shape))


class ConvSumInDiffBlock(nn.Module):

    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvSumInDiffBlock, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.pad = (0, 0) * dim
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        y = self.conv(x)
        if y.size(1) != x.size(1):
            z = F.pad(x, self.pad + (0, y.size(1) - x.size(1)), 'constant', 0.0)
            y += z
        else:
            y += x
        return y


class ConvSwishOutplace(nn.Module):

    def __init__(self, dim, in_channels, out_channels, kernel_size, image_size):
        super(ConvSwishOutplace, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a1 = self.conv(x)
        b1 = torch.sigmoid(a1)
        c1 = torch.mul(a1, b1)
        return c1


class ConvSwishInplace(nn.Module):

    def __init__(self, dim, in_channels, out_channels, kernel_size, image_size):
        super(ConvSwishInplace, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a = self.conv(x)
        b = torch.sigmoid(a)
        res = a.mul_(b)
        return res


class ConvSwishOutplaceSumOutplace(nn.Module):

    def __init__(self, dim, in_channels, out_channels, kernel_size, image_size):
        super(ConvSwishOutplaceSumOutplace, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.conv1 = conv_module[dim](in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a1 = self.conv(x)
        b1 = torch.sigmoid(a1)
        c1 = torch.mul(a1, b1)
        a2 = self.conv1(x)
        b2 = torch.sigmoid(a2)
        c2 = torch.mul(a2, b2)
        return c1 + c2


class ConvSwishInplaceSumInplace(nn.Module):

    def __init__(self, dim, in_channels, out_channels, kernel_size, image_size):
        super(ConvSwishInplaceSumInplace, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.conv1 = conv_module[dim](in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a1 = self.conv(x)
        b1 = torch.sigmoid(a1)
        c1 = a1.mul_(b1)
        a2 = self.conv1(x)
        b2 = torch.sigmoid(a2)
        c2 = a2.mul_(b2)
        return c1.add_(c2)


class ConvTransposeSigmoidMul(nn.Module):

    def __init__(self, mul, dim, in_channels, out_channels, kernel_size, image_size):
        super(ConvTransposeSigmoidMul, self).__init__()
        self.conv_transpose = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.mul_op = mul

    def forward(self, x):
        a1 = self.conv_transpose(x)
        b1 = torch.sigmoid(a1)
        c1 = self.mul_op(a1, b1)
        return c1


class ChannelShuffle_with_Static_Shape(nn.Module):

    def __init__(self, batchsize, num_channels, height, width, groups):
        super(ChannelShuffle_with_Static_Shape, self).__init__()
        self.batchsize = batchsize
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.groups = groups

    def forward(self, x):
        channels_per_group = self.num_channels // self.groups
        x = x.view(self.batchsize, self.groups, channels_per_group, self.height, self.width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(self.batchsize, -1, self.height, self.width)
        return x


class ChannelShuffle_with_Dynamic_Shape(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle_with_Dynamic_Shape, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class NotChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(NotChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, width, height)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, width, height)
        return x


class MatmulDivOutplaceOutModifiedByOtherOP_v1(nn.Module):

    def __init__(self, div_scalar=False, with_out=True):
        super(MatmulDivOutplaceOutModifiedByOtherOP_v1, self).__init__()
        self.div_scalar = div_scalar
        self.with_out = with_out

    def forward(self, x):
        y = torch.transpose(x, -1, -2).contiguous()
        mm_res_shape = x.size()[:-1] + y.size()[-1:]
        mm_res = torch.randn(mm_res_shape, dtype=x.dtype)
        mm_out = torch.empty(mm_res_shape, dtype=x.dtype)
        mm_res = torch.matmul(x, y, out=mm_out)
        if self.div_scalar:
            div_res = mm_res.div(2.0)
        else:
            div_res = mm_res.div(torch.ones(mm_res_shape, dtype=x.dtype) + 1)
        mm_out.add_(5)
        return div_res


class MatmulDivOutplaceOutModifiedByOtherOP_v2(nn.Module):

    def __init__(self, div_scalar=False, with_out=True):
        super(MatmulDivOutplaceOutModifiedByOtherOP_v2, self).__init__()
        self.div_scalar = div_scalar
        self.with_out = with_out

    def forward(self, x):
        y = torch.transpose(x, -1, -2).contiguous()
        mm_res_shape = x.size()[:-1] + y.size()[-1:]
        mm_res = torch.randn(mm_res_shape, dtype=x.dtype)
        mm_out = torch.empty(mm_res_shape, dtype=x.dtype)
        mm_res = torch.matmul(x, y, out=mm_out)
        if self.div_scalar:
            div_res = mm_res.div(2.0)
        else:
            div_res = mm_res.div(torch.ones(mm_res_shape, dtype=x.dtype) + 1)
        mm_out.add_(5)
        div_out_equal = mm_out == div_res
        return div_res + div_out_equal


class MatmulDivOutplace(nn.Module):

    def __init__(self, div_scalar=False, with_out=False):
        super(MatmulDivOutplace, self).__init__()
        self.div_scalar = div_scalar
        self.with_out = with_out

    def forward(self, x):
        mm_res = None
        y = torch.transpose(x, -1, -2).contiguous()
        mm_res_shape = x.size()[:-1] + y.size()[-1:]
        if self.with_out:
            mm_res = torch.randn(mm_res_shape, dtype=x.dtype)
            torch.matmul(x, y, out=mm_res)
        else:
            mm_res = torch.matmul(x, y)
        if self.div_scalar:
            return mm_res.div(2.0)
        else:
            return mm_res.div(torch.ones(mm_res_shape, dtype=x.dtype) + 1)


class MatmulDivInplace(nn.Module):

    def __init__(self, div_scalar=False, with_out=False):
        super(MatmulDivInplace, self).__init__()
        self.div_scalar = div_scalar
        self.with_out = with_out

    def forward(self, x):
        mm_res = None
        y = torch.transpose(x, -1, -2).contiguous()
        mm_res_shape = x.size()[:-1] + y.size()[-1:]
        if self.with_out:
            mm_res = torch.randn(mm_res_shape, dtype=x.dtype)
            torch.matmul(x, y, out=mm_res)
        else:
            mm_res = torch.matmul(x, y)
        if self.div_scalar:
            return mm_res.div_(2.0)
        else:
            return mm_res.div_(torch.ones(mm_res_shape, dtype=x.dtype) + 1)


class MatmulMul(nn.Module):

    def __init__(self, mul_scalar=False, with_out=False):
        super(MatmulMul, self).__init__()
        self.with_out = with_out
        self.mul_scalar = mul_scalar

    def forward(self, x):
        mm_res = None
        y = torch.transpose(x, -1, -2).contiguous()
        mm_res_shape = x.size()[:-1] + y.size()[-1:]
        if not self.mul_scalar:
            x = x * (torch.ones([1], dtype=x.dtype) + 1)
        if self.with_out:
            mm_res = torch.randn(mm_res_shape, dtype=x.dtype)
            mm_res = torch.matmul(x, y, out=mm_res)
        else:
            mm_res = torch.matmul(x, y)
        if self.mul_scalar:
            mm_res = mm_res * 0.125
        else:
            mm_res = mm_res * (torch.ones([1], dtype=x.dtype) + 1)
        return mm_res


class TransposedMatmulDiv(nn.Module):

    def __init__(self):
        super(TransposedMatmulDiv, self).__init__()

    def forward(self, batch1, batch2):
        bmm_res = torch.matmul(batch1, batch2)
        res = bmm_res * 0.3
        return res


class BmmAdd(nn.Module):

    def __init__(self):
        super(BmmAdd, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        res = torch.add(bmm_res, input)
        return res


class MHAScoresCalculation(nn.Module):

    def __init__(self, dim_per_head, softmax_dim=-1):
        super(MHAScoresCalculation, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.dim_per_head = dim_per_head

    def forward(self, mat1, mat2, bias):
        mat1 = mat1 / math.sqrt(self.dim_per_head)
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        scores = qk + bias
        return self.softmax(scores)


class MHAScoresCalculation_v2(nn.Module):

    def __init__(self, dim_per_head, softmax_dim=-1):
        super(MHAScoresCalculation_v2, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.scale = 1 / math.sqrt(dim_per_head)

    def forward(self, mat1, mat2, bias):
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        qk = qk * self.scale
        scores = qk + bias
        return self.softmax(scores)


class MHAScoresCalculation_v3(nn.Module):

    def __init__(self, dim_per_head, softmax_dim=-1):
        super(MHAScoresCalculation_v3, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.scale = 1 / math.sqrt(dim_per_head)

    def forward(self, mat1, mat2, bias):
        mat1 = mat1 * self.scale
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        scores = qk + bias
        return self.softmax(scores)


class MHAScoresCalculation_v1(nn.Module):

    def __init__(self, dim_per_head, softmax_dim=-1):
        super(MHAScoresCalculation_v1, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.dim_per_head = dim_per_head

    def forward(self, mat1, mat2, bias):
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        qk = qk / math.sqrt(self.dim_per_head)
        scores = qk + bias
        return self.softmax(scores)


class DistilMHAScoresCalculation_v1(nn.Module):

    def __init__(self, dim_per_head, fill_value, softmax_dim=-1):
        super(DistilMHAScoresCalculation_v1, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.dim_per_head = dim_per_head
        self.fill = fill_value

    def forward(self, mat1, mat2, mask):
        mask_shape = [mat1.shape[0], 1, 1, mat1.shape[3]]
        mat1 = mat1 / math.sqrt(self.dim_per_head)
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk.masked_fill_(mask, self.fill)
        return self.softmax(qk)


class DistilMHAScoresCalculation_v2(nn.Module):

    def __init__(self, dim_per_head, fill_value, softmax_dim=-1):
        super(DistilMHAScoresCalculation_v2, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.dim_per_head = dim_per_head
        self.fill = fill_value

    def forward(self, mat1, mat2, mask):
        mask_shape = [mat1.shape[0], 1, 1, mat1.shape[3]]
        mat1 = mat1 / math.sqrt(self.dim_per_head)
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk.masked_fill_(mask, self.fill)
        return self.softmax(qk)


class VitMHAScoresCalculation_v1(nn.Module):

    def __init__(self, dim_per_head):
        super(VitMHAScoresCalculation_v1, self).__init__()
        self.scale = dim_per_head ** -0.5

    def forward(self, mat1, mat2, mask):
        qk = torch.matmul(mat1, mat2.transpose(-1, 2)) * self.scale
        mask_value = -torch.finfo(qk.dtype).max
        qk = qk.masked_fill(mask, mask_value)
        return nn.functional.softmax(qk, dim=-1)


class VitMHAScoresCalculation_v2(nn.Module):

    def __init__(self, dim_per_head):
        super(VitMHAScoresCalculation_v2, self).__init__()
        self.scale = dim_per_head ** -0.5

    def forward(self, mat1, mat2, mask):
        q = mat1 * self.scale
        qk = torch.matmul(q, mat2.transpose(-1, 2))
        mask_value = -torch.finfo(qk.dtype).max
        qk = qk.masked_fill(mask, mask_value)
        return nn.functional.softmax(qk, dim=-1)


class Maskedfill__softmax(nn.Module):

    def __init__(self, fill_value, softmax_dim=-1):
        super(Maskedfill__softmax, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.fill = fill_value

    def forward(self, qk, mask):
        mask_shape = [qk.shape[0], 1, 1, qk.shape[3]]
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk.masked_fill_(mask, self.fill)
        return self.softmax(qk)


class Maskedfill_softmax(nn.Module):

    def __init__(self, fill_value):
        super(Maskedfill_softmax, self).__init__()
        self.fill = fill_value

    def forward(self, qk, mask):
        mask_shape = [qk.shape[0], 1, 1, qk.shape[3]]
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk = qk.masked_fill(mask, self.fill)
        return nn.functional.softmax(qk, dim=-1)


class AtenSoftmaxRepalce(nn.Module):

    def __init__(self, dim=-1):
        super(AtenSoftmaxRepalce, self).__init__()
        self.softmax = torch.nn.Softmax(dim)

    def forward(self, x):
        return self.softmax(x)


class AtenBatchNormRepalce(nn.Module):

    def __init__(self):
        super(AtenBatchNormRepalce, self).__init__()
        self.bn = torch.nn.BatchNorm2d(10)

    def forward(self, x):
        return self.bn(x)


class AddLayerNorm(torch.nn.Module):

    def __init__(self, dim=32):
        super(AddLayerNorm, self).__init__()
        self.layernorm = torch.nn.LayerNorm(dim)

    def forward(self, x, y):
        z = torch.add(x, y)
        return self.layernorm(z)


class AddLayerNorm_v1(torch.nn.Module):

    def __init__(self, dim=32):
        super(AddLayerNorm_v1, self).__init__()
        self.layernorm = torch.nn.LayerNorm(dim)

    def forward(self, x, y, z):
        x = x + y + z
        return self.layernorm(x)


class AddLayerNorm_v2(torch.nn.Module):

    def __init__(self, dim=32):
        super(AddLayerNorm_v2, self).__init__()
        self.dim = dim

    def forward(self, x, y, w):
        z = torch.add(x, y)
        return torch.nn.functional.layer_norm(z, [self.dim], weight=w)


class ConcatBnRelu(torch.nn.Module):

    def __init__(self, dim, cat_dim, in_channels, **kwargs):
        super(ConcatBnRelu, self).__init__()
        self.bn = bn_module[dim](in_channels)
        self.relu = torch.nn.ReLU()
        self.cat_dim = cat_dim

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=self.cat_dim)
        x = self.bn(x)
        return self.relu(x)


class ConcatBnReluV2(torch.nn.Module):

    def __init__(self, dim, cat_dim, in_channels, **kwargs):
        super(ConcatBnReluV2, self).__init__()
        self.bn = bn_module[dim](in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.cat_dim = cat_dim

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=self.cat_dim)
        x = self.bn(x)
        return self.relu(x)


class ConcatBnReluV3(torch.nn.Module):

    def __init__(self, dim, cat_dim, in_channels, **kwargs):
        super(ConcatBnReluV3, self).__init__()
        self.bn = bn_module[dim](in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.cat_dim = cat_dim

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=self.cat_dim)
        x = self.bn(x)
        y = self.relu(x)
        x += 2
        return y + x


class ModMultLinear(nn.Module):

    def __init__(self, w1_dim, w2_dim):
        super(ModMultLinear, self).__init__()
        self.linear1 = nn.Linear(5, w1_dim)
        self.linear2 = nn.Linear(5, w2_dim)
        self.linear3 = nn.Linear(w1_dim, 5)
        self.linear4 = nn.Linear(w1_dim, 5)

    def forward(self, x):
        res1 = self.linear1(x)
        res2 = self.linear2(x)
        res3 = self.linear3(res1)
        res4 = self.linear4(res1)
        return res1, res2, res3, res4


class ModMultLinearWithOrWithoutBias(nn.Module):

    def __init__(self):
        super(ModMultLinearWithOrWithoutBias, self).__init__()
        self.linear1 = nn.Linear(10, 32, bias=False)
        self.linear2 = nn.Linear(10, 32, bias=True)
        self.linear3 = nn.Linear(10, 32, bias=True)
        self.linear4 = nn.Linear(10, 32, bias=False)

    def forward(self, x):
        res1 = self.linear1(x)
        res2 = self.linear2(x)
        res3 = self.linear3(x)
        res4 = self.linear4(x)
        return res1, res2, res3, res4


class LinearSwishNaive(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(LinearSwishNaive, self).__init__()
        self.linear = nn.Linear(in_feature, out_feature)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        linear_out = self.linear(input)
        sigmoid_out = self.sigmoid(linear_out)
        return torch.mul(linear_out, sigmoid_out)


class Bottleneck_v1(nn.Module):

    def __init__(self):
        super(Bottleneck_v1, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.downsample = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        y1 = self.conv1(x).relu_()
        y2 = self.conv2(y1).relu_()
        y3 = self.conv3(y2)
        y3 += self.downsample(x)
        return y3.relu_()


class Bottleneck_v2(nn.Module):

    def __init__(self):
        super(Bottleneck_v2, self).__init__()
        self.conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        x = self.conv(x)
        y1 = self.conv1(x).relu_()
        y2 = self.conv2(y1).relu_()
        y3 = self.conv3(y2)
        y3 += x
        return y3.relu_()


class EinsumAdd(nn.Module):

    def __init__(self, equation):
        super(EinsumAdd, self).__init__()
        self.equation = equation

    def forward(self, input1, input2, bias):
        return torch.einsum(self.equation, input1, input2) + bias


class EinsumAddScalar(nn.Module):

    def __init__(self, equation):
        super(EinsumAddScalar, self).__init__()
        self.equation = equation

    def forward(self, input1, input2):
        return torch.einsum(self.equation, input1, input2) + 12.0


class EinsumAddInplace(nn.Module):

    def __init__(self, equation):
        super(EinsumAddInplace, self).__init__()
        self.equation = equation

    def forward(self, input1, input2, bias):
        return torch.einsum(self.equation, input1, input2).add_(bias)


class EinsumAddInplaceV1(nn.Module):

    def __init__(self, equation):
        super(EinsumAddInplaceV1, self).__init__()
        self.equation = equation

    def forward(self, input1, input2, bias):
        return bias.add_(torch.einsum(self.equation, input1, input2))


class AddMulDiv(nn.Module):

    def __init__(self):
        super(AddMulDiv, self).__init__()

    def forward(self, input):
        return torch.div(torch.mul(input, torch.add(input, 3)), 6)


class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = torch.nn.ModuleList()
        self.mlp.append(torch.nn.Linear(10, 10))
        self.mlp.append(torch.nn.ReLU())
        self.mlp.append(torch.nn.Linear(10, 10))
        self.mlp.append(torch.nn.Sigmoid())

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class Conv2d(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.conv(x)


class MatmulDiv(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.transpose(x, -1, -2).contiguous()
        z = torch.matmul(x, y)
        return z.div(2.0)


class MHA_Model_BERT(nn.Module):

    def __init__(self, scale, num_heads, head_dims, permute_idx, trans_a, trans_b):
        super(MHA_Model_BERT, self).__init__()
        self.scale = scale
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.embed_dims = self.num_heads * self.head_dims
        self.query = nn.Linear(self.embed_dims, self.embed_dims, bias=True)
        self.key = nn.Linear(self.embed_dims, self.embed_dims, bias=True)
        self.value = nn.Linear(self.embed_dims, self.embed_dims, bias=True)
        self.permute_idx = permute_idx
        self.trans_a = trans_a
        self.trans_b = trans_b

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dims)
        x = x.view(new_x_shape)
        return x.permute(self.permute_idx)

    def forward(self, x, mask):
        query_layer = self.transpose_for_scores(self.query(x))
        key_layer = self.transpose_for_scores(self.key(x)).transpose(self.trans_a, self.trans_b)
        value_layer = self.transpose_for_scores(self.value(x))
        attention_scores = torch.matmul(query_layer, key_layer) / self.scale + mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(self.permute_idx).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dims,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class MHA_Model_Distil(nn.Module):

    def __init__(self, scale, num_heads, head_dims, trans_a, trans_b, trans_c, fill_value=-float('inf')):
        super(MHA_Model_Distil, self).__init__()
        self.scale = scale
        self.n_head = num_heads
        self.head_dims = head_dims
        self.dim = self.n_head * self.head_dims
        self.q_lin = nn.Linear(self.dim, self.dim, bias=True)
        self.k_lin = nn.Linear(self.dim, self.dim, bias=True)
        self.v_lin = nn.Linear(self.dim, self.dim, bias=True)
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.fill_value = fill_value

    def forward(self, x, mask):
        bs, q_length, dim = x.size()
        k_length = x.size(1)

        def shape(x: torch.Tensor) ->torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_head, self.head_dims).transpose(self.trans_a, self.trans_b)

        def unshape(x: torch.Tensor) ->torch.Tensor:
            """group heads"""
            return x.transpose(self.trans_a, self.trans_b).contiguous().view(bs, -1, self.n_head * self.head_dims)
        q = shape(self.q_lin(x))
        k = shape(self.k_lin(x))
        v = shape(self.v_lin(x))
        mask_reshp = bs, 1, 1, k_length
        q = q / self.scale
        scores = torch.matmul(q, k.transpose(self.trans_b, self.trans_c))
        mask = (mask == 0).view(mask_reshp).expand_as(scores)
        scores = scores.masked_fill(mask, self.fill_value)
        weights = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context_layer = unshape(context)
        return context_layer


class MHA_Model_ViT(nn.Module):

    def __init__(self, scale, num_heads, head_dims, permute_idx, trans_a, trans_b, select_a, select_b):
        super(MHA_Model_ViT, self).__init__()
        self.scale = 1.0 / scale
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.embed_dims = self.num_heads * self.head_dims
        self.qkv = nn.Linear(self.embed_dims, self.embed_dims * 3, bias=True)
        self.permute_idx = permute_idx
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.select_a = select_a
        self.select_b = select_b

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims).permute(self.permute_idx)
        q, k, v = qkv[0], qkv[self.select_a], qkv[self.select_b]
        attn = q @ k.transpose(self.trans_a, self.trans_b) * self.scale
        attn = attn.softmax(dim=-1)
        context_layer = (attn @ v).transpose(self.select_a, self.select_b).reshape(B, N, self.embed_dims)
        return context_layer


class SimpleNet(torch.nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y


class SimpleNet_v2(torch.nn.Module):

    def __init__(self):
        super(SimpleNet_v2, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.conv2(x1)
        y = torch.flatten(x1, start_dim=1)
        return y


class TestInputOutputModule(torch.nn.Module):

    def __init__(self):
        super(TestInputOutputModule, self).__init__()

    def forward(self, *args, **kwargs):
        return args


class TestInputOutputModule2(torch.nn.Module):

    def __init__(self):
        super(TestInputOutputModule2, self).__init__()

    def forward(self, param1):
        return param1


class softmax_with_multiuse_input(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = nn.Softmax(dim=-1)(x)
        x2 = x + x1
        return x1, x2


class softmax_with_alias_input(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = x
        x2 = nn.Softmax(dim=-1)(x)
        return x1, x2


class inplace_softmax_with_blocks(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, flag):
        if flag:
            x1 = x + 1
        else:
            x1 = x + 3
        x2 = torch.softmax(x1, dim=-1)
        return x2


class softmax_MHA(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        attention_scores = torch.matmul(x, x.transpose(-1, -2))
        attention_scores = attention_scores / 64
        attention_scores = attention_scores + x
        attention_scores = nn.Softmax(dim=-1)(attention_scores)
        return attention_scores


class inplace_softmax_with_TE_group(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = x + 1
        x2 = x + 2
        x3 = x + 3
        x4 = x + 4
        x5 = x + 5
        y1 = (x1 / x2).softmax(dim=-1)
        y2 = ((x4 - x3) / x5).softmax(dim=-1)
        return y1, y2


class softmax_dtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.softmax(x, dtype=x.dtype, dim=1)


class inplace_softmax_dtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = x + 1
        return torch.nn.functional.softmax(x1, dtype=x.dtype, dim=1)


class IPEXConvAdd(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvAdd, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return a.add_(b)


class IPEXConvAddRelu(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvAddRelu, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = F.relu(self.conv1(x))
        b = self.conv2(x)
        return F.relu(a.add_(b), inplace=True)


class IPEXConvConvRelu(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvConvRelu, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        return F.relu(res, inplace=True)


class IPEXConvSigmoidMul(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvSigmoidMul, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv(x)
        b = torch.sigmoid(a)
        return a.mul_(b)


class IPEXLinearAdd(nn.Module):

    def __init__(self, in_channels, out_channels, bias):
        super(IPEXLinearAdd, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        a = self.linear1(x)
        b = self.linear2(x)
        return a.add_(b)


class IPEXLinearAddRelu(nn.Module):

    def __init__(self, in_channels, out_channels, bias):
        super(IPEXLinearAddRelu, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        a = F.relu(self.linear(x))
        b = self.linear(x)
        return F.relu(a.add_(b), inplace=True)


class IPEXLinearSigmoidMul(nn.Module):

    def __init__(self, in_channels, out_channels, bias):
        super(IPEXLinearSigmoidMul, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        a = self.linear(x)
        b = torch.sigmoid(a)
        return a.mul_(b)


class IPEXMatmulDiv(nn.Module):

    def __init__(self):
        super(IPEXMatmulDiv, self).__init__()
        seed = 2018
        torch.manual_seed(seed)

    def forward(self, x1, x2, x3):
        return torch.matmul(x1, x2) / x3 + x3


class TransFree_FP32_Bmm(nn.Module):

    def __init__(self):
        super(TransFree_FP32_Bmm, self).__init__()

    def forward(self, x1, y1):
        out = torch.matmul(x1, y1)
        return out


class OutTransFree_FP32_Bmm_v1(nn.Module):

    def __init__(self):
        super(OutTransFree_FP32_Bmm_v1, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.transpose(1, 2)
        return out


class OutTransFree_FP32_Bmm_v2(nn.Module):

    def __init__(self):
        super(OutTransFree_FP32_Bmm_v2, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.permute(0, 2, 1, 3)
        return out


class OutTransFree_FP32_Bmm_v3(nn.Module):

    def __init__(self):
        super(OutTransFree_FP32_Bmm_v3, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.transpose(1, 3)
        return out


class OutTransFree_FP32_Bmm_v4(nn.Module):

    def __init__(self):
        super(OutTransFree_FP32_Bmm_v4, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.permute(0, 1, 3, 2)
        return out


class TransFree_BF16_Bmm(nn.Module):

    def __init__(self):
        super(TransFree_BF16_Bmm, self).__init__()

    def forward(self, x1, y1):
        out = torch.matmul(x1, y1)
        return out


class OutTransFree_BF16_Bmm_v1(nn.Module):

    def __init__(self):
        super(OutTransFree_BF16_Bmm_v1, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.transpose(1, 2)
        return out


class OutTransFree_BF16_Bmm_v2(nn.Module):

    def __init__(self):
        super(OutTransFree_BF16_Bmm_v2, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.permute(0, 2, 1, 3)
        return out


class OutTransFree_BF16_Bmm_v3(nn.Module):

    def __init__(self):
        super(OutTransFree_BF16_Bmm_v3, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.transpose(1, 3)
        return out


class OutTransFree_BF16_Bmm_v4(nn.Module):

    def __init__(self):
        super(OutTransFree_BF16_Bmm_v4, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.permute(0, 1, 3, 2)
        return out


class Module(torch.nn.Module):

    def __init__(self):
        super(Module, self).__init__()
        self.conv = torch.nn.Conv2d(1, 10, 5, 1)

    def forward(self, x):
        y = self.conv(x)
        return y


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddLayerNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 32, 32]), torch.rand([4, 4, 32, 32])], {}),
     True),
    (AddLayerNorm_v1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 32, 32]), torch.rand([4, 4, 32, 32]), torch.rand([4, 4, 32, 32])], {}),
     True),
    (AddMulDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AtenSoftmaxRepalce,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNormConv_Fixed,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm_Conv_BatchNorm,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BmmAdd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (Bottleneck_v1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (Bottleneck_v2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ChannelShuffle_with_Dynamic_Shape,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelShuffle_with_Static_Shape,
     lambda: ([], {'batchsize': 4, 'num_channels': 4, 'height': 4, 'width': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ConvBatchNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ConvBatchNormSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ConvBatchNorm_Fixed,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBatchNorm_Fixed2,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBatchNorm_Fixed3,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBroadcastSum,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvEltwise,
     lambda: ([], {'eltwise_fn': _mock_layer(), 'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'image_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvRelu_Fixed,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvScalarSum,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvSum,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvSum_v2,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvSwishInplace,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'image_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvSwishInplaceSumInplace,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'image_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvSwishOutplace,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'image_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvSwishOutplaceSumOutplace,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'image_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvTranspose,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvTranspose2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 4, 4])], {}),
     True),
    (ConvTransposeEltwise,
     lambda: ([], {'eltwise_fn': _mock_layer(), 'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'image_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv_Bn_Relu,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv_Conv_Concat,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv_IF_Relu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 64, 64])], {}),
     True),
    (Conv_Relu_Add,
     lambda: ([], {'dim': 2, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DistilMHAScoresCalculation_v1,
     lambda: ([], {'dim_per_head': 4, 'fill_value': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 1, 1, 4])], {}),
     True),
    (DistilMHAScoresCalculation_v2,
     lambda: ([], {'dim_per_head': 4, 'fill_value': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 1, 1, 4])], {}),
     True),
    (IPEXConvAdd,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IPEXConvAddRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IPEXConvConvRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (IPEXConvSigmoidMul,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IPEXLinearAdd,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IPEXLinearAddRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IPEXLinearSigmoidMul,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IPEXMatmulDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearAdd,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearAddRelu,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4, 'inplace': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearEltwise,
     lambda: ([], {'eltwise_fn': _mock_layer(), 'in_channels': 4, 'out_channels': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearSigmoidMul,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearSwishNaive,
     lambda: ([], {'in_feature': 4, 'out_feature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MHAScoresCalculation,
     lambda: ([], {'dim_per_head': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MHAScoresCalculation_v1,
     lambda: ([], {'dim_per_head': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MHAScoresCalculation_v2,
     lambda: ([], {'dim_per_head': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MHAScoresCalculation_v3,
     lambda: ([], {'dim_per_head': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Maskedfill__softmax,
     lambda: ([], {'fill_value': 4}),
     lambda: ([torch.rand([4, 1, 1, 4]), torch.rand([4, 1, 1, 4])], {}),
     True),
    (Maskedfill_softmax,
     lambda: ([], {'fill_value': 4}),
     lambda: ([torch.rand([4, 1, 1, 4]), torch.rand([4, 1, 1, 4])], {}),
     True),
    (MatmulDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MatmulDivInplace,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MatmulDivOutplace,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MatmulDivOutplaceOutModifiedByOtherOP_v1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MatmulDivOutplaceOutModifiedByOtherOP_v2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MatmulMul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Model,
     lambda: ([], {'ic': 4, 'oc': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Module,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NotChannelShuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutTransFree_BF16_Bmm_v1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutTransFree_BF16_Bmm_v2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutTransFree_BF16_Bmm_v3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutTransFree_BF16_Bmm_v4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutTransFree_FP32_Bmm_v1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutTransFree_FP32_Bmm_v2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutTransFree_FP32_Bmm_v3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutTransFree_FP32_Bmm_v4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (SimpleNet_v2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestInputOutputModule,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (TestInputOutputModule2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TestModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (TransFree_BF16_Bmm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransFree_FP32_Bmm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransposedMatmulDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (inplace_softmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (inplace_softmax_dtype,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (inplace_softmax_with_TE_group,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (inplace_softmax_with_blocks,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     True),
    (softmax_MHA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (softmax_dtype,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (softmax_with_alias_input,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (softmax_with_multiuse_input,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_intel_intel_extension_for_pytorch(_paritybench_base):
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

    def test_059(self):
        self._check(*TESTCASES[59])

    def test_060(self):
        self._check(*TESTCASES[60])

    def test_061(self):
        self._check(*TESTCASES[61])

    def test_062(self):
        self._check(*TESTCASES[62])

    def test_063(self):
        self._check(*TESTCASES[63])

    def test_064(self):
        self._check(*TESTCASES[64])

    def test_065(self):
        self._check(*TESTCASES[65])

    def test_066(self):
        self._check(*TESTCASES[66])

    def test_067(self):
        self._check(*TESTCASES[67])

    def test_068(self):
        self._check(*TESTCASES[68])

    def test_069(self):
        self._check(*TESTCASES[69])

    def test_070(self):
        self._check(*TESTCASES[70])

    def test_071(self):
        self._check(*TESTCASES[71])

    def test_072(self):
        self._check(*TESTCASES[72])

    def test_073(self):
        self._check(*TESTCASES[73])

    def test_074(self):
        self._check(*TESTCASES[74])

    def test_075(self):
        self._check(*TESTCASES[75])

    def test_076(self):
        self._check(*TESTCASES[76])

    def test_077(self):
        self._check(*TESTCASES[77])

    def test_078(self):
        self._check(*TESTCASES[78])

    def test_079(self):
        self._check(*TESTCASES[79])

    def test_080(self):
        self._check(*TESTCASES[80])

    def test_081(self):
        self._check(*TESTCASES[81])

    def test_082(self):
        self._check(*TESTCASES[82])

    def test_083(self):
        self._check(*TESTCASES[83])

    def test_084(self):
        self._check(*TESTCASES[84])

    def test_085(self):
        self._check(*TESTCASES[85])

    def test_086(self):
        self._check(*TESTCASES[86])

    def test_087(self):
        self._check(*TESTCASES[87])

    def test_088(self):
        self._check(*TESTCASES[88])

    def test_089(self):
        self._check(*TESTCASES[89])

    def test_090(self):
        self._check(*TESTCASES[90])

