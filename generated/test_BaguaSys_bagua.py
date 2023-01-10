import sys
_module = sys.modules[__name__]
del sys
bagua = _module
bagua_define = _module
distributed = _module
launch = _module
run = _module
script = _module
baguarun = _module
service = _module
autotune_service = _module
autotune_system = _module
autotune_task_manager = _module
bayesian_optimizer = _module
torch_api = _module
algorithms = _module
async_model_average = _module
base = _module
bytegrad = _module
decentralized = _module
gradient_allreduce = _module
q_adam = _module
bucket = _module
checkpoint = _module
checkpointing = _module
communication = _module
contrib = _module
cache_loader = _module
cached_dataset = _module
fuse = _module
optimizer = _module
load_balancing_data_loader = _module
sync_batchnorm = _module
utils = _module
redis_store = _module
store = _module
data_parallel = _module
bagua_distributed = _module
distributed = _module
functional = _module
distributed = _module
env = _module
model_parallel = _module
moe = _module
experts = _module
layer = _module
sharded_moe = _module
utils = _module
tensor = _module
utils = _module
version = _module
bagua_core = _module
_environment = _module
bagua_install_deps = _module
conf = _module
synthetic_benchmark = _module
main = _module
main = _module
main = _module
main = _module
mnist_main = _module
main = _module
setup = _module
tests = _module
test_communicator = _module
test_cached_dataset = _module
test_fused_optimizer = _module
test_load_balancing_data_loader = _module
test_store = _module
test_sync_bn = _module
internal = _module
common_utils = _module
compressor = _module
multi_process = _module
_core = _module
common_cuda = _module
common_distributed = _module
common_utils = _module
expecttest = _module
pytorch_lightning = _module
boring_model = _module
test_bagua_strategy = _module
test_autotune_service = _module
test_bayesian_optimizer = _module
test_bagua_define = _module
test_async_model_average = _module
test_bagua_ddp = _module
test_bagua_module = _module
test_broadcast_state = _module
test_c10d_common = _module
test_decentralized = _module
test_gradient_allreduce = _module
test_low_precision_decentralized = _module
test_multi_models = _module
test_async_model_average = _module
test_bagua_module = _module
test_broadcast_state = _module
test_decentralized = _module
test_gradient_allreduce = _module
test_low_precision_decentralized = _module
test_multi_models = _module
test_process_group = _module
test_qadam = _module
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


import time


from typing import Optional


from typing import IO


from typing import List


from typing import Any


import logging


import uuid


from typing import Callable


from typing import Tuple


from typing import Union


import torch


from torch.distributed.argparse_util import check_env


from torch.distributed.argparse_util import env


from torch.distributed.elastic.multiprocessing import Std


from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config


from torch.distributed.elastic.utils import macros


from torch.distributed.elastic.utils.logging import get_logger


from torch.distributed.launcher.api import LaunchConfig


from torch.distributed.launcher.api import elastic_launch


from enum import IntEnum


from typing import Dict


from torch.optim.optimizer import Optimizer


import math


import re


import torch.distributed as dist


from collections import defaultdict


from functools import lru_cache


import torch.distributed.distributed_c10d as c10d


from torch.distributed import ProcessGroup as TorchProcessGroup


from torch.utils.data.dataset import Dataset


import copy


import itertools


from torch.utils.data.sampler import Sampler


from typing import Iterator


from collections import OrderedDict


from torch.autograd.function import Function


import torch.nn.functional as F


from torch.nn.modules.batchnorm import _BatchNorm


import collections


from torch.nn.modules import Module


from torch.nn.parallel import DistributedDataParallel as TorchDistributedDataParallel


import warnings


import enum


from torch.autograd import Function


import torch.nn


import typing


from torch import Tensor


from torch.nn import Module


from typing import TYPE_CHECKING


import numpy as np


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data.distributed


from torchvision import models


import torch.nn as nn


from torchvision import datasets


from torchvision import transforms


from torch.optim.lr_scheduler import StepLR


import random


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import TensorDataset


from typing import cast


import functools


import torch.cuda


import inspect


from enum import Enum


import types


from typing import NamedTuple


from functools import wraps


import torch.distributed as c10d


import torch.cuda.nccl


from functools import partial


from functools import reduce


from collections.abc import Sequence


from itertools import product


from copy import deepcopy


from numbers import Number


from typing import Iterable


from torch._six import string_classes


import torch.backends.cudnn


import torch.backends.mkl


import string


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


from torch.utils.data import Subset


from torch import nn


import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD


import torch.multiprocessing as mp


class comm(object):
    WORLD = object()


class CommMember(object):
    WORLD = comm.WORLD
    NON_COMM_MEMBER = object()


_default_pg = None


def is_initialized():
    """
    Checking if the default process group has been initialized.

    """
    return _default_pg is not None


def _get_default_group():
    """
    Getting the default process group created by :func:`init_process_group`.

    """
    if not is_initialized():
        raise RuntimeError('Default process group has not been initialized, please make sure to call init_process_group.')
    return _default_pg


class ReduceOp(enum.IntEnum):
    """An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``, ``MIN``, ``MAX``, ``BAND``,
    ``BOR``, ``BXOR`` and ``AVG``."""
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BOR = 7
    BAND = 8
    BXOR = 9
    AVG = 10


class _SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum):
        input = input.contiguous()
        size = input.numel() // input.size(1)
        count = torch.tensor([size])
        mean, invstd = torch.batch_norm_stats(input, eps)
        count, mean, invstd = count, mean, invstd
        nums_ranks = bagua.get_world_size()
        count_all = torch.tensor([torch.empty_like(count).cpu().detach().numpy() for _ in range(nums_ranks)])
        mean_all = torch.tensor([torch.empty_like(mean).cpu().detach().numpy() for _ in range(nums_ranks)])
        invstd_all = torch.tensor([torch.empty_like(invstd).cpu().detach().numpy() for _ in range(nums_ranks)])
        allgather(count.unsqueeze(0), count_all)
        allgather(mean.unsqueeze(0), mean_all)
        allgather(invstd.unsqueeze(0), invstd_all)
        if _SYNC_BN_V3:
            counts_for_bngswc = count_all.view(-1).float()
        else:
            counts_for_bngswc = count_all.view(-1).tolist()
        mean, invstd = torch.batch_norm_gather_stats_with_counts(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, counts_for_bngswc)
        self.save_for_backward(input, weight, mean, invstd, count_all)
        return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_all = self.saved_tensors
        need_input_grad, need_weight_grad, need_bias_grad = self.needs_input_grad[0:3]
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, saved_input, mean, invstd, weight, need_input_grad, need_weight_grad, need_bias_grad)
        if need_input_grad:
            allreduce(sum_dy, sum_dy)
            allreduce(sum_dy_xmu, sum_dy_xmu)
            if _SYNC_BN_V4:
                count_all = count_all
            elif _SYNC_BN_V2 or _SYNC_BN_V3:
                count = count_all.sum()
            else:
                count = bagua.get_world_size()
            if _SYNC_BN_V4:
                grad_input = torch.batch_norm_backward_elemt(grad_output, saved_input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_all)
            else:
                grad_input = torch.batch_norm_backward_elemt(grad_output, saved_input, mean, invstd, weight, sum_dy / count, sum_dy_xmu / count)
        else:
            grad_input = None
        if weight is None or not need_weight_grad:
            grad_weight = None
        if weight is None or not need_bias_grad:
            grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """Applies synchronous BatchNorm for distributed module with N-dimensional BatchNorm layer(s).
    See `BatchNorm <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html?highlight=batchnorm#torch.nn.BatchNorm2d>`_ for more details.

    Arguments:
        num_features: Number of channels :math:`C` from the shape :math:`(N, C, ...)`.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.
        momentum: The value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1.
        affine: A boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``.
        track_running_stats: A boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``.

    .. note:: Only GPU input tensors are supported in the training mode.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'.format(input.dim()))

    def _run_bn(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, self.momentum, self.eps)

    @torch.jit.unused
    def _maybe_run_sync_bn(self, input):
        if bagua.get_world_size() == 1:
            return self._run_bn(input)
        return _SyncBatchNorm.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.momentum)

    def forward(self, input):
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')
        self._check_input_dim(input)
        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked = self.num_batches_tracked + 1
        if not self.training and self.track_running_stats:
            return self._run_bn(input)
        else:
            return self._maybe_run_sync_bn(input)

    @classmethod
    def convert_sync_batchnorm(cls, module):
        """Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        `torch.nn.SyncBatchNorm <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html?highlight=syncbatchnorm#torch.nn.SyncBatchNorm>`_ layers.

        Arguments:
            module (nn.Module): Module containing one or more :attr:`BatchNorm*D` layers

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        .. note:: This function must be called before :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua` method.

        Example::
            >>> # Network with nn.BatchNorm layer
            >>> model = torch.nn.Sequential(
            ...      torch.nn.Linear(D_in, H),
            ...      torch.nn.ReLU(),
            ...      torch.nn.Linear(H, D_out),
            ...    )
            >>> optimizer = torch.optim.SGD(
            ...      model.parameters(),
            ...      lr=0.01,
            ...      momentum=0.9
            ...    )
            >>> sync_bn_model = bagua.torch_api.contrib.sync_batchnorm.SyncBatchNorm.convert_sync_batchnorm(model)
            >>> bagua_model = sync_bn_model.with_bagua([optimizer], GradientAllReduce())
        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
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
            module_output.add_module(name, cls.convert_sync_batchnorm(child))
        del module
        return module_output


class TensorDtype(str, enum.Enum):
    F32 = 'f32'
    F16 = 'f16'
    U8 = 'u8'


def get_local_rank() ->int:
    """
    Get the rank of current node.

    Local rank is a unique identifier assigned to each process within a node.
    They are always consecutive integers ranging from 0 to ``local_size``.

    Returns:
        The local rank of the node.
    """
    return int(os.environ.get('LOCAL_RANK', 0))


def _is_elastic_launched():
    """Returns ``True`` if the current process was launched using the bagua.distributed.run command."""
    required_env_vars = {'RANK', 'GROUP_RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'}
    return required_env_vars.issubset(os.environ.keys())


def get_node_rank() ->int:
    """
    Get the rank among all nodes.

    Returns:
        The node rank of the node.
    """
    if _is_elastic_launched():
        return int(os.environ.get('GROUP_RANK', 0))
    else:
        return int(os.environ.get('NODE_RANK', 0))


def get_rank() ->int:
    """
    Get the rank of the default process group.

    Rank is a unique identifier assigned to each process within the default
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Returns:
        The rank of the default process group.
    """
    return int(os.environ.get('RANK', 0))


def get_world_size() ->int:
    """
    Get the number of processes in the default process group.

    Returns:
        The world size of the default process group.
    """
    return int(os.environ.get('WORLD_SIZE', 1))


@lru_cache(maxsize=None)
def _get_rank_mappings():
    rank_mappings = {}
    rank_tensors = torch.LongTensor(get_world_size(), 2)
    rank_tensors[get_rank()][0] = get_node_rank()
    rank_tensors[get_rank()][1] = get_local_rank()
    allgather_inplace(rank_tensors)
    for i in range(get_world_size()):
        rank_mappings[i] = rank_tensors[i][0].item(), rank_tensors[i][1].item()
    return rank_mappings


def broadcast_nccl_unique_id(comm_key: str, root):
    global _default_store
    if get_rank() == root:
        idstr = B.BaguaSingleCommunicatorPy.generate_nccl_unique_id_str()
        _default_store.set(comm_key, idstr)
    else:
        idstr = _default_store.get(comm_key)
        idstr = str(idstr, encoding='utf-8')
    return idstr


@lru_cache(maxsize=None)
def get_communicator(group_name: str, comm_name: str):
    global _pg_map
    pg = _pg_map[group_name]
    if comm_name == 'global':
        ranks = pg.ranks
    elif comm_name == 'inter':
        ranks = pg._get_inter_ranks()
    elif comm_name == 'intra':
        ranks = pg._get_intra_ranks()
    else:
        raise ValueError("comm_name should be one of ['global', 'inter', 'intra']")
    comm_key = '{}_{}_{}'.format(group_name, comm_name, ','.join(map(str, ranks)))
    nccl_unique_id = broadcast_nccl_unique_id(comm_key, root=ranks[0])
    if get_rank() not in ranks:
        return CommMember.NON_COMM_MEMBER
    rank = ranks.index(get_rank())
    nranks = len(ranks)
    comm = B.BaguaSingleCommunicatorPy(rank=rank, nranks=nranks, device_id=get_local_rank(), stream_ptr=pg.stream.cuda_stream, nccl_unique_id_str=nccl_unique_id)
    logging.debug('init bagua communicator %s-%s ok, global rank: %s rank: %s', group_name, comm_name, get_rank(), comm.rank())
    comm.cuda_stream = pg.stream
    return comm


class StatisticalAverage:

    def __init__(self, last_update_time: float=time.time(), records: List[float]=[], record_tail: Tuple[float, float]=(0.0, 0.0)) ->None:
        """Track and record the average over a period of time.

        Args:
            last_update_time (float, optional): last update time.
                Defaults to time.time().
            records (List[float], optional): statistical average value from
                `last_update_time`, records[i] is the average value from
                last_update_time to last_update_time + 2 ^ i (unit: seconds).
                Defaults to [].
            tail (Tuple[float, float], optional): tail of record, first one
                is tail length (unit: seconds), second one is tail average
                value. Defaults to (0., 0.).
        """
        self.last_update_time: float = last_update_time
        self.records: List[float] = records
        self.record_tail: Tuple[float, float] = record_tail

    def record_seconds(self) ->float:
        return 2.0 ** (len(self.records) - 1) if len(self.records) != 0 else 0.0

    def total_recording_time(self) ->float:
        tail_seconds, _ = self.record_tail
        return self.record_seconds() + tail_seconds

    def get_records_mean(self, last_n_seconds: float) ->float:
        if last_n_seconds <= 0.0:
            return 0.0
        records_seconds = self.record_seconds()
        tail_seconds, tail_mean = self.record_tail
        if len(self.records) == 0:
            return tail_mean
        if last_n_seconds < 1.0:
            return self.records[0]
        if last_n_seconds <= records_seconds:
            floor_id = max(0, math.floor(math.log(last_n_seconds, 2.0)))
            floor_time = 2.0 ** floor_id
            if floor_id + 1 < len(self.records):
                a, b = self.records[floor_id], self.records[floor_id + 1]
                a_l, b_l = floor_time, floor_time * 2.0
                mean = a + (b - a) * (last_n_seconds - a_l) / (b_l - a_l)
            else:
                mean = self.records[floor_id]
        elif last_n_seconds <= records_seconds + tail_seconds:
            a, b = self.records[-1], tail_mean
            a_l, b_l = records_seconds, records_seconds + tail_seconds
            mean = a + (b - a) * (last_n_seconds - a_l) / (b_l - a_l)
        else:
            mean = tail_mean
        return mean

    def record(self, val: float):
        now = time.time()
        time_dist: float = now - self.last_update_time
        new_records: List[float] = []
        new_tail: Tuple[float, float] = (0.0, 0.0)
        for i in range(64):
            coverage_period = 2.0 ** i
            if coverage_period <= time_dist:
                new_records.append(val)
            elif coverage_period <= time_dist + self.total_recording_time():
                record_contribution_percentage = time_dist / coverage_period
                new_val = val * record_contribution_percentage + self.get_records_mean(coverage_period - time_dist) * (1.0 - record_contribution_percentage)
                new_records.append(new_val)
                if coverage_period > time_dist + self.total_recording_time():
                    break
            else:
                new_total_time = time_dist + self.total_recording_time()
                report_contribution_percentage = time_dist / new_total_time
                tail_len = new_total_time - 2.0 ** (len(new_records) - 1)
                tail_val = val * report_contribution_percentage + self.get_records_mean(self.total_recording_time()) * (1.0 - report_contribution_percentage)
                new_tail = tail_len, tail_val
                break
        self.last_update_time = now
        self.records = new_records
        self.record_tail = new_tail

    def get(self, last_n_seconds: float) ->float:
        time_dist = time.time() - self.last_update_time
        if last_n_seconds <= time_dist:
            if len(self.records) != 0:
                return self.records[0]
            else:
                tail_mean, _ = self.record_tail
                return tail_mean
        return self.get_records_mean(last_n_seconds - time_dist)

    def __str__(self) ->str:
        return str({'last_update_time': self.last_update_time, 'records': self.records, 'record_tail': self.record_tail})


@lru_cache(maxsize=None)
def get_backend(model_name: str):
    backend = B.BaguaCommBackendPy(100, device_id=get_local_rank())
    backend.model_name = model_name
    return backend


def reset_error_retry(request_func):
    """Retry request when catch ConnectionResetError."""

    def wrap(*args, **kwargs):
        MAX_RETRIES = 3
        for retry in range(MAX_RETRIES + 1):
            try:
                result = request_func(*args, **kwargs)
                return result
            except (ConnectionResetError, requests.exceptions.ConnectionError) as e:
                if retry == MAX_RETRIES:
                    raise e
                logging.warning('request failed, retry={}, e={}'.format(retry, e))
                time.sleep(1)
    return wrap


def get_master_addr() ->str:
    return os.environ.get('MASTER_ADDR', '127.0.0.1')


@lru_cache(maxsize=None)
def get_hyperparameters_service_client():
    global _autotune_service_port
    hyperparameters_service_client = AutotuneClient(get_master_addr(), _autotune_service_port)
    return hyperparameters_service_client


def is_moe_param(param: torch.Tensor) ->bool:
    if hasattr(param, 'expert') and param.expert:
        return True
    return False


def to_bagua_datatype(datatype):
    if datatype == torch.float32:
        return 'f32'
    elif datatype == torch.float16:
        return 'f16'
    elif datatype == torch.uint8:
        return 'u8'
    elif datatype == torch.long:
        return 'i64'
    else:
        raise ValueError(f'unsupported data type {datatype}.')


class _AlgorithmRegistry(dict):

    def register(self, name: str, algorithm: Callable, description: Optional[str]=None):
        """Registers an Bagua Algorithm mapped to a name and with required metadata.

        Args:
            name: The name that identifies a Bagua algorithm, e.g. "gradient_allreduce".
            algorithm: Class of the Bagua algorithm.
            description: Description of the Bagua algorithm.
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'`name` must be a str, found {name}')
        if name in self:
            raise ValueError(f"'{name}' is already present in the registry.")
        data: Dict[str, Any] = {}
        data['algorithm'] = algorithm
        data['description'] = description if description is not None else ''
        self[name] = data

    def get(self, name: str) ->Callable:
        """Calls the registered Bagua algorithm with the name and returns the algorithm class.

        Args:
            name: The name that identifies a Bagua algorithm, e.g. "gradient_allreduce".

        Returns:
            The class of the Bagua algorithm.
        """
        if name in self:
            data = self[name]
            return data['algorithm']
        err_msg = "'{}' not found in registry. Available names: {}"
        available_names = ', '.join(sorted(self.keys())) or 'none'
        raise KeyError(err_msg.format(name, available_names))

    def available_algorithms(self) ->List[str]:
        """Returns a list of registered Bagua algorithms."""
        return list(self.keys())

    def __str__(self) ->str:
        return 'Registered Algorithms: {}'.format(', '.join(self.keys()))


GlobalAlgorithmRegistry = _AlgorithmRegistry()


def check_contiguous(tensors):
    data_ptr = None
    for t in tensors:
        if data_ptr is not None and t.data_ptr() != data_ptr:
            return False
        data_ptr = t.data_ptr() + t.numel() * t.element_size()
    return True


def get_flattened_tensor(tensors: List[torch.Tensor]) ->torch.Tensor:
    if len(tensors) == 0:
        return
    total_size = 0
    for tensor in tensors:
        total_size += tensor.numel()
    flatten_tensor = torch.zeros(total_size, dtype=tensors[0].dtype, device=tensors[0].device)
    offset = 0
    for tensor in tensors:
        flatten_tensor[offset:offset + tensor.numel()] = tensor.reshape(-1)
        offset += tensor.numel()
    return flatten_tensor


class Experts(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1):
        super(Experts, self).__init__()
        self.bagua_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        for expert in self.bagua_experts:
            for name, param in expert.named_parameters():
                param.expert = True

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.bagua_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]
            expert_outputs += [out]
        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output


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


def multiplicative_jitter(x, device: torch.device, epsilon=0.01):
    """
    Modified from swtich transformer paper. mesh transformers
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


def gumbel_rsample(shape: Tuple, device: torch.device) ->Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample
        gumbel_map[device] = gumbel
    return gumbel(shape)


def top1gating(logits: torch.Tensor, capacity_factor: float, min_capacity: int, used_token: torch.Tensor=None, noisy_gate_policy: Optional[str]=None) ->Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    gates = F.softmax(logits, dim=1)
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = math.ceil(num_tokens / num_experts * capacity_factor)
    if capacity < min_capacity:
        capacity = min_capacity
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    if used_token is not None:
        mask1 = torch.einsum('s,se->se', used_token, mask1)
    exp_counts = torch.sum(mask1, dim=0).detach()
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts
    uniform = exp_selection_uniform_map.get(logits.device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device), high=torch.tensor(1.0, device=logits.device)).rsample
        exp_selection_uniform_map[logits.device] = uniform
    mask1_rand = mask1 * uniform(mask1.shape)
    assert logits.shape[0] >= min_capacity, 'No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or inrease your batch size.'
    _, top_idx = torch.topk(mask1_rand, k=capacity, dim=0)
    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    locations1 = torch.cumsum(new_mask1, dim=0) - 1
    locations1_s = torch.sum(locations1 * new_mask1, dim=1)
    mask1_float = new_mask1.float()
    gates = gates * mask1_float
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).float()
    combine_weights = torch.einsum('se,sc->sec', gates, locations1_sc)
    dispatch_mask = combine_weights.bool()
    return l_aux, combine_weights, dispatch_mask, exp_counts


def top2gating(logits: torch.Tensor, capacity_factor: float) ->Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    gates = F.softmax(logits, dim=1)
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = math.ceil(2 * num_tokens / num_experts * capacity_factor)
    indices1_s = torch.argmax(gates, dim=1)
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
    gates1_s = torch.einsum('se,se->s', gates, mask1_float)
    gates2_s = torch.einsum('se,se->s', gates, mask2_float)
    denom_s = gates1_s + gates2_s
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    gates1 = torch.einsum('s,se->se', gates1_s, mask1_float)
    gates2 = torch.einsum('s,se->se', gates2_s, mask2_float)
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).float()
    locations2_sc = F.one_hot(locations2_s, num_classes=capacity).float()
    combine1_sec = torch.einsum('se,sc->sec', gates1, locations1_sc)
    combine2_sec = torch.einsum('se,sc->sec', gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    return l_aux, combine_weights, dispatch_mask, exp_counts


class TopKGate(torch.nn.Module):
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

    def __init__(self, model_dim: int, num_experts: int, k: int=1, capacity_factor: float=1.0, eval_capacity_factor: float=1.0, min_capacity: int=4, noisy_gate_policy: Optional[str]=None) ->None:
        super().__init__()
        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy

    def forward(self, input: torch.Tensor, used_token: torch.Tensor=None) ->Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(input_fp32)
        if self.k == 1:
            return top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor, self.min_capacity, used_token, self.noisy_gate_policy if self.training else None)
        else:
            return top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor)


class MoE(torch.nn.Module):

    def __init__(self, hidden_size, expert, num_local_experts=1, k=1, output_dropout_prob=0.0, capacity_factor=1.0, eval_capacity_factor=1.0, min_capacity=4, noisy_gate_policy: typing.Optional[str]=None):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.

            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).

            num_local_experts (int, optional): default=1, number of local experts per gpu.

            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.

            output_dropout_prob (float, optional): default=0.0, output dropout probability.

            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.

            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.

            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.

            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
        """
        super(MoE, self).__init__()
        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], 'Unsupported noisy_gate_policy: ' + noisy_gate_policy
        self.num_experts = num_local_experts * bagua.get_world_size()
        logging.info(f'num_experts: {self.num_experts} | num_local_experts: {num_local_experts} | world_size: {bagua.get_world_size()}')
        experts = Experts(expert, num_local_experts)
        self.bagua_moe = MOELayer(TopKGate(hidden_size, self.num_experts, k, capacity_factor, eval_capacity_factor, min_capacity, noisy_gate_policy), experts, num_local_experts, group=dist.group.WORLD)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states, used_token=None):
        """MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.bagua_moe(hidden_states, used_token)
        output = self.dropout(output)
        return output, self.bagua_moe.l_aux, self.bagua_moe.exp_counts


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=True)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class DoubleGpuNet(nn.Module):

    def __init__(self, gpus):
        super(DoubleGpuNet, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(torch.tensor([2, 2]).long(), requires_grad=False)

    def forward(self, x):
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class QuadraGpuNet(nn.Module):

    def __init__(self, gpus):
        super(QuadraGpuNet, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.fc4 = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(torch.tensor([2, 2]).long(), requires_grad=False)

    def forward(self, x):
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        dev2 = self.fc3.weight.device
        dev3 = self.fc4.weight.device
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)


class ConvNet(nn.Module):

    def __init__(self, gpus, layouts, dtypes):
        super(ConvNet, self).__init__()
        self.dtypes = dtypes
        if isinstance(gpus, list):
            self.layer_gpus = gpus
        else:
            gpus = [gpus] * 4
        self.conv0 = torch.nn.Conv2d(8, 16, (2, 2))
        self.conv1 = torch.nn.Conv2d(16, 32, (2, 2))
        self.conv2 = torch.nn.Conv2d(32, 16, (2, 2))
        self.conv3 = torch.nn.Conv2d(16, 8, (2, 2))

    def forward(self, x):
        x = x
        gpus = self.layer_gpus if hasattr(self, 'layer_gpus') else [x.device] * 4
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class Task(nn.Module):

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        return self.p + x


class ModuleForDdpCommHook(nn.Module):

    def __init__(self):
        super().__init__()
        self.t0 = Task()

    def forward(self, x, rank):
        return self.t0(x + rank)


class SparseGradientModule(nn.Module):

    def __init__(self):
        super(SparseGradientModule, self).__init__()
        self.embedding = nn.EmbeddingBag(10, 10, sparse=True)

    def forward(self, x):
        return F.softmax(self.embedding(x), dim=1)


flatten = torch._utils._flatten_dense_tensors


def get_peer_rank(peer_selection_mode, rank, nranks, step, communication_interval):
    comm_step = step // communication_interval
    if peer_selection_mode == 'shift_one':
        if rank < nranks // 2:
            return (comm_step + rank) % ((nranks + 1) // 2) + nranks // 2
        else:
            return (rank - nranks // 2 - comm_step) % (nranks // 2)
    else:
        ValueError('Unsupported `peer_selection_mode`')


unflatten = torch._utils._unflatten_dense_tensors


class DecentralizedAlgor(nn.Module):

    def __init__(self, module, optimizer, hierarchical, peer_selection_mode, communication_interval):
        super(DecentralizedAlgor, self).__init__()
        self.module = module
        self.optimizer = optimizer
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval
        self.step_count = 0
        assert torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data, src=0)

    def _build_params(self):
        return [param.data for param in list(self.module.parameters()).__reversed__()]

    def communicate_with_peer(self):
        if self.peer_selection_mode == 'all':
            torch.distributed.all_reduce(self.peer_weight)
            self.peer_weight /= self.world_size
        elif self.peer_selection_mode == 'shift_one':
            peer_rank = get_peer_rank(self.peer_selection_mode, self.rank, self.world_size, self.step_count, self.communication_interval)
            weight = self.weight.cpu()
            peer_weight = self.peer_weight.cpu()
            requests = []
            requests.append(torch.distributed.isend(weight, peer_rank))
            requests.append(torch.distributed.irecv(peer_weight, peer_rank))
            for req in requests:
                req.wait()
            self.peer_weight = peer_weight
            self.weight = weight
            self.peer_weight += self.weight
            self.peer_weight /= 2
        else:
            raise ValueError('Unsupported `peer_selection_mode`')

    def _should_communicate(self):
        return self.step_count % self.communication_interval == 0

    def forward(self, *inputs, **kwargs):
        if self._should_communicate():
            self.weight = flatten(self._build_params())
            self.peer_weight = flatten(self._build_params())
            self.communicate_with_peer()
        result = self.module(*inputs, **kwargs)
        return result

    def step(self):
        if self._should_communicate():
            params = self._build_params()
            for buf, synced in zip(params, unflatten(self.peer_weight, params)):
                buf.copy_(synced)
        self.optimizer.step()
        self.step_count += 1


class MinMaxUInt8:

    def __init__(self):
        self.eps = 1e-07
        self.quantization_level = 255.0

    def compress(self, tensor: torch.Tensor) ->(torch.Tensor, torch.Tensor):
        _min = torch.min(tensor)
        _max = torch.max(tensor)
        scale = self.quantization_level / (_max - _min + self.eps)
        upper_bound = torch.round(_max * scale)
        lower_bound = upper_bound - self.quantization_level
        level = torch.round(tensor * scale)
        level = torch.clamp(level, max=upper_bound)
        _minmax = torch.zeros(2, dtype=tensor.dtype, device=tensor.device)
        _minmax[0] = _min
        _minmax[1] = _max
        return _minmax, level - lower_bound

    def decompress(self, _minmax: torch.Tensor, compressed: torch.Tensor) ->torch.Tensor:
        _min = _minmax[0]
        _max = _minmax[1]
        scale = self.quantization_level / (_max - _min + self.eps)
        upper_bound = torch.round(_max * scale)
        lower_bound = upper_bound - self.quantization_level
        return (compressed.float() + lower_bound) / scale


def apply_flattened_call(bucket, call, extra_args=None):
    coalesced = flatten(bucket)
    if extra_args is not None:
        call(coalesced, *extra_args)
    else:
        call(coalesced)
    if call is dist.all_reduce:
        coalesced /= dist.get_world_size()
    for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
        buf.copy_(synced)


class LowPrecDecentralizedAlgor(nn.Module):

    def __init__(self, module, optimizer, hierarchical, communication_interval):
        super(LowPrecDecentralizedAlgor, self).__init__()
        self.module = module
        self.optimizer = optimizer
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.compressor = MinMaxUInt8()
        self.step_count = 0
        assert torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data, src=0)
        self.weight = flatten(self._build_params())
        self.left_peer_weight = self.weight.detach().clone()
        self.right_peer_weight = self.weight.detach().clone()

    def _build_params(self):
        return [param.data for param in list(self.module.parameters()).__reversed__()]

    def _should_communicate(self):
        return self.step_count % self.communication_interval == 0

    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        return result

    def step(self):
        self.optimizer.step()

        def communicate_with_peers(tensor: torch.Tensor, comm_size: int) ->(torch.Tensor, torch.Tensor):
            if comm_size == 1:
                return tensor, tensor
            tensor = tensor.cpu()
            left_tensor = torch.zeros_like(tensor)
            right_tensor = torch.zeros_like(tensor)
            left_peer_rank = (self.rank + self.world_size - 1) % comm_size
            right_peer_rank = (self.rank + 1) % comm_size
            requests = []
            requests.append(torch.distributed.isend(tensor, left_peer_rank))
            requests.append(torch.distributed.isend(tensor, right_peer_rank))
            requests.append(torch.distributed.irecv(left_tensor, left_peer_rank))
            requests.append(torch.distributed.irecv(right_tensor, right_peer_rank))
            for req in requests:
                req.wait()
            return left_tensor, right_tensor

        def update_weight_fn(x, comm_size):
            x += 1 / 3 * self.left_peer_weight
            x += 1 / 3 * self.right_peer_weight
            x -= 5 / 3 * self.weight
            minmax, compressed = self.compressor.compress(x)
            left_compressed, right_compressed = communicate_with_peers(compressed, comm_size)
            left_minmax, right_minmax = communicate_with_peers(minmax, comm_size)
            self.left_peer_weight += self.compressor.decompress(left_minmax, left_compressed)
            self.right_peer_weight += self.compressor.decompress(right_minmax, right_compressed)
            diff = self.compressor.decompress(minmax, compressed)
            x.copy_(self.weight + diff)
            self.weight.copy_(x)

        def hierarchical_update_weight_fn(x):
            torch.distributed.reduce(x, dst=0)
            if self.rank == 0:
                x /= self.world_size
                update_weight_fn(x, comm_size=1)
            torch.distributed.broadcast(x, 0)
        if self._should_communicate():
            weights = self._build_params()
            if self.hierarchical:
                apply_flattened_call(weights, hierarchical_update_weight_fn)
            else:
                apply_flattened_call(weights, lambda x: update_weight_fn(x, self.world_size))
        self.step_count += 1


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=True)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 30, bias=True)
        self.fc3 = nn.Linear(30, 20, bias=True)
        self.fc4 = nn.Linear(20, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvNet,
     lambda: ([], {'gpus': False, 'layouts': 4, 'dtypes': torch.float32}),
     lambda: ([torch.rand([4, 8, 64, 64])], {}),
     True),
    (Experts,
     lambda: ([], {'expert': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ModuleForDdpCommHook,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 2, 2]), torch.rand([4, 4, 2, 2])], {}),
     True),
    (Task,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     True),
]

class Test_BaguaSys_bagua(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

