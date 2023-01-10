import sys
_module = sys.modules[__name__]
del sys
get_changed_code_files = _module
build_utils = _module
conf = _module
mocks = _module
adasum_small_model = _module
pytorch_imagenet_resnet50_elastic = _module
pytorch_mnist_elastic = _module
pytorch_synthetic_benchmark_elastic = _module
tensorflow_keras_mnist_elastic = _module
tensorflow2_keras_mnist_elastic = _module
tensorflow2_mnist_elastic = _module
tensorflow2_synthetic_benchmark_elastic = _module
keras_imagenet_resnet50 = _module
keras_mnist = _module
keras_mnist_advanced = _module
mxnet2_mnist = _module
mxnet_imagenet_resnet50 = _module
mxnet_mnist = _module
pytorch_imagenet_resnet50 = _module
pytorch_lightning_mnist = _module
pytorch_mnist = _module
pytorch_synthetic_benchmark = _module
basic_ray_elastic = _module
pytorch_ray_elastic = _module
tensorflow2_mnist_ray = _module
keras_spark3_rossmann = _module
keras_spark_mnist = _module
keras_spark_rossmann_estimator = _module
keras_spark_rossmann_run = _module
pytorch_lightning_spark_mnist = _module
pytorch_lightning_spark_mnist_legacy = _module
pytorch_spark_mnist = _module
tensorflow2_mnist_data_service = _module
tensorflow_keras_mnist = _module
tensorflow_mnist = _module
tensorflow_mnist_eager = _module
tensorflow_mnist_estimator = _module
tensorflow_synthetic_benchmark = _module
tensorflow_word2vec = _module
tensorflow2_keras_mnist = _module
tensorflow2_keras_synthetic_benchmark = _module
tensorflow2_mnist = _module
tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher = _module
tensorflow2_mnist_data_service_train_fn_training_side_dispatcher = _module
tensorflow2_synthetic_benchmark = _module
horovod = _module
_keras = _module
callbacks = _module
elastic = _module
common = _module
basics = _module
exceptions = _module
process_sets = _module
util = _module
data = _module
data_loader_base = _module
keras = _module
mxnet = _module
compression = _module
functions = _module
mpi_ops = _module
ray = _module
adapter = _module
driver_service = _module
elastic_v2 = _module
ray_logger = _module
runner = _module
strategy = _module
utils = _module
worker = _module
service = _module
compute_service = _module
task_service = _module
codec = _module
config_parser = _module
env = _module
host_hash = _module
hosts = _module
network = _module
safe_shell_exec = _module
secret = _module
settings = _module
timeout = _module
tiny_shell_exec = _module
driver = _module
constants = _module
discovery = _module
registration = _module
rendezvous = _module
gloo_run = _module
http = _module
http_client = _module
http_server = _module
js_run = _module
launch = _module
mpi_run = _module
run_task = _module
task = _module
task_fn = _module
cache = _module
lsf = _module
remote = _module
streams = _module
threads = _module
spark = _module
_namedtuple_fix = _module
backend = _module
datamodule = _module
estimator = _module
params = _module
serialization = _module
store = _module
data_loaders = _module
pytorch_data_loaders = _module
host_discovery = _module
job_id = _module
mpirun_rsh = _module
rsh = _module
bare = _module
optimizer = _module
tensorflow = _module
lightning = _module
estimator = _module
legacy = _module
remote = _module
util = _module
gloo_exec_fn = _module
mpirun_exec_fn = _module
task_info = _module
compute_worker = _module
estimator = _module
remote = _module
util = _module
gradient_aggregation = _module
gradient_aggregation_eager = _module
sync_batch_norm = _module
compression = _module
elastic = _module
sampler = _module
state = _module
functions = _module
mpi_ops = _module
optimizer = _module
sync_batch_norm = _module
setup = _module
conftest = _module
elastic_tensorflow2_main = _module
elastic_tensorflow_keras_main = _module
elastic_tensorflow_main = _module
elastic_torch_main = _module
elastic_common = _module
elastic_spark_common = _module
test_elastic_spark_tensorflow = _module
test_elastic_spark_tensorflow2 = _module
test_elastic_spark_torch = _module
test_elastic_tensorflow = _module
test_elastic_tensorflow2 = _module
test_elastic_tensorflow_keras = _module
test_elastic_torch = _module
test_interactiverun = _module
test_spark = _module
test_spark_keras = _module
test_spark_lightning = _module
test_spark_torch = _module
test_stall = _module
test_static_run = _module
base_test_mxnet = _module
base_test_tensorflow = _module
test_adasum_pytorch = _module
test_adasum_tensorflow = _module
test_common = _module
test_compute_worker = _module
test_keras = _module
test_mxnet1 = _module
test_mxnet2 = _module
test_process_sets_multi_comm = _module
test_process_sets_static = _module
test_tensorflow = _module
test_tensorflow2_keras = _module
test_tensorflow2_keras_process_sets = _module
test_tensorflow_keras = _module
test_tensorflow_process_sets = _module
test_tensorflow_process_sets_dynamic = _module
test_timeline = _module
test_torch = _module
test_xla = _module
test_xla_process_sets = _module
run_safe_shell_exec = _module
sleep = _module
test_buildkite = _module
test_compute_service = _module
test_elastic_discovery = _module
test_elastic_driver = _module
test_elastic_run = _module
test_ray = _module
test_ray_elastic = _module
test_ray_elastic_v2 = _module
test_run = _module
test_service = _module
test_task_service = _module
test_torch_elastic = _module
test_util = _module
spark_common = _module

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


import re


from typing import List


from typing import Dict


from typing import Set


import torch


import random


import numpy as np


import torch.backends.cudnn as cudnn


import torch.multiprocessing as mp


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data.distributed


from torch.utils.tensorboard import SummaryWriter


from torchvision import datasets


from torchvision import transforms


from torchvision import models


import math


import torch.nn as nn


import warnings


import logging


import copy


import numbers


import time


import torch.utils.data


import collections


from collections.abc import Iterable


from torch.autograd.function import Function


from torch.nn.modules.batchnorm import _BatchNorm


import uuid


import itertools


from torch.nn import functional as F


import inspect


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


version = __version__


class HorovodInternalError(RuntimeError):
    """Internal error raised when a Horovod collective operation (e.g., allreduce) fails.

    This is handled in elastic mode as a recoverable error, and will result in a reset event.
    """
    pass


def is_iterable(x):
    try:
        _ = iter(x)
    except TypeError:
        return False
    return True


_NULL = ''


def _allgather_function_factory(tensor):
    return 'horovod_torch_allgather_async_' + tensor.type().replace('.', '_')


def _check_function(function_factory, tensor):
    function = function_factory(tensor)
    if not hasattr(mpi_lib, function):
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    if not tensor.is_contiguous():
        raise ValueError('Tensor is required to be contiguous.')
    return function


_handle_map = {}


def _allreduce_function_factory(tensor):
    return 'horovod_torch_allreduce_async_' + tensor.type().replace('.', '_')


def _cache(f):
    cache = dict()

    def wrapper(*args, **kwargs):
        key = args, frozenset(kwargs.items())
        if key in cache:
            return cache[key]
        else:
            retval = f(*args, **kwargs)
            cache[key] = retval
            return retval
    return wrapper


def _check_extension_lambda(ext_base_name, fn, fn_desc, verbose):
    """
    Tries to load the extension in a new process.  If successful, puts fn(ext)
    to the queue or False otherwise.  Mutes all stdout/stderr.
    """

    def _target_fn(ext_base_name, fn, fn_desc, queue, verbose):
        if verbose:
            None
        else:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        try:
            ext = importlib.import_module('.' + ext_base_name, 'horovod')
            result = fn(ext)
        except:
            traceback.print_exc()
            result = None
        if verbose:
            None
        queue.put(result)
    ctx = multiprocessing.get_context('fork')
    queue = ctx.Queue()
    p = ctx.Process(target=_target_fn, args=(ext_base_name, fn, fn_desc, queue, verbose))
    p.daemon = True
    p.start()
    p.join()
    return queue.get_nowait()


@_cache
def gpu_available(ext_base_name, verbose=False):
    available_fn = lambda ext: ext._check_has_gpu()
    return _check_extension_lambda(ext_base_name, available_fn, 'running with GPU', verbose) or False


def num_rank_is_power_2(num_rank):
    """
    Tests if the given number of ranks is of power of 2. This check is required
    for Adasum allreduce.
    TODO support non-power of 2 ranks.
    """
    return num_rank != 0 and num_rank & num_rank - 1 == 0


def get_average_backwards_compatibility_fun(reduce_ops):
    """
    Handle backwards compatibility between the old average and the new op parameters.
    Old code using the average parameter (e.g. hvd.allreduce(tensor, average=False))
    gets unchanged behavior, but mixing old and new is disallowed (e.g. no
    hvd.allreduce(tensor, average=False, op=hvd.Adasum)).
    """

    def impl(op, average):
        if op is not None:
            if average is not None:
                raise ValueError('The op parameter supersedes average. Please provide only one of them.')
            return op
        elif average is not None:
            warnings.warn('Parameter `average` has been replaced with `op` and will be removed in v1.0', DeprecationWarning)
            return reduce_ops.Average if average else reduce_ops.Sum
        else:
            return reduce_ops.Average
    return impl


def synchronize(handle):
    """
    Synchronizes an asynchronous allreduce, allgather, alltoall, broadcast, or reducescatter operation
    until  it's completed. Returns the result of the operation.

    Arguments:
        handle: A handle returned by an allreduce, allgather, alltoall, broadcast, or reducescatter
                asynchronous operation.

    Returns:
        A single output tensor of the operation or a tuple of multiple output tensors.
    """
    if handle not in _handle_map:
        return
    try:
        mpi_lib.horovod_torch_wait_and_clear(handle)
        output = _handle_map.pop(handle)[-1]
        return output
    except RuntimeError as e:
        _handle_map.pop(handle, None)
        raise HorovodInternalError(e)


class _SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum):
        input = input.contiguous()
        size = input.numel() // input.size(1)
        count = torch.tensor([size])
        mean, invstd = torch.batch_norm_stats(input, eps)
        count_handle = allgather_async(count.unsqueeze(0), name='sync_batch_norm.count')
        mean_handle = allgather_async(mean.unsqueeze(0), name='sync_batch_norm.mean')
        invstd_handle = allgather_async(invstd.unsqueeze(0), name='sync_batch_norm.invstd')
        count_all = synchronize(count_handle)
        mean_all = synchronize(mean_handle)
        invstd_all = synchronize(invstd_handle)
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
            sum_dy_handle = allreduce_async(sum_dy, op=Sum, name='sync_batch_norm.sum_dy')
            sum_dy_xmu_handle = allreduce_async(sum_dy_xmu, op=Sum, name='sync_batch_norm.sum_dy_xmu')
            sum_dy = synchronize(sum_dy_handle)
            sum_dy_xmu = synchronize(sum_dy_xmu_handle)
            if _SYNC_BN_V4:
                count_all = count_all
            elif _SYNC_BN_V2 or _SYNC_BN_V3:
                count = count_all.sum()
            else:
                count = size()
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
    """Applies synchronous version of N-dimensional BatchNorm.

    In this version, normalization parameters are synchronized across workers during forward pass.
    This is very useful in situations where each GPU can fit a very small number of examples.

    See https://pytorch.org/docs/stable/nn.html#batchnorm2d for more details about BatchNorm.

    Arguments:
        num_features: number of channels `C` from the shape `(N, C, ...)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to `None` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to `True`, this module has
            learnable affine parameters. Default: `True`
        track_running_stats: a boolean value that when set to `True`, this
            module tracks the running mean and variance, and when set to `False`,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: `True`
    
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
        if size() == 1:
            return self._run_bn(input)
        return _SyncBatchNorm.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.momentum)

    def forward(self, input):
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')
        self._check_input_dim(input)
        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
        if not self.training and self.track_running_stats:
            return self._run_bn(input)
        else:
            return self._maybe_run_sync_bn(input)


class LegacyXOR(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LegacyXOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 8)
        self.lin2 = nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x


class XOR(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 8)
        self.lin2 = nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LegacyXOR,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (XOR,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_horovod_horovod(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

