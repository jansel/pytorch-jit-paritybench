import sys
_module = sys.modules[__name__]
del sys
byteps = _module
__version__ = _module
_keras = _module
callbacks = _module
common = _module
keras = _module
misc = _module
imagenet18 = _module
mxnet = _module
ops = _module
server = _module
tensorflow = _module
compression = _module
util = _module
compression = _module
cross_barrier = _module
ops = _module
parallel = _module
distributed = _module
keras_imagenet_resnet50 = _module
keras_mnist = _module
keras_mnist_advanced = _module
data = _module
data_byteps = _module
find_mxnet = _module
fit = _module
fit_byteps = _module
modelzoo = _module
symbols = _module
alexnet = _module
googlenet = _module
lenet = _module
mlp = _module
mobilenet = _module
mobilenetv2 = _module
resnet = _module
resnetv1 = _module
resnext = _module
vgg = _module
train_gluon_mnist_byteps = _module
train_imagenet_byteps = _module
benchmark_byteps = _module
benchmark_byteps_ddp = _module
benchmark_cross_barrier_byteps = _module
elastic_benchmark_byteps = _module
train_imagenet_resnet50_byteps = _module
train_imagenet_resnet_byteps_ddp = _module
train_mnist_byteps = _module
synthetic_benchmark = _module
synthetic_benchmark_tf2 = _module
tensorflow2_mnist = _module
tensorflow_mnist = _module
dist_launcher = _module
launch = _module
pre_setup = _module
setup = _module
test_mxnet = _module
test_tensorflow_keras = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import time


import torch


import collections


import logging


import math


from torch.nn.modules import Module


from torch.cuda._utils import _get_device_index


import numpy as np


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data.distributed


from torchvision import models


import torch.multiprocessing as mp


import torchvision


import torchvision.transforms as transforms


import torch.nn as nn


import torch.distributed as dist


from torchvision import datasets


from torchvision import transforms


import random


import warnings


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torchvision.datasets as datasets


import torchvision.models as models


import re


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class NoneCompressor(Compressor):
    """Default no-op compression."""

    @staticmethod
    def compress(tensor):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class Compression(object):
    """Optional gradient compression algorithm used during push_pull."""
    """Do not compress the gradients. This is the default."""
    none = NoneCompressor
    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix
    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix
    return '.so'


def byteps_push_pull(tensor, version=0, priority=0, name=None, is_average=True):
    """
    A function that performs pushing and pulling tensors

    The operation is keyed by the name. If name is not provided, an
    incremented auto-generated name is used. The tensor type and shape must be
    the same on all BytePS processes for a given name. The reduction will not
    start until all processes are ready to send and receive the tensor.

    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires tensors, then callings this function will allow tensors
    to be computed and backpropagated.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.

    Returns:
        None
    """
    c_in = tensor.handle
    if isinstance(name, string_types):
        check_call(MXNET_LIB_CTYPES.byteps_mxnet_push_pull_async(c_in, c_str(name), ctypes.c_int(version), ctypes.c_int(priority), ctypes.c_bool(is_average)))
    else:
        check_call(MXNET_LIB_CTYPES.byteps_mxnet_push_pull_async(c_in, name, ctypes.c_int(version), ctypes.c_int(priority), ctypes.c_bool(is_average)))
    return


def byteps_torch_set_num_grads(num_grads_):
    c_lib.byteps_torch_set_num_grads(num_grads_)
    return 0


def declare(name):
    c_lib.byteps_torch_declare_tensor(name.encode())
    return 0


class DistributedDataParallel(Module):
    """Implements distributed data parallelism that is based on
    byteps push-pull.

    This container parallelizes the application of the given module by splitting
    the input across multiple devices, and each device handles a portion of the
    input. During the backwards pass, gradients from each node are averaged.

    ``DistributedDataParallel`` can be used in the following way:

    Single-Process Single-GPU

    This is currently the only way to use ``DistributedDataParallel``, with
    multiple processes, each of which operates on a single GPU.

    Here is how to use it: on each host with N GPUs, you should spawn up N
    processes, while ensuring that each process individually works on a single
    GPU from 0 to N-1. Therefore, it is your job to ensure that your training
    script operates on a single given GPU by calling:

        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> model = DistributedDataParallel(model, device_ids=[i])

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

    .. warning::
        This module works only with the ``device_ids`` containing one entry.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) is a distributed synchronization
        point. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
        This module assumes all buffers and gradients are dense.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    .. warning::
        You should never try to change your model's parameters after wrapping
        up your model with DistributedDataParallel. In other words, when
        wrapping up your model with DistributedDataParallel, the constructor of
        DistributedDataParallel will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model's parameters after
        the DistributedDataParallel construction, this is not supported and
        unexpected behaviors can happen, since some parameters' gradient
        reduction functions might not get called.

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in rank 0 to all
        other replicas in the system in every iteration.

    .. note::
        Some models have branches, part of the model is skipped during the
        forward pass. In that case it's required to call the
        DistributedDataParallel.synchronize() after loss.backward(), e.g:

            >>> model = DistributedDataParallel(model, device_ids=[i])
            >>> output = model(data)
            >>> loss = F.nll_loss(output, target)
            >>> loss.backward()
            >>> model.synchronize()
            >>> optimizer.step()

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices. This should
                   contain only one entry. The `module` replica is placed on
                   ``device_ids[0]``.
        broadcast_buffers (bool): flag that enables syncing (broadcasting) buffers of
                          the module at beginning of the forward function.
                          (default: ``True``)

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> net = torch.nn.DistributedDataParallel(model, device_ids=[2])
    """

    def __init__(self, module, device_ids=None, broadcast_buffers=True, compression=Compression.none):
        super(DistributedDataParallel, self).__init__()
        assert device_ids and len(device_ids) == 1, 'DistributedDataParallel device_ids contain exactlyone entry, but got {}.'.format(device_ids)
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.module = module
        self.broadcast_buffers = broadcast_buffers
        self.require_forward_param_sync = broadcast_buffers
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._num_grads = 1
        self.modules_buffers = [list(self.module.buffers())]
        self._compression = compression
        self._enable_async = False
        named_parameters = self.module.named_parameters()
        named_parameters = list(named_parameters)
        if len(named_parameters) > 0:
            if isinstance(named_parameters[0][1], torch.Tensor):
                if any([(not isinstance(p, torch.Tensor)) for name, p in named_parameters]):
                    raise ValueError('named_parameters should consistently be a sequence of tuples (name, torch.Tensor)')
                self._is_tensor_instance = True
                self._parameter_names = {v.__hash__(): k for k, v in sorted(named_parameters)}
                self._tensor_list = [tensor for name, tensor in named_parameters]
            else:
                self._is_tensor_instance = False
                self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        else:
            self._is_tensor_instance = False
            self._parameter_names = {v: ('push_pull.noname.%s' % i) for param_group in self.param_groups for i, v in enumerate(param_group['params'])}
        if size() > 1:
            self._register_hooks()
            named_params = self.module.named_parameters()
            self._num_grads = sum(p.requires_grad for _, p in named_params)
            byteps_torch_set_num_grads(self._num_grads)
        for name in sorted(self._parameter_names.values()):
            declare('Gradient.' + name)
        for name in sorted(self._parameter_names.values()):
            declare('Parameter.' + name)
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            bps.torch.broadcast_parameters(self.module.state_dict(), root_rank=0)

    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()
        return self.module(*inputs, **kwargs)

    def _sync_params(self):
        with torch.no_grad():
            if self.broadcast_buffers and len(self.modules_buffers[0]) > 0:
                bps.torch.broadcast_parameters(list(self.module.named_buffers()), root_rank=0)

    def _register_hooks(self):
        for _, p in self.module.named_parameters():
            if p.requires_grad:
                p.grad = p.data.new(p.size()).zero_()
                self._requires_update.add(p)
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(p, self._num_grads))
                self._grad_accs.append(grad_acc)

    def _push_pull_grad_group_sync(self, p, num_grads_):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        if self._enable_async:
            handle, ctx = None, None
        else:
            tensor = p.grad
            tensor_compressed, ctx = self._compression.compress(tensor)
            handle, grad_count = byteps_push_pull_group(tensor_compressed, average=True, name='Gradient.' + name)
        return handle, ctx, grad_count

    def _push_pull_grad_async(self, p):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        if self._enable_async:
            handle, ctx = None, None
        else:
            tensor = p.grad
            tensor_compressed, ctx = self._compression.compress(tensor)
            handle = byteps_push_pull(tensor_compressed, average=True, name='Gradient.' + name)
        return handle, ctx

    def _make_hook(self, p, num_grads):

        def hook(*ignore):
            handle, ctx = None, None
            handle, ctx, grad_count = self._push_pull_grad_group_sync(p, num_grads)
            self._handles[p] = handle, ctx
            if grad_count == self._num_grads:
                self.synchronize()
        return hook

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx, grad_count = self._push_pull_grad_group_sync(p, self._num_grads)
            self._handles[p] = handle, ctx
        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx, grad_count = self._push_pull_grad_group_sync(p)
                self._handles[p] = handle, ctx
        for p, (handle, _) in self._handles.items():
            output = synchronize(handle)
            if not self._enable_async:
                p.grad.set_(self._compression.decompress(output, ctx))
        self._handles.clear()


class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


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

