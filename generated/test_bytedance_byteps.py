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
compression = _module
ops = _module
server = _module
tensorflow = _module
distribute = _module
cross_device_ops = _module
mirrored_strategy = _module
util = _module
compression = _module
cross_barrier = _module
ops = _module
parallel = _module
distributed = _module
keras_imagenet_resnet50 = _module
keras_mnist = _module
keras_mnist_advanced = _module
keras_synthetic_benchmark_tf2 = _module
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
train_cifar100_byteps_gc = _module
train_gluon_imagenet_byteps_gc = _module
train_gluon_mnist_byteps = _module
train_gluon_mnist_byteps_gc = _module
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
tensorflow2_keras_mnist = _module
tensorflow2_mnist = _module
tensorflow2_mnist_bps_MirroredStrategy = _module
tensorflow_keras_mnist = _module
tensorflow_mnist = _module
dist_launcher = _module
launch = _module
pre_setup = _module
setup = _module
meta_test = _module
test_dithering = _module
test_mxnet = _module
test_onebit = _module
test_randomk = _module
test_tensorflow_keras = _module
test_topk = _module
utils = _module

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


from torch.nn.parallel import DistributedDataParallel as DDP


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

