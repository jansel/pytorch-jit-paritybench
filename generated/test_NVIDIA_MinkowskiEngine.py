import sys
_module = sys.modules[__name__]
del sys
MinkowskiBroadcast = _module
MinkowskiChannelwiseConvolution = _module
MinkowskiCommon = _module
MinkowskiConvolution = _module
MinkowskiCoordinateManager = _module
MinkowskiFunctional = _module
MinkowskiInterpolation = _module
MinkowskiKernelGenerator = _module
MinkowskiNetwork = _module
MinkowskiNonlinearity = _module
MinkowskiNormalization = _module
MinkowskiOps = _module
MinkowskiPooling = _module
MinkowskiPruning = _module
MinkowskiSparseTensor = _module
MinkowskiTensor = _module
MinkowskiTensorField = _module
MinkowskiUnion = _module
MinkowskiEngine = _module
diagnostics = _module
modules = _module
resnet_block = _module
senet_block = _module
sparse_matrix_functions = _module
utils = _module
collation = _module
coords = _module
gradcheck = _module
init = _module
quantization = _module
summary = _module
conf = _module
examples = _module
classification_modelnet40 = _module
common = _module
completion = _module
convolution = _module
example = _module
indoor = _module
minkunet = _module
multigpu = _module
multigpu_ddp = _module
multigpu_lightning = _module
pointnet = _module
reconstruction = _module
resnet = _module
sparse_tensor_basic = _module
stack_unet = _module
training = _module
unet = _module
vae = _module
setup = _module
cpp = _module
convolution_cpu = _module
convolution_cpu_test = _module
convolution_gpu_test = _module
coordinate_map_cpu_test = _module
coordinate_map_gpu_test = _module
coordinate_map_key_test = _module
coordinate_map_manager_cpu_test = _module
coordinate_map_manager_gpu_test = _module
coordinate_test = _module
kernel_region_cpu_test = _module
kernel_region_gpu_test = _module
setup = _module
type_test = _module
utils = _module
python = _module
broadcast = _module
chwise_conv = _module
common = _module
conv_on_coords = _module
convolution = _module
coordinate_manager = _module
dense = _module
direct_pool = _module
interpolation = _module
kernel_map = _module
network_speed = _module
norm = _module
pool = _module
pruning = _module
quantization = _module
sparse_tensor = _module
spmm = _module
stack = _module
strided_conv = _module
summary = _module
tensor_field = _module
union = _module
utility_functions = _module
run_test = _module

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


from typing import Union


import torch


from torch.nn import Module


from torch.autograd import Function


import math


from torch.nn import Parameter


from collections.abc import Sequence


import numpy as np


from typing import List


from typing import Tuple


import warnings


import torch.nn.functional as F


from collections import namedtuple


from functools import reduce


from abc import ABC


from abc import abstractmethod


import torch.nn as nn


from torch.nn.modules import Module


import copy


from enum import Enum


import logging


import collections.abc


from torch.types import _TensorOrTensors


from typing import Callable


from typing import Optional


from torch.autograd.gradcheck import gradcheck as _gradcheck


from torch.autograd import Variable


from collections import OrderedDict


import sklearn.metrics as metrics


import torch.utils.data


from torch.utils.data import DataLoader


import torch.optim as optim


import random


import time


from torch.utils.data.sampler import Sampler


from time import time


from torch.optim import SGD


import torch.nn.parallel as parallel


import torch.multiprocessing as mp


import torch.distributed as dist


from torch.utils.data import Dataset


import re


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.cuda


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import ROCM_HOME


import collections


def convert_to_int_list(arg: Union[int, Sequence, np.ndarray, torch.Tensor], dimension: int):
    if isinstance(arg, list):
        assert len(arg) == dimension
        return arg
    if isinstance(arg, (Sequence, np.ndarray, torch.Tensor)):
        tmp = [i for i in arg]
        assert len(tmp) == dimension
    elif np.isscalar(arg):
        tmp = [int(arg) for i in range(dimension)]
    else:
        raise ValueError('Input must be a scalar or a sequence')
    return tmp


def spmm(rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor, size: torch.Size, mat: torch.Tensor, is_sorted: bool=False, cuda_spmm_alg: int=1) ->torch.Tensor:
    assert len(rows) == len(cols), 'Invalid length'
    assert len(rows) == len(vals), 'Invalid length'
    assert vals.dtype == mat.dtype, 'dtype mismatch'
    assert vals.device == mat.device, 'device mismatch'
    if mat.is_cuda:
        rows = rows.int()
        cols = cols.int()
        result = MEB.coo_spmm_int32(rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg, is_sorted)
    else:
        COO = torch.stack((rows, cols), 0).long()
        torchSparseTensor = None
        if vals.dtype == torch.float64:
            torchSparseTensor = torch.sparse.DoubleTensor
        elif vals.dtype == torch.float32:
            torchSparseTensor = torch.sparse.FloatTensor
        else:
            raise ValueError(f'Unsupported data type: {vals.dtype}')
        sp = torchSparseTensor(COO, vals, size)
        result = sp.matmul(mat)
    return result


def spmm_average(rows: torch.Tensor, cols: torch.Tensor, size: torch.Size, mat: torch.Tensor, cuda_spmm_alg: int=1) ->(torch.Tensor, torch.Tensor, torch.Tensor):
    assert len(rows) == len(cols), 'Invalid length'
    if mat.is_cuda:
        rows = rows.int()
        cols = cols.int()
        result, COO, vals = MEB.coo_spmm_average_int32(rows, cols, size[0], size[1], mat, cuda_spmm_alg)
    else:
        rows, sort_ind = torch.sort(rows)
        cols = cols[sort_ind]
        COO = torch.stack((rows, cols), 0).long()
        _, inverse_ind, counts = torch.unique(rows, return_counts=True, return_inverse=True)
        vals = 1 / counts[inverse_ind]
        torchSparseTensor = None
        if mat.dtype == torch.float64:
            torchSparseTensor = torch.sparse.DoubleTensor
        elif mat.dtype == torch.float32:
            torchSparseTensor = torch.sparse.FloatTensor
        else:
            raise ValueError(f'Unsupported data type: {mat.dtype}')
        sp = torchSparseTensor(COO, vals, size)
        result = sp.matmul(mat)
    return result, COO, vals


class MinkowskiSPMMAverageFunction(Function):

    @staticmethod
    def forward(ctx, rows: torch.Tensor, cols: torch.Tensor, size: torch.Size, mat: torch.Tensor, cuda_spmm_alg: int=1):
        ctx.misc_args = size, cuda_spmm_alg
        result, COO, vals = spmm_average(rows, cols, size, mat, cuda_spmm_alg=cuda_spmm_alg)
        ctx.save_for_backward(COO, vals)
        return result

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        size, cuda_spmm_alg = ctx.misc_args
        COO, vals = ctx.saved_tensors
        new_size = torch.Size([size[1], size[0]])
        grad = spmm(COO[1], COO[0], vals, new_size, grad, is_sorted=False, cuda_spmm_alg=cuda_spmm_alg)
        return None, None, None, grad, None


class MinkowskiSPMMFunction(Function):

    @staticmethod
    def forward(ctx, rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor, size: torch.Size, mat: torch.Tensor, cuda_spmm_alg: int=1):
        ctx.misc_args = size, cuda_spmm_alg
        ctx.save_for_backward(rows, cols, vals)
        result = spmm(rows, cols, vals, size, mat, is_sorted=False, cuda_spmm_alg=cuda_spmm_alg)
        return result

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        size, cuda_spmm_alg = ctx.misc_args
        rows, cols, vals = ctx.saved_tensors
        new_size = torch.Size([size[1], size[0]])
        grad = spmm(cols, rows, vals, new_size, grad, is_sorted=False, cuda_spmm_alg=cuda_spmm_alg)
        return None, None, None, None, grad, None


class SparseTensorOperationMode(Enum):
    """Enum class for SparseTensor internal instantiation modes.

    :attr:`SEPARATE_COORDINATE_MANAGER`: always create a new coordinate manager.

    :attr:`SHARE_COORDINATE_MANAGER`: always use the globally defined coordinate
    manager. Must clear the coordinate manager manually by
    :attr:`MinkowskiEngine.SparseTensor.clear_global_coordinate_manager`.

    """
    SEPARATE_COORDINATE_MANAGER = 0
    SHARE_COORDINATE_MANAGER = 1


class SparseTensorQuantizationMode(Enum):
    """
    `RANDOM_SUBSAMPLE`: Subsample one coordinate per each quantization block randomly.
    `UNWEIGHTED_AVERAGE`: average all features within a quantization block equally.
    `UNWEIGHTED_SUM`: sum all features within a quantization block equally.
    `NO_QUANTIZATION`: No quantization is applied. Should not be used for normal operation.
    `MAX_POOL`: Voxel-wise max pooling is applied.
    `SPLAT_LINEAR_INTERPOLATION`: Splat features using N-dimensional linear interpolation to 2^N neighbors.
    """
    RANDOM_SUBSAMPLE = 0
    UNWEIGHTED_AVERAGE = 1
    UNWEIGHTED_SUM = 2
    NO_QUANTIZATION = 3
    MAX_POOL = 4
    SPLAT_LINEAR_INTERPOLATION = 5


StrideType = Union[int, Sequence, np.ndarray, torch.IntTensor]


def global_coordinate_manager():
    """Return the current global coordinate manager"""
    global _global_coordinate_manager
    return _global_coordinate_manager


def set_global_coordinate_manager(coordinate_manager):
    """Set the global coordinate manager.

    :attr:`MinkowskiEngine.CoordinateManager` The coordinate manager which will
    be set to the global coordinate manager.
    """
    global _global_coordinate_manager
    _global_coordinate_manager = coordinate_manager


def sparse_tensor_operation_mode() ->SparseTensorOperationMode:
    """Return the current sparse tensor operation mode."""
    global _sparse_tensor_operation_mode
    return copy.deepcopy(_sparse_tensor_operation_mode)


class MinkowskiModuleBase(Module):
    pass


def get_kernel_volume(region_type, kernel_size, region_offset, axis_types, dimension):
    """
    when center is True, the custom region_offset will be centered at the
    origin. Currently, for HYPER_CUBE, HYPER_CROSS with odd kernel sizes cannot
    use center=False.
    """
    if region_type == RegionType.HYPER_CUBE:
        assert reduce(lambda k1, k2: k1 > 0 and k2 > 0, kernel_size), 'kernel_size must be positive'
        assert region_offset is None, 'Region offset must be None when region_type is given'
        assert axis_types is None, 'Axis types must be None when region_type is given'
        kernel_volume = torch.prod(torch.IntTensor(kernel_size)).item()
    elif region_type == RegionType.HYPER_CROSS:
        assert reduce(lambda k1, k2: k1 > 0 and k2 > 0, kernel_size), 'kernel_size must be positive'
        assert (torch.IntTensor(kernel_size) % 2).prod().item() == 1, 'kernel_size must be odd for region_type HYPER_CROSS'
        kernel_volume = (torch.sum(torch.IntTensor(kernel_size) - 1) + 1).item()
    elif region_type == RegionType.CUSTOM:
        assert region_offset.numel() > 0, 'region_offset must be non empty when region_type is CUSTOM'
        assert region_offset.size(1) == dimension, 'region_offset must have the same dimension as the network'
        kernel_volume = int(region_offset.size(0))
    else:
        raise NotImplementedError()
    return kernel_volume


def get_postfix(tensor: torch.Tensor):
    postfix = 'GPU' if tensor.is_cuda else 'CPU'
    return postfix


def get_minkowski_function(name, variable):
    fn_name = name + get_postfix(variable)
    if hasattr(MEB, fn_name):
        return getattr(MEB, fn_name)
    elif variable.is_cuda:
        raise ValueError(f'Function {fn_name} not available. Please compile MinkowskiEngine with `torch.cuda.is_available()` is `True`.')
    else:
        raise ValueError(f'Function {fn_name} not available.')


class MinkowskiNetwork(nn.Module, ABC):
    """
    MinkowskiNetwork: an abstract class for sparse convnets.

    Note: All modules that use the same coordinates must use the same net_metadata
    """

    def __init__(self, D):
        super(MinkowskiNetwork, self).__init__()
        self.D = D

    @abstractmethod
    def forward(self, x):
        pass

    def init(self, x):
        """
        Initialize coordinates if it does not exist
        """
        nrows = self.get_nrows(1)
        if nrows < 0:
            if isinstance(x, SparseTensor):
                self.initialize_coords(x.coords_man)
            else:
                raise ValueError('Initialize input coordinates')
        elif nrows != x.F.size(0):
            raise ValueError('Input size does not match the coordinate size')


COORDINATE_KEY_DIFFERENT_ERROR = 'SparseTensors must have the same coordinate_map_key.'


COORDINATE_MANAGER_DIFFERENT_ERROR = 'SparseTensors must share the same coordinate manager for this operation. Please refer to the SparseTensor creation API (https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html) to share the coordinate manager, or set the sparse tensor operation mode with `set_sparse_tensor_operation_mode` to share it by default.'


class MinkowskiDirectMaxPoolingFunction(Function):

    @staticmethod
    def forward(ctx, in_map: torch.Tensor, out_map: torch.Tensor, in_feat: torch.Tensor, out_nrows: int, is_sorted: bool=False):
        out_feat, max_mask = _C.direct_max_pool_fw(in_map, out_map, in_feat, out_nrows, is_sorted)
        ctx.in_nrows = in_feat.size(0)
        ctx.save_for_backward(max_mask)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_out_feat = grad_out_feat.contiguous()
        max_mask = ctx.saved_tensors[0]
        grad = _C.direct_max_pool_bw(grad_out_feat, max_mask, ctx.in_nrows)
        return None, None, grad, None, None


def create_splat_coordinates(coordinates: torch.Tensor) ->torch.Tensor:
    """Create splat coordinates. splat coordinates could have duplicate coordinates."""
    dimension = coordinates.shape[1] - 1
    region_offset = [[0] * (dimension + 1)]
    for d in reversed(range(1, dimension + 1)):
        new_offset = []
        for offset in region_offset:
            offset = offset.copy()
            offset[d] = 1
            new_offset.append(offset)
        region_offset.extend(new_offset)
    region_offset = torch.IntTensor(region_offset)
    coordinates = torch.floor(coordinates).int().unsqueeze(1) + region_offset.unsqueeze(0)
    return coordinates.reshape(-1, dimension + 1)


class MinkowskiNonlinearityBase(MinkowskiModuleBase):
    MODULE = None

    def __init__(self, *args, **kwargs):
        super(MinkowskiNonlinearityBase, self).__init__()
        self.module = self.MODULE(*args, **kwargs)

    def forward(self, input):
        output = self.module(input.F)
        if isinstance(input, TensorField):
            return TensorField(output, coordinate_field_map_key=input.coordinate_field_map_key, coordinate_manager=input.coordinate_manager, quantization_mode=input.quantization_mode)
        else:
            return SparseTensor(output, coordinate_map_key=input.coordinate_map_key, coordinate_manager=input.coordinate_manager)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinkowskiELU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.ELU


class MinkowskiHardshrink(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Hardshrink


class MinkowskiHardsigmoid(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Hardsigmoid


class MinkowskiHardtanh(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Hardtanh


class MinkowskiHardswish(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Hardswish


class MinkowskiLeakyReLU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.LeakyReLU


class MinkowskiLogSigmoid(MinkowskiNonlinearityBase):
    MODULE = torch.nn.LogSigmoid


class MinkowskiPReLU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.PReLU


class MinkowskiReLU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.ReLU


class MinkowskiReLU6(MinkowskiNonlinearityBase):
    MODULE = torch.nn.ReLU6


class MinkowskiRReLU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.RReLU


class MinkowskiSELU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.SELU


class MinkowskiCELU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.CELU


class MinkowskiGELU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.GELU


class MinkowskiSigmoid(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Sigmoid


class MinkowskiSiLU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.SiLU


class MinkowskiSoftplus(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Softplus


class MinkowskiSoftshrink(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Softshrink


class MinkowskiSoftsign(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Softsign


class MinkowskiTanh(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Tanh


class MinkowskiTanhshrink(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Tanhshrink


class MinkowskiThreshold(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Threshold


class MinkowskiSoftmin(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Softmin


class MinkowskiSoftmax(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Softmax


class MinkowskiLogSoftmax(MinkowskiNonlinearityBase):
    MODULE = torch.nn.LogSoftmax


class MinkowskiAdaptiveLogSoftmaxWithLoss(MinkowskiNonlinearityBase):
    MODULE = torch.nn.AdaptiveLogSoftmaxWithLoss


class MinkowskiDropout(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Dropout


class MinkowskiAlphaDropout(MinkowskiNonlinearityBase):
    MODULE = torch.nn.AlphaDropout


class MinkowskiBatchNorm(Module):
    """A batch normalization layer for a sparse tensor.

    See the pytorch :attr:`torch.nn.BatchNorm1d` for more details.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MinkowskiBatchNorm, self).__init__()
        self.bn = torch.nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, input):
        output = self.bn(input.F)
        if isinstance(input, TensorField):
            return TensorField(output, coordinate_field_map_key=input.coordinate_field_map_key, coordinate_manager=input.coordinate_manager, quantization_mode=input.quantization_mode)
        else:
            return SparseTensor(output, coordinate_map_key=input.coordinate_map_key, coordinate_manager=input.coordinate_manager)

    def __repr__(self):
        s = '({}, eps={}, momentum={}, affine={}, track_running_stats={})'.format(self.bn.num_features, self.bn.eps, self.bn.momentum, self.bn.affine, self.bn.track_running_stats)
        return self.__class__.__name__ + s


class MinkowskiSyncBatchNorm(MinkowskiBatchNorm):
    """A batch normalization layer with multi GPU synchronization."""

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None):
        Module.__init__(self)
        self.bn = torch.nn.SyncBatchNorm(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, process_group=process_group)

    def forward(self, input):
        output = self.bn(input.F)
        if isinstance(input, TensorField):
            return TensorField(output, coordinate_field_map_key=input.coordinate_field_map_key, coordinate_manager=input.coordinate_manager, quantization_mode=input.quantization_mode)
        else:
            return SparseTensor(output, coordinate_map_key=input.coordinate_map_key, coordinate_manager=input.coordinate_manager)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        """Helper function to convert
        :attr:`MinkowskiEngine.MinkowskiBatchNorm` layer in the model to
        :attr:`MinkowskiEngine.MinkowskiSyncBatchNorm` layer.

        Args:
            module (nn.Module): containing module
            process_group (optional): process group to scope synchronization,
            default is the whole world

        Returns:
            The original module with the converted
            :attr:`MinkowskiEngine.MinkowskiSyncBatchNorm` layer

        Example::

            >>> # Network with MinkowskiBatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            MinkowskiLinear(20, 100),
            >>>            MinkowskiBatchNorm1d(100)
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # process_ids is a list of int identifying rank ids.
            >>> process_group = torch.distributed.new_group(process_ids)
            >>> sync_bn_module = convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, MinkowskiBatchNorm):
            module_output = MinkowskiSyncBatchNorm(module.bn.num_features, module.bn.eps, module.bn.momentum, module.bn.affine, module.bn.track_running_stats, process_group)
            if module.bn.affine:
                with torch.no_grad():
                    module_output.bn.weight = module.bn.weight
                    module_output.bn.bias = module.bn.bias
            module_output.bn.running_mean = module.bn.running_mean
            module_output.bn.running_var = module.bn.running_var
            module_output.bn.num_batches_tracked = module.bn.num_batches_tracked
            if hasattr(module, 'qconfig'):
                module_output.bn.qconfig = module.bn.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output


class MinkowskiStableInstanceNorm(MinkowskiModuleBase):

    def __init__(self, num_features):
        Module.__init__(self)
        self.num_features = num_features
        self.eps = 1e-06
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_in = MinkowskiGlobalAvgPooling()
        self.glob_sum = MinkowskiBroadcastAddition()
        self.glob_sum2 = MinkowskiBroadcastAddition()
        self.glob_mean = MinkowskiGlobalAvgPooling()
        self.glob_times = MinkowskiBroadcastMultiplication()
        self.reset_parameters()

    def __repr__(self):
        s = f'(nchannels={self.num_features})'
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        neg_mean_in = self.mean_in(SparseTensor(-x.F, coords_key=x.coords_key, coords_manager=x.coords_man))
        centered_in = self.glob_sum(x, neg_mean_in)
        temp = SparseTensor(centered_in.F ** 2, coordinate_map_key=centered_in.coordinate_map_key, coordinate_manager=centered_in.coordinate_manager)
        var_in = self.glob_mean(temp)
        instd_in = SparseTensor(1 / (var_in.F + self.eps).sqrt(), coordinate_map_key=var_in.coordinate_map_key, coordinate_manager=var_in.coordinate_manager)
        x = self.glob_times(self.glob_sum2(x, neg_mean_in), instd_in)
        return SparseTensor(x.F * self.weight + self.bias, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)


def to_sparse(x: torch.Tensor, format: str=None, coordinates=None, device=None):
    """Convert a batched tensor (dimension 0 is the batch dimension) to a SparseTensor

    :attr:`x` (:attr:`torch.Tensor`): a batched tensor. The first dimension is the batch dimension.

    :attr:`format` (:attr:`str`): Format of the tensor. It must include 'B' and 'C' indicating the batch and channel dimension respectively. The rest of the dimensions must be 'X'. .e.g. format="BCXX" if image data with BCHW format is used. If a 3D data with the channel at the last dimension, use format="BXXXC" indicating Batch X Height X Width X Depth X Channel. If not provided, the format will be "BCX...X".

    :attr:`device`: Device the sparse tensor will be generated on. If not provided, the device of the input tensor will be used.

    """
    assert x.ndim > 2, 'Input has 0 spatial dimension.'
    assert isinstance(x, torch.Tensor)
    if format is None:
        format = ['X'] * x.ndim
        format[0] = 'B'
        format[1] = 'C'
        format = ''.join(format)
    assert x.ndim == len(format), f'Invalid format: {format}. len(format) != x.ndim'
    assert 'B' in format and 'B' == format[0] and format.count('B') == 1, "The input must have the batch axis and the format must include 'B' indicating the batch axis."
    assert 'C' in format and format.count('C') == 1, 'The format must indicate the channel axis'
    if device is None:
        device = x.device
    ch_dim = format.find('C')
    reduced_x = torch.abs(x).sum(ch_dim)
    bcoords = torch.where(reduced_x != 0)
    stacked_bcoords = torch.stack(bcoords, dim=1).int()
    indexing = [f'bcoords[{i}]' for i in range(len(bcoords))]
    indexing.insert(ch_dim, ':')
    features = torch.zeros((len(stacked_bcoords), x.size(ch_dim)), dtype=x.dtype, device=x.device)
    exec('features[:] = x[' + ', '.join(indexing) + ']')
    return SparseTensor(features=features, coordinates=stacked_bcoords, device=device)


def dense_coordinates(shape: Union[list, torch.Size]):
    """
    coordinates = dense_coordinates(tensor.shape)
    """
    """
    Assume the input to have BxCxD1xD2x....xDN format.

    If the shape of the tensor do not change, use 
    """
    spatial_dim = len(shape) - 2
    assert spatial_dim > 0, 'Invalid shape. Shape must be batch x channel x spatial dimensions.'
    size = [i for i in shape]
    B = size[0]
    coordinates = torch.from_numpy(np.stack([s.reshape(-1) for s in np.meshgrid(np.linspace(0, B - 1, B), *(np.linspace(0, s - 1, s) for s in size[2:]), indexing='ij')], 1)).int()
    return coordinates


def to_sparse_all(dense_tensor: torch.Tensor, coordinates: torch.Tensor=None):
    """Converts a (differentiable) dense tensor to a sparse tensor with all coordinates.

    Assume the input to have BxCxD1xD2x....xDN format.

    If the shape of the tensor do not change, use `dense_coordinates` to cache the coordinates.
    Please refer to tests/python/dense.py for usage

    Example::

       >>> dense_tensor = torch.rand(3, 4, 5, 6, 7, 8)  # BxCxD1xD2xD3xD4
       >>> dense_tensor.requires_grad = True
       >>> stensor = to_sparse(dense_tensor)

    """
    spatial_dim = dense_tensor.ndim - 2
    assert spatial_dim > 0, 'Invalid shape. Shape must be batch x channel x spatial dimensions.'
    if coordinates is None:
        coordinates = dense_coordinates(dense_tensor.shape)
    feat_tensor = dense_tensor.permute(0, *(2 + i for i in range(spatial_dim)), 1)
    return SparseTensor(feat_tensor.reshape(-1, dense_tensor.size(1)), coordinates, device=dense_tensor.device)


def _tuple_operator(*sparse_tensors, operator):
    if len(sparse_tensors) == 1:
        assert isinstance(sparse_tensors[0], (tuple, list))
        sparse_tensors = sparse_tensors[0]
    assert len(sparse_tensors) > 1, f'Invalid number of inputs. The input must be at least two len(sparse_tensors) > 1'
    if isinstance(sparse_tensors[0], SparseTensor):
        device = sparse_tensors[0].device
        coordinate_manager = sparse_tensors[0].coordinate_manager
        coordinate_map_key = sparse_tensors[0].coordinate_map_key
        for s in sparse_tensors:
            assert isinstance(s, SparseTensor), 'Inputs must be either SparseTensors or TensorFields.'
            assert device == s.device, f'Device must be the same. {device} != {s.device}'
            assert coordinate_manager == s.coordinate_manager, COORDINATE_MANAGER_DIFFERENT_ERROR
            assert coordinate_map_key == s.coordinate_map_key, COORDINATE_KEY_DIFFERENT_ERROR + str(coordinate_map_key) + ' != ' + str(s.coordinate_map_key)
        tens = []
        for s in sparse_tensors:
            tens.append(s.F)
        return SparseTensor(operator(tens), coordinate_map_key=coordinate_map_key, coordinate_manager=coordinate_manager)
    elif isinstance(sparse_tensors[0], TensorField):
        device = sparse_tensors[0].device
        coordinate_manager = sparse_tensors[0].coordinate_manager
        coordinate_field_map_key = sparse_tensors[0].coordinate_field_map_key
        for s in sparse_tensors:
            assert isinstance(s, TensorField), 'Inputs must be either SparseTensors or TensorFields.'
            assert device == s.device, f'Device must be the same. {device} != {s.device}'
            assert coordinate_manager == s.coordinate_manager, COORDINATE_MANAGER_DIFFERENT_ERROR
            assert coordinate_field_map_key == s.coordinate_field_map_key, COORDINATE_KEY_DIFFERENT_ERROR + str(coordinate_field_map_key) + ' != ' + str(s.coordinate_field_map_key)
        tens = []
        for s in sparse_tensors:
            tens.append(s.F)
        return TensorField(operator(tens), coordinate_field_map_key=coordinate_field_map_key, coordinate_manager=coordinate_manager)
    else:
        raise ValueError('Invalid data type. The input must be either a list of sparse tensors or a list of tensor fields.')


def cat(*sparse_tensors):
    """Concatenate sparse tensors

    Concatenate sparse tensor features. All sparse tensors must have the same
    `coordinate_map_key` (the same coordinates). To concatenate sparse tensors
    with different sparsity patterns, use SparseTensor binary operations, or
    :attr:`MinkowskiEngine.MinkowskiUnion`.

    Example::

       >>> import MinkowskiEngine as ME
       >>> sin = ME.SparseTensor(feats, coords)
       >>> sin2 = ME.SparseTensor(feats2, coordinate_map_key=sin.coordinate_map_key, coordinate_mananger=sin.coordinate_manager)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.cat(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """
    return _tuple_operator(*sparse_tensors, operator=lambda xs: torch.cat(xs, dim=-1))


class MinkowskiStackCat(torch.nn.Sequential):

    def forward(self, x):
        return cat([module(x) for module in self])


def _sum(*sparse_tensors):
    """Compute the sum of sparse tensor features

    Sum all sparse tensor features. All sparse tensors must have the same
    `coordinate_map_key` (the same coordinates). To sum sparse tensors with
    different sparsity patterns, use SparseTensor binary operations, or
    :attr:`MinkowskiEngine.MinkowskiUnion`.

    Example::

       >>> import MinkowskiEngine as ME
       >>> sin = ME.SparseTensor(feats, coords)
       >>> sin2 = ME.SparseTensor(feats2, coordinate_map_key=sin.coordinate_map_key, coordinate_manager=sin.coordinate_manager)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.sum(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """

    def return_sum(xs):
        tmp = xs[0] + xs[1]
        for x in xs[2:]:
            tmp += x
        return tmp
    return _tuple_operator(*sparse_tensors, operator=lambda xs: return_sum(xs))


class MinkowskiStackSum(torch.nn.Sequential):

    def forward(self, x):
        return _sum([module(x) for module in self])


def mean(*sparse_tensors):
    """Compute the average of sparse tensor features

    Sum all sparse tensor features. All sparse tensors must have the same
    `coordinate_map_key` (the same coordinates). To sum sparse tensors with
    different sparsity patterns, use SparseTensor binary operations, or
    :attr:`MinkowskiEngine.MinkowskiUnion`.

    Example::

       >>> import MinkowskiEngine as ME
       >>> sin = ME.SparseTensor(feats, coords)
       >>> sin2 = ME.SparseTensor(feats2, coordinate_map_key=sin.coordinate_map_key, coordinate_manager=sin.coordinate_manager)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.mean(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """

    def return_mean(xs):
        tmp = xs[0] + xs[1]
        for x in xs[2:]:
            tmp += x
        return tmp / len(xs)
    return _tuple_operator(*sparse_tensors, operator=lambda xs: return_mean(xs))


class MinkowskiStackMean(torch.nn.Sequential):

    def forward(self, x):
        return mean([module(x) for module in self])


def var(*sparse_tensors):
    """Compute the variance of sparse tensor features

    Sum all sparse tensor features. All sparse tensors must have the same
    `coordinate_map_key` (the same coordinates). To sum sparse tensors with
    different sparsity patterns, use SparseTensor binary operations, or
    :attr:`MinkowskiEngine.MinkowskiUnion`.

    Example::

       >>> import MinkowskiEngine as ME
       >>> sin = ME.SparseTensor(feats, coords)
       >>> sin2 = ME.SparseTensor(feats2, coordinate_map_key=sin.coordinate_map_key, coordinate_manager=sin.coordinate_manager)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.var(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """

    def return_var(xs):
        tmp = xs[0] + xs[1]
        for x in xs[2:]:
            tmp += x
        mean = tmp / len(xs)
        var = (xs[0] - mean) ** 2
        for x in xs[1:]:
            var += (x - mean) ** 2
        return var / len(xs)
    return _tuple_operator(*sparse_tensors, operator=lambda xs: return_var(xs))


class MinkowskiStackVar(torch.nn.Sequential):

    def forward(self, x):
        return var([module(x) for module in self])


class MinkowskiUnion(Module):
    """Create a union of all sparse tensors and add overlapping features.

    Args:
        None

    .. warning::
       This function is experimental and the usage can be changed in the future updates.

    """

    def __init__(self):
        super(MinkowskiUnion, self).__init__()
        self.union = MinkowskiUnionFunction()

    def forward(self, *inputs):
        """
        Args:
            A variable number of :attr:`MinkowskiEngine.SparseTensor`'s.

        Returns:
            A :attr:`MinkowskiEngine.SparseTensor` with coordinates = union of all
            input coordinates, and features = sum of all features corresponding to the
            coordinate.

        Example::

            >>> # Define inputs
            >>> input1 = SparseTensor(
            >>>     torch.rand(N, in_channels, dtype=torch.double), coords=coords)
            >>> # All inputs must share the same coordinate manager
            >>> input2 = SparseTensor(
            >>>     torch.rand(N, in_channels, dtype=torch.double),
            >>>     coords=coords + 1,
            >>>     coords_manager=input1.coordinate_manager,  # Must use same coords manager
            >>>     force_creation=True  # The tensor stride [1, 1] already exists.
            >>> )
            >>> union = MinkowskiUnion()
            >>> output = union(input1, iput2)

        """
        assert isinstance(inputs, (list, tuple)), 'The input must be a list or tuple'
        for s in inputs:
            assert isinstance(s, SparseTensor), 'Inputs must be sparse tensors.'
        assert len(inputs) > 1, 'input must be a set with at least 2 SparseTensors'
        ref_coordinate_manager = inputs[0].coordinate_manager
        for s in inputs:
            assert ref_coordinate_manager == s.coordinate_manager, 'Invalid coordinate manager. All inputs must have the same coordinate manager.'
        in_coordinate_map_key = inputs[0].coordinate_map_key
        coordinate_manager = inputs[0].coordinate_manager
        out_coordinate_map_key = CoordinateMapKey(in_coordinate_map_key.get_coordinate_size())
        output = self.union.apply([input.coordinate_map_key for input in inputs], out_coordinate_map_key, coordinate_manager, *[input.F for input in inputs])
        return SparseTensor(output, coordinate_map_key=out_coordinate_map_key, coordinate_manager=coordinate_manager)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(planes, planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(planes * self.expansion, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16, D=-1):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(ME.MinkowskiLinear(channel, channel // reduction), ME.MinkowskiReLU(inplace=True), ME.MinkowskiLinear(channel // reduction, channel), ME.MinkowskiSigmoid())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class SEBasicBlock(BasicBlock):

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, reduction=16, D=-1):
        super(SEBasicBlock, self).__init__(inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, D=D)
        self.se = SELayer(planes, reduction=reduction, D=D)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(Bottleneck):

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, D=3, reduction=16):
        super(SEBottleneck, self).__init__(inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, D=D)
        self.se = SELayer(planes * self.expansion, reduction=reduction, D=D)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class GlobalMaxAvgPool(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, tensor):
        x = self.global_max_pool(tensor)
        y = self.global_avg_pool(tensor)
        return ME.cat(x, y)


class CompletionNet(nn.Module):
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self, resolution, in_nchannel=512):
        nn.Module.__init__(self)
        self.resolution = resolution
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS
        self.enc_block_s1 = nn.Sequential(ME.MinkowskiConvolution(1, enc_ch[0], kernel_size=3, stride=1, dimension=3), ME.MinkowskiBatchNorm(enc_ch[0]), ME.MinkowskiELU())
        self.enc_block_s1s2 = nn.Sequential(ME.MinkowskiConvolution(enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(enc_ch[1]), ME.MinkowskiELU(), ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(enc_ch[1]), ME.MinkowskiELU())
        self.enc_block_s2s4 = nn.Sequential(ME.MinkowskiConvolution(enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(enc_ch[2]), ME.MinkowskiELU(), ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(enc_ch[2]), ME.MinkowskiELU())
        self.enc_block_s4s8 = nn.Sequential(ME.MinkowskiConvolution(enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(enc_ch[3]), ME.MinkowskiELU(), ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(enc_ch[3]), ME.MinkowskiELU())
        self.enc_block_s8s16 = nn.Sequential(ME.MinkowskiConvolution(enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(enc_ch[4]), ME.MinkowskiELU(), ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(enc_ch[4]), ME.MinkowskiELU())
        self.enc_block_s16s32 = nn.Sequential(ME.MinkowskiConvolution(enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(enc_ch[5]), ME.MinkowskiELU(), ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(enc_ch[5]), ME.MinkowskiELU())
        self.enc_block_s32s64 = nn.Sequential(ME.MinkowskiConvolution(enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(enc_ch[6]), ME.MinkowskiELU(), ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(enc_ch[6]), ME.MinkowskiELU())
        self.dec_block_s64s32 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(enc_ch[6], dec_ch[5], kernel_size=4, stride=2, dimension=3), ME.MinkowskiBatchNorm(dec_ch[5]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[5]), ME.MinkowskiELU())
        self.dec_s32_cls = ME.MinkowskiConvolution(dec_ch[5], 1, kernel_size=1, bias=True, dimension=3)
        self.dec_block_s32s16 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(enc_ch[5], dec_ch[4], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(dec_ch[4]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[4]), ME.MinkowskiELU())
        self.dec_s16_cls = ME.MinkowskiConvolution(dec_ch[4], 1, kernel_size=1, bias=True, dimension=3)
        self.dec_block_s16s8 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(dec_ch[4], dec_ch[3], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(dec_ch[3]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[3]), ME.MinkowskiELU())
        self.dec_s8_cls = ME.MinkowskiConvolution(dec_ch[3], 1, kernel_size=1, bias=True, dimension=3)
        self.dec_block_s8s4 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(dec_ch[3], dec_ch[2], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(dec_ch[2]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[2]), ME.MinkowskiELU())
        self.dec_s4_cls = ME.MinkowskiConvolution(dec_ch[2], 1, kernel_size=1, bias=True, dimension=3)
        self.dec_block_s4s2 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(dec_ch[2], dec_ch[1], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(dec_ch[1]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[1]), ME.MinkowskiELU())
        self.dec_s2_cls = ME.MinkowskiConvolution(dec_ch[1], 1, kernel_size=1, bias=True, dimension=3)
        self.dec_block_s2s1 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(dec_ch[1], dec_ch[0], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(dec_ch[0]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[0]), ME.MinkowskiELU())
        self.dec_s1_cls = ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=3)
        self.pruning = ME.MinkowskiPruning()

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(target_key, out.tensor_stride[0])
            kernel_map = cm.kernel_map(out.coordinate_map_key, strided_target_key, kernel_size=kernel_size, region_type=1)
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, partial_in, target_key):
        out_cls, targets = [], []
        enc_s1 = self.enc_block_s1(partial_in)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)
        dec_s32 = self.dec_block_s64s32(enc_s64)
        dec_s32 = dec_s32 + enc_s32
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32_cls.F > 0).squeeze()
        target = self.get_target(dec_s32, target_key)
        targets.append(target)
        out_cls.append(dec_s32_cls)
        if self.training:
            keep_s32 += target
        dec_s32 = self.pruning(dec_s32, keep_s32)
        dec_s16 = self.dec_block_s32s16(dec_s32)
        dec_s16 = dec_s16 + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()
        target = self.get_target(dec_s16, target_key)
        targets.append(target)
        out_cls.append(dec_s16_cls)
        if self.training:
            keep_s16 += target
        dec_s16 = self.pruning(dec_s16, keep_s16)
        dec_s8 = self.dec_block_s16s8(dec_s16)
        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)
        target = self.get_target(dec_s8, target_key)
        targets.append(target)
        out_cls.append(dec_s8_cls)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()
        if self.training:
            keep_s8 += target
        dec_s8 = self.pruning(dec_s8, keep_s8)
        dec_s4 = self.dec_block_s8s4(dec_s8)
        dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)
        target = self.get_target(dec_s4, target_key)
        targets.append(target)
        out_cls.append(dec_s4_cls)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()
        if self.training:
            keep_s4 += target
        dec_s4 = self.pruning(dec_s4, keep_s4)
        dec_s2 = self.dec_block_s4s2(dec_s4)
        dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)
        target = self.get_target(dec_s2, target_key)
        targets.append(target)
        out_cls.append(dec_s2_cls)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()
        if self.training:
            keep_s2 += target
        dec_s2 = self.pruning(dec_s2, keep_s2)
        dec_s1 = self.dec_block_s2s1(dec_s2)
        dec_s1_cls = self.dec_s1_cls(dec_s1)
        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)
        target = self.get_target(dec_s1, target_key)
        targets.append(target)
        out_cls.append(dec_s1_cls)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()
        dec_s1 = self.pruning(dec_s1, keep_s1)
        return out_cls, targets, dec_s1


class DummyNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.net = nn.Sequential(ME.MinkowskiConvolution(in_channels, 32, 3, dimension=D), ME.MinkowskiBatchNorm(32), ME.MinkowskiReLU(), ME.MinkowskiConvolution(32, 64, 3, stride=2, dimension=D), ME.MinkowskiBatchNorm(64), ME.MinkowskiReLU(), ME.MinkowskiConvolutionTranspose(64, 32, 3, stride=2, dimension=D), ME.MinkowskiBatchNorm(32), ME.MinkowskiReLU(), ME.MinkowskiConvolution(32, out_channels, kernel_size=1, dimension=D))

    def forward(self, x):
        return self.net(x)


class PointNet(nn.Module):

    def __init__(self, in_channel, out_channel, embedding_channel=1024):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embedding_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embedding_channel)
        self.linear1 = nn.Linear(embedding_channel, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, out_channel, bias=True)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class GenerativeNet(nn.Module):
    CHANNELS = [1024, 512, 256, 128, 64, 32, 16]

    def __init__(self, resolution, in_nchannel=512):
        nn.Module.__init__(self)
        self.resolution = resolution
        ch = self.CHANNELS
        self.block1 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(in_nchannel, ch[0], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU(), ME.MinkowskiGenerativeConvolutionTranspose(ch[0], ch[1], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU())
        self.block1_cls = ME.MinkowskiConvolution(ch[1], 1, kernel_size=1, bias=True, dimension=3)
        self.block2 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[1], ch[2], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU())
        self.block2_cls = ME.MinkowskiConvolution(ch[2], 1, kernel_size=1, bias=True, dimension=3)
        self.block3 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[2], ch[3], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU())
        self.block3_cls = ME.MinkowskiConvolution(ch[3], 1, kernel_size=1, bias=True, dimension=3)
        self.block4 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[3], ch[4], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU())
        self.block4_cls = ME.MinkowskiConvolution(ch[4], 1, kernel_size=1, bias=True, dimension=3)
        self.block5 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[4], ch[5], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU())
        self.block5_cls = ME.MinkowskiConvolution(ch[5], 1, kernel_size=1, bias=True, dimension=3)
        self.block6 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[5], ch[6], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU())
        self.block6_cls = ME.MinkowskiConvolution(ch[6], 1, kernel_size=1, bias=True, dimension=3)
        self.pruning = ME.MinkowskiPruning()

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])
        kernel_map = cm.kernel_map(out.coordinate_map_key, strided_target_key, kernel_size=kernel_size, region_type=1)
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z, target_key):
        out_cls, targets = [], []
        out1 = self.block1(z)
        out1_cls = self.block1_cls(out1)
        target = self.get_target(out1, target_key)
        targets.append(target)
        out_cls.append(out1_cls)
        keep1 = (out1_cls.F > 0).squeeze()
        if self.training:
            keep1 += target
        out1 = self.pruning(out1, keep1)
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).squeeze()
        if self.training:
            keep2 += target
        out2 = self.pruning(out2, keep2)
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).squeeze()
        if self.training:
            keep3 += target
        out3 = self.pruning(out3, keep3)
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).squeeze()
        if self.training:
            keep4 += target
        out4 = self.pruning(out4, keep4)
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        target = self.get_target(out5, target_key)
        targets.append(target)
        out_cls.append(out5_cls)
        keep5 = (out5_cls.F > 0).squeeze()
        if self.training:
            keep5 += target
        out5 = self.pruning(out5, keep5)
        out6 = self.block6(out5)
        out6_cls = self.block6_cls(out6)
        target = self.get_target(out6, target_key)
        targets.append(target)
        out_cls.append(out6_cls)
        keep6 = (out6_cls.F > 0).squeeze()
        out6 = self.pruning(out6, keep6)
        return out_cls, targets, out6


class Encoder(nn.Module):
    CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self):
        nn.Module.__init__(self)
        ch = self.CHANNELS
        self.block1 = nn.Sequential(ME.MinkowskiConvolution(1, ch[0], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU())
        self.block2 = nn.Sequential(ME.MinkowskiConvolution(ch[0], ch[1], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU())
        self.block3 = nn.Sequential(ME.MinkowskiConvolution(ch[1], ch[2], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU())
        self.block4 = nn.Sequential(ME.MinkowskiConvolution(ch[2], ch[3], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU())
        self.block5 = nn.Sequential(ME.MinkowskiConvolution(ch[3], ch[4], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU())
        self.block6 = nn.Sequential(ME.MinkowskiConvolution(ch[4], ch[5], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU())
        self.block7 = nn.Sequential(ME.MinkowskiConvolution(ch[5], ch[6], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU())
        self.global_pool = ME.MinkowskiGlobalPooling()
        self.linear_mean = ME.MinkowskiLinear(ch[6], ch[6], bias=True)
        self.linear_log_var = ME.MinkowskiLinear(ch[6], ch[6], bias=True)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, sinput):
        out = self.block1(sinput)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.global_pool(out)
        mean = self.linear_mean(out)
        log_var = self.linear_log_var(out)
        return mean, log_var


class Decoder(nn.Module):
    CHANNELS = [1024, 512, 256, 128, 64, 32, 16]
    resolution = 128

    def __init__(self):
        nn.Module.__init__(self)
        ch = self.CHANNELS
        self.block1 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[0], ch[0], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU(), ME.MinkowskiGenerativeConvolutionTranspose(ch[0], ch[1], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU())
        self.block1_cls = ME.MinkowskiConvolution(ch[1], 1, kernel_size=1, bias=True, dimension=3)
        self.block2 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[1], ch[2], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU())
        self.block2_cls = ME.MinkowskiConvolution(ch[2], 1, kernel_size=1, bias=True, dimension=3)
        self.block3 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[2], ch[3], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU())
        self.block3_cls = ME.MinkowskiConvolution(ch[3], 1, kernel_size=1, bias=True, dimension=3)
        self.block4 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[3], ch[4], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU())
        self.block4_cls = ME.MinkowskiConvolution(ch[4], 1, kernel_size=1, bias=True, dimension=3)
        self.block5 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[4], ch[5], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU())
        self.block5_cls = ME.MinkowskiConvolution(ch[5], 1, kernel_size=1, bias=True, dimension=3)
        self.block6 = nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(ch[5], ch[6], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU())
        self.block6_cls = ME.MinkowskiConvolution(ch[6], 1, kernel_size=1, bias=True, dimension=3)
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])
        kernel_map = cm.kernel_map(out.coordinate_map_key, strided_target_key, kernel_size=kernel_size, region_type=1)
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z_glob, target_key):
        out_cls, targets = [], []
        z = ME.SparseTensor(features=z_glob.F, coordinates=z_glob.C, tensor_stride=self.resolution, coordinate_manager=z_glob.coordinate_manager)
        out1 = self.block1(z)
        out1_cls = self.block1_cls(out1)
        target = self.get_target(out1, target_key)
        targets.append(target)
        out_cls.append(out1_cls)
        keep1 = (out1_cls.F > 0).squeeze()
        if self.training:
            keep1 += target
        out1 = self.pruning(out1, keep1)
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).squeeze()
        if self.training:
            keep2 += target
        out2 = self.pruning(out2, keep2)
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).squeeze()
        if self.training:
            keep3 += target
        out3 = self.pruning(out3, keep3)
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).squeeze()
        if self.training:
            keep4 += target
        out4 = self.pruning(out4, keep4)
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        target = self.get_target(out5, target_key)
        targets.append(target)
        out_cls.append(out5_cls)
        keep5 = (out5_cls.F > 0).squeeze()
        if self.training:
            keep5 += target
        out5 = self.pruning(out5, keep5)
        out6 = self.block6(out5)
        out6_cls = self.block6_cls(out6)
        target = self.get_target(out6, target_key)
        targets.append(target)
        out_cls.append(out6_cls)
        keep6 = (out6_cls.F > 0).squeeze()
        if keep6.sum() > 0:
            out6 = self.pruning(out6, keep6)
        return out_cls, targets, out6


class VAE(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, sinput, gt_target):
        means, log_vars = self.encoder(sinput)
        zs = means
        if self.training:
            zs = zs + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        out_cls, targets, sout = self.decoder(zs, gt_target)
        return out_cls, targets, sout, means, log_vars, zs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PointNet,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     True),
]

class Test_NVIDIA_MinkowskiEngine(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

