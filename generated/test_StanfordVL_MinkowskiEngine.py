import sys
_module = sys.modules[__name__]
del sys
Common = _module
MinkowskiBroadcast = _module
MinkowskiChannelwiseConvolution = _module
MinkowskiConvolution = _module
MinkowskiCoords = _module
MinkowskiFunctional = _module
MinkowskiNetwork = _module
MinkowskiNonlinearity = _module
MinkowskiNormalization = _module
MinkowskiOps = _module
MinkowskiPooling = _module
MinkowskiPruning = _module
MinkowskiUnion = _module
SparseTensor = _module
MinkowskiEngine = _module
modules = _module
resnet_block = _module
senet_block = _module
utils = _module
collation = _module
coords = _module
gradcheck = _module
init = _module
quantization = _module
conf = _module
examples = _module
common = _module
completion = _module
convolution = _module
example = _module
indoor = _module
minkunet = _module
modelnet40 = _module
multigpu = _module
pointnet = _module
reconstruction = _module
resnet = _module
sparse_tensor_basic = _module
training = _module
unet = _module
vae = _module
setup = _module
tests = _module
broadcast = _module
chwise_conv = _module
common = _module
conv = _module
conv_on_coords = _module
coords = _module
dense = _module
kernel_map = _module
norm = _module
pool = _module
pruning = _module
quantization = _module
sparse_tensor = _module
strided_conv = _module
union = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


from collections import Sequence


import numpy as np


from enum import Enum


from itertools import product


from typing import Union


import torch


from torch.nn import Module


from torch.autograd import Function


from torch.nn import Parameter


from typing import List


import torch.nn.functional as F


from abc import ABC


from abc import abstractmethod


import torch.nn as nn


from torch.nn.modules import Module


import warnings


import copy


import logging


import collections.abc


import torch.testing


from typing import Callable


from typing import Optional


from torch.autograd.gradcheck import _as_tuple


from torch.autograd.gradcheck import _differentiable_outputs


from torch.autograd.gradcheck import get_analytical_jacobian


from torch.autograd.gradcheck import get_numerical_jacobian


from torch.autograd.gradcheck import iter_tensors


import time


from time import time


import torch.utils.data


import torch.optim as optim


from torch.optim import SGD


from torch.utils.data.sampler import Sampler


from torchvision.transforms import Compose as VisionCompose


from scipy.linalg import expm


from scipy.linalg import norm


import torch.nn.parallel as parallel


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import re


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import BuildExtension


COORDS_KEY_DIFFERENT_ERROR = 'SparseTensors must have the same coords_key.'


COORDS_MAN_DIFFERENT_ERROR = 'SparseTensors must share the same coordinate manager for this operation. Please refer to the SparseTensor creation API (https://stanfordvl.github.io/MinkowskiEngine/sparse_tensor.html) to share the coordinate manager, or set the sparse tensor operation mode with `set_sparse_tensor_operation_mode` to share it by default.'


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


class CoordsKey:

    def __init__(self, D):
        self.D = D
        self.CPPCoordsKey = MEB.CoordsKey()
        self.CPPCoordsKey.setDimension(D)

    def isKeySet(self):
        return self.CPPCoordsKey.isKeySet()

    def setKey(self, key):
        self.CPPCoordsKey.setKey(key)

    def getKey(self):
        return self.CPPCoordsKey.getKey()

    def setTensorStride(self, tensor_stride):
        tensor_stride = convert_to_int_list(tensor_stride, self.D)
        self.CPPCoordsKey.setTensorStride(tensor_stride)

    def getTensorStride(self):
        return self.CPPCoordsKey.getTensorStride()

    def __repr__(self):
        return str(self.CPPCoordsKey)

    def __eq__(self, other):
        assert isinstance(other, CoordsKey)
        return self.getKey() == other.getKey()


def convert_to_int_tensor(arg: Union[int, Sequence, np.ndarray, torch.IntTensor], dimension: int):
    if isinstance(arg, torch.IntTensor):
        assert arg.numel() == dimension
        return arg
    if isinstance(arg, (Sequence, np.ndarray)):
        tmp = torch.IntTensor([i for i in arg])
        assert tmp.numel() == dimension
    elif np.isscalar(arg):
        tmp = torch.IntTensor([int(arg) for i in range(dimension)])
    else:
        raise ValueError('Input must be a scalar or a sequence')
    return tmp


class RegionType(Enum):
    """
    Define the kernel region type
    """
    HYPERCUBE = 0, 'HYPERCUBE'
    HYPERCROSS = 1, 'HYPERCROSS'
    CUSTOM = 2, 'CUSTOM'
    HYBRID = 3, 'HYBRID'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


def prep_args(tensor_stride: Union[int, Sequence, np.ndarray, torch.IntTensor], stride: Union[int, Sequence, np.ndarray, torch.IntTensor], kernel_size: Union[int, Sequence, np.ndarray, torch.IntTensor], dilation: Union[int, Sequence, np.ndarray, torch.IntTensor], region_type: Union[int, RegionType], D=-1):
    assert torch.prod(kernel_size > 0), f'kernel_size must be a positive integer, provided {kernel_size}'
    assert D > 0, f'dimension must be a positive integer, {D}'
    tensor_stride = convert_to_int_tensor(tensor_stride, D)
    stride = convert_to_int_tensor(stride, D)
    kernel_size = convert_to_int_tensor(kernel_size, D)
    dilation = convert_to_int_tensor(dilation, D)
    region_type = int(region_type)
    return tensor_stride, stride, kernel_size, dilation, region_type


class SparseTensorOperationMode(Enum):
    """
    `SEPARATE_COORDS_MANAGER`: always create a new coordinate manager.
    `SHARE_COORDS_MANAGER`: always use the globally defined coordinate manager. Must clear the coordinate manager manually by :attr:`MinkowskiEngine.SparseTensor.clear_global_coords_man`
    """
    SEPARATE_COORDS_MANAGER = 0
    SHARE_COORDS_MANAGER = 1


class SparseTensorQuantizationMode(Enum):
    """
    `RANDOM_SUBSAMPLE`: Subsample one coordinate per each quantization block randomly.
    `UNWEIGHTED_AVERAGE`: average all features within a quantization block equally.
    """
    RANDOM_SUBSAMPLE = 0
    UNWEIGHTED_AVERAGE = 1


class MinkowskiModuleBase(Module):
    MODULE = None

    def __init__(self, *args, **kwargs):
        super(MinkowskiModuleBase, self).__init__()
        self.module = self.MODULE(*args, **kwargs)

    def forward(self, input):
        output = self.module(input.F)
        return SparseTensor(output, coords_key=input.coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class OperationType(Enum):
    ADDITION = 0
    MULTIPLICATION = 1


def get_postfix(tensor: torch.Tensor):
    postfix = 'GPU' if tensor.is_cuda else 'CPU'
    if isinstance(tensor, torch.DoubleTensor) or isinstance(tensor, torch.DoubleTensor):
        postfix += 'd'
    else:
        postfix += 'f'
    return postfix


def get_minkowski_function(name, variable):
    fn_name = name + get_postfix(variable)
    if hasattr(MEB, fn_name):
        return getattr(MEB, fn_name)
    elif variable.is_cuda:
        raise ValueError(f'Function {fn_name} not available. Please compile MinkowskiEngine where `torch.cuda.is_available()` is `True`.')
    else:
        raise ValueError(f'Function {fn_name} not available.')


op_to_int = {i: i.value for i in OperationType}


def operation_type_to_int(op):
    assert isinstance(op, OperationType)
    return op_to_int[op]


class MinkowskiBroadcastFunction(Function):

    @staticmethod
    def forward(ctx, input_features, input_features_global, operation_type, in_coords_key, glob_coords_key, coords_manager):
        assert input_features.shape[1] == input_features_global.shape[1]
        assert input_features.type() == input_features_global.type()
        assert isinstance(operation_type, OperationType)
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()
        if not input_features_global.is_contiguous():
            input_features_global = input_features_global.contiguous()
        ctx.op = operation_type_to_int(operation_type)
        ctx.in_feat = input_features
        ctx.in_feat_glob = input_features_global
        ctx.in_coords_key = in_coords_key
        ctx.glob_coords_key = glob_coords_key
        ctx.coords_manager = coords_manager
        fw_fn = get_minkowski_function('BroadcastForward', input_features)
        out_feat = fw_fn(ctx.in_feat, ctx.in_feat_glob, ctx.op, ctx.in_coords_key.CPPCoordsKey, ctx.glob_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()
        grad_in_feat = grad_out_feat.new()
        grad_in_feat_glob = grad_out_feat.new()
        bw_fn = get_minkowski_function('BroadcastBackward', grad_out_feat)
        bw_fn(ctx.in_feat, grad_in_feat, ctx.in_feat_glob, grad_in_feat_glob, grad_out_feat, ctx.op, ctx.in_coords_key.CPPCoordsKey, ctx.glob_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager)
        return grad_in_feat, grad_in_feat_glob, None, None, None, None


class AbstractMinkowskiBroadcast(Module):

    def __init__(self, operation_type):
        super(AbstractMinkowskiBroadcast, self).__init__()
        assert isinstance(operation_type, OperationType)
        self.operation_type = operation_type
        self.broadcast = MinkowskiBroadcastFunction()

    def forward(self, input, input_glob):
        assert isinstance(input, SparseTensor)
        output = self.broadcast.apply(input.F, input_glob.F, self.operation_type, input.coords_key, input_glob.coords_key, input.coords_man)
        return SparseTensor(output, coords_key=input.coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__


class MinkowskiBroadcastAddition(AbstractMinkowskiBroadcast):
    """Broadcast the reduced features to all input coordinates.

    .. math::

        \\mathbf{y}_\\mathbf{u} = \\mathbf{x}_{1, \\mathbf{u}} + \\mathbf{x}_2
        \\; \\text{for} \\; \\mathbf{u} \\in \\mathcal{C}^\\text{in}


    For all input :math:`\\mathbf{x}_\\mathbf{u}`, add :math:`\\mathbf{x}_2`. The
    output coordinates will be the same as the input coordinates
    :math:`\\mathcal{C}^\\text{in} = \\mathcal{C}^\\text{out}`.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def __init__(self):
        AbstractMinkowskiBroadcast.__init__(self, OperationType.ADDITION)


class MinkowskiBroadcastMultiplication(AbstractMinkowskiBroadcast):
    """Broadcast reduced features to all input coordinates.

    .. math::

        \\mathbf{y}_\\mathbf{u} = \\mathbf{x}_{1, \\mathbf{u}} \\times \\mathbf{x}_2
        \\; \\text{for} \\; \\mathbf{u} \\in \\mathcal{C}^\\text{in}


    For all input :math:`\\mathbf{x}_\\mathbf{u}`, multiply :math:`\\mathbf{x}_2`
    element-wise. The output coordinates will be the same as the input
    coordinates :math:`\\mathcal{C}^\\text{in} = \\mathcal{C}^\\text{out}`.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def __init__(self):
        AbstractMinkowskiBroadcast.__init__(self, OperationType.MULTIPLICATION)


class MinkowskiBroadcast(Module):
    """Broadcast reduced features to all input coordinates.

    .. math::

        \\mathbf{y}_\\mathbf{u} = \\mathbf{x}_2 \\; \\text{for} \\; \\mathbf{u} \\in
        \\mathcal{C}^\\text{in}


    For all input :math:`\\mathbf{x}_\\mathbf{u}`, copy value :math:`\\mathbf{x}_2`
    element-wise. The output coordinates will be the same as the input
    coordinates :math:`\\mathcal{C}^\\text{in} = \\mathcal{C}^\\text{out}`. The
    first input :math:`\\mathbf{x}_1` is only used for defining the output
    coordinates.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, input, input_glob):
        assert isinstance(input, SparseTensor)
        assert isinstance(input_glob, SparseTensor)
        broadcast_feat = input.F.new(len(input), input_glob.size()[1])
        row_inds = input.coords_man.get_row_indices_per_batch(input.coords_key)
        for b, row_ind in enumerate(row_inds):
            broadcast_feat[row_ind] = input_glob.F[b]
        return SparseTensor(broadcast_feat, coords_key=input.coords_key, coords_manager=input.coords_man)


class MinkowskiBroadcastConcatenation(MinkowskiBroadcast):
    """Broadcast reduced features to all input coordinates and concatenate to the input.

    .. math::

        \\mathbf{y}_\\mathbf{u} = [\\mathbf{x}_{1,\\mathbf{u}}, \\mathbf{x}_2] \\;
        \\text{for} \\; \\mathbf{u} \\in \\mathcal{C}^\\text{in}


    For all input :math:`\\mathbf{x}_\\mathbf{u}`, concatenate vector
    :math:`\\mathbf{x}_2`. :math:`[\\cdot, \\cdot]` is a concatenation operator.
    The output coordinates will be the same as the input coordinates
    :math:`\\mathcal{C}^\\text{in} = \\mathcal{C}^\\text{out}`.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def forward(self, input, input_glob):
        assert isinstance(input, SparseTensor)
        assert isinstance(input_glob, SparseTensor)
        broadcast_feat = input.F.new(len(input), input_glob.size()[1])
        row_inds = input.coords_man.get_row_indices_per_batch(input.coords_key)
        for b, row_ind in enumerate(row_inds):
            broadcast_feat[row_ind] = input_glob.F[b]
        broadcast_cat = torch.cat((input.F, broadcast_feat), dim=1)
        return SparseTensor(broadcast_cat, coords_key=input.coords_key, coords_manager=input.coords_man)


def convert_region_type(region_type: RegionType, tensor_stride: Union[Sequence, np.ndarray, torch.IntTensor], kernel_size: Union[Sequence, np.ndarray, torch.IntTensor], up_stride: Union[Sequence, np.ndarray, torch.IntTensor], dilation: Union[Sequence, np.ndarray, torch.IntTensor], region_offset: Union[Sequence, np.ndarray, torch.IntTensor], axis_types: Union[Sequence, np.ndarray, torch.IntTensor], dimension: int, center: bool=True):
    """
    when center is True, the custom region_offset will be centered at the
    origin. Currently, for HYPERCUBE, HYPERCROSS with odd kernel sizes cannot
    use center=False.

    up_stride: stride for conv_transpose, otherwise set it as 1
    """
    if region_type == RegionType.HYPERCUBE:
        assert region_offset is None, 'Region offset must be None when region_type is given'
        assert axis_types is None, 'Axis types must be None when region_type is given'
        assert torch.prod(kernel_size > 0) == 1
        kernel_volume = int(torch.prod(kernel_size))
    elif region_type == RegionType.HYPERCROSS:
        assert torch.prod(kernel_size > 0) == 1, 'kernel_size must be positive'
        assert (kernel_size % 2).prod() == 1, 'kernel_size must be odd for region_type HYPERCROSS'
        kernel_volume = int(torch.sum(kernel_size - 1) + 1)
    elif region_type == RegionType.HYBRID:
        assert region_offset is None, 'region_offset must be None when region_type is HYBRID'
        region_offset = [[0] * dimension]
        kernel_size_list = kernel_size.tolist()
        for axis_type, curr_kernel_size, d in zip(axis_types, kernel_size_list, range(dimension)):
            new_offset = []
            if axis_type == RegionType.HYPERCUBE:
                for offset in region_offset:
                    for curr_offset in range(curr_kernel_size):
                        off_center = int(math.floor((curr_kernel_size - 1) / 2)) if center else 0
                        offset = offset.copy()
                        if curr_offset == off_center:
                            continue
                        offset[d] = (curr_offset - off_center) * dilation[d] * (tensor_stride[d] / up_stride[d])
                        new_offset.append(offset)
            region_offset.extend(new_offset)
        for axis_type, curr_kernel_size, d in zip(axis_types, kernel_size_list, range(dimension)):
            new_offset = []
            if axis_type == RegionType.HYPERCROSS:
                for curr_offset in range(curr_kernel_size):
                    off_center = int(math.floor((curr_kernel_size - 1) / 2)) if center else 0
                    offset = [0] * dimension
                    if curr_offset == off_center:
                        continue
                    offset[d] = (curr_offset - off_center) * dilation[d] * (tensor_stride[d] / up_stride[d])
                    new_offset.append(offset)
            region_offset.extend(new_offset)
        region_type = RegionType.CUSTOM
        region_offset = torch.IntTensor(region_offset)
        kernel_volume = int(region_offset.size(0))
    elif region_type == RegionType.CUSTOM:
        assert region_offset.numel() > 0, 'region_offset must be non empty when region_type is CUSTOM'
        assert region_offset.size(1) == dimension, 'region_offset must have the same dimension as the network'
        kernel_volume = int(region_offset.size(0))
        assert isinstance(region_offset.dtype, torch.IntTensor), 'region_offset must be a torch.IntTensor.'
    else:
        raise NotImplementedError()
    if region_offset is None:
        region_offset = torch.IntTensor()
    return region_type, region_offset, kernel_volume


def get_kernel_volume(region_type, kernel_size, region_offset, axis_types, dimension):
    """
    when center is True, the custom region_offset will be centered at the
    origin. Currently, for HYPERCUBE, HYPERCROSS with odd kernel sizes cannot
    use center=False.
    """
    if region_type == RegionType.HYPERCUBE:
        assert region_offset is None, 'Region offset must be None when region_type is given'
        assert axis_types is None, 'Axis types must be None when region_type is given'
        assert torch.prod(kernel_size > 0) == 1
        kernel_volume = int(torch.prod(kernel_size))
    elif region_type == RegionType.HYPERCROSS:
        assert torch.prod(kernel_size > 0) == 1, 'kernel_size must be positive'
        assert (kernel_size % 2).prod() == 1, 'kernel_size must be odd for region_type HYPERCROSS'
        kernel_volume = int(torch.sum(kernel_size - 1) + 1)
    elif region_type == RegionType.HYBRID:
        assert region_offset is None, 'region_offset must be None when region_type is HYBRID'
        kernel_size_list = kernel_size.tolist()
        kernel_volume = 1
        for axis_type, curr_kernel_size, d in zip(axis_types, kernel_size_list, range(dimension)):
            if axis_type == RegionType.HYPERCUBE:
                kernel_volume *= curr_kernel_size
        for axis_type, curr_kernel_size, d in zip(axis_types, kernel_size_list, range(dimension)):
            if axis_type == RegionType.HYPERCROSS:
                kernel_volume += curr_kernel_size - 1
    elif region_type == RegionType.CUSTOM:
        assert region_offset.numel() > 0, 'region_offset must be non empty when region_type is CUSTOM'
        assert region_offset.size(1) == dimension, 'region_offset must have the same dimension as the network'
        kernel_volume = int(region_offset.size(0))
    else:
        raise NotImplementedError()
    return kernel_volume


class KernelGenerator:

    def __init__(self, kernel_size=-1, stride=1, dilation=1, is_transpose=False, region_type=RegionType.HYPERCUBE, region_offsets=None, axis_types=None, dimension=-1):
        """
            :attr:`region_type` (RegionType, optional): defines the kernel
            shape. Please refer to MinkowskiEngine.Comon for details.

            :attr:`region_offset` (torch.IntTensor, optional): when the
            :attr:`region_type` is :attr:`RegionType.CUSTOM`, the convolution
            kernel uses the provided `region_offset` to define offsets. It
            should be a matrix of size :math:`N \\times D` where :math:`N` is
            the number of offsets and :math:`D` is the dimension of the
            space.

            :attr:`axis_types` (list of RegionType, optional): If given, it
            uses different methods to create a kernel for each axis. e.g., when
            it is `[RegionType.HYPERCUBE, RegionType.HYPERCUBE,
            RegionType.HYPERCROSS]`, the kernel would be rectangular for the
            first two dimensions and cross shaped for the thrid dimension.
        """
        assert dimension > 0
        assert isinstance(region_type, RegionType)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)
        self.cache = {}
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offsets = region_offsets
        self.axis_types = axis_types
        self.dimension = dimension
        self.kernel_volume = get_kernel_volume(region_type, kernel_size, region_offsets, axis_types, dimension)

    def get_kernel(self, tensor_stride, is_transpose):
        assert len(tensor_stride) == self.dimension
        if tuple(tensor_stride) not in self.cache:
            up_stride = self.stride if is_transpose else torch.Tensor([1] * self.dimension)
            self.cache[tuple(tensor_stride)] = convert_region_type(self.region_type, tensor_stride, self.kernel_size, up_stride, self.dilation, self.region_offsets, self.axis_types, self.dimension)
        return self.cache[tuple(tensor_stride)]


class MinkowskiConvolutionFunction(Function):

    @staticmethod
    def forward(ctx, input_features, kernel, tensor_stride=1, stride=1, kernel_size=-1, dilation=1, region_type=0, region_offset=None, in_coords_key=None, out_coords_key=None, coords_manager=None):
        """
        region_type=0 HyperCube
        """
        assert input_features.shape[1] == kernel.shape[1], 'The input shape ' + str(list(input_features.shape)) + ' does not match the kernel shape ' + str(list(kernel.shape))
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        assert input_features.type() == kernel.type(), f'Type mismatch input: {input_features.type()} != kernel: {kernel.type()}'
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()
        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key.D)
        if region_offset is None:
            region_offset = torch.IntTensor()
        ctx.in_feat = input_features
        ctx.kernel = kernel
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key, out_coords_key, coords_manager)
        D = in_coords_key.D
        out_feat = input_features.new()
        fw_fn = get_minkowski_function('ConvolutionForward', input_features)
        fw_fn(ctx.in_feat, out_feat, kernel, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), region_type, region_offset, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = get_minkowski_function('ConvolutionBackward', grad_out_feat)
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel, grad_kernel, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), ctx.region_type, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, grad_kernel, None, None, None, None, None, None, None, None, None


class MinkowskiConvolutionTransposeFunction(Function):

    @staticmethod
    def forward(ctx, input_features, kernel, tensor_stride=1, stride=1, kernel_size=-1, dilation=1, region_type=0, region_offset=None, generate_new_coords=False, in_coords_key=None, out_coords_key=None, coords_manager=None):
        """
        region_type=0 HyperCube
        """
        assert input_features.shape[1] == kernel.shape[1], 'The input shape ' + str(list(input_features.shape)) + ' does not match the kernel shape ' + str(list(kernel.shape))
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        assert input_features.type() == kernel.type(), f'Type mismatch input: {input_features.type()} != kernel: {kernel.type()}'
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()
        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key.D)
        if region_offset is None:
            region_offset = torch.IntTensor()
        ctx.in_feat = input_features
        ctx.kernel = kernel
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key, out_coords_key, coords_manager)
        D = in_coords_key.D
        out_feat = input_features.new()
        fw_fn = get_minkowski_function('ConvolutionTransposeForward', input_features)
        fw_fn(ctx.in_feat, out_feat, kernel, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), region_type, region_offset, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager, generate_new_coords)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = get_minkowski_function('ConvolutionTransposeBackward', grad_out_feat)
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel, grad_kernel, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), ctx.region_type, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, grad_kernel, None, None, None, None, None, None, None, None, None, None


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

    def get_index_map(self, coords, tensor_stride):
        """
        Get the current coords (with duplicates) index map.

        If `tensor_stride > 1`, use

        .. code-block:: python

           coords = torch.cat(((coords[:, :D] / tensor_stride) * tensor_stride, coords[:, D:]), dim=1)

        """
        assert isinstance(coords, torch.IntTensor), 'Coord must be IntTensor'
        index_map = torch.IntTensor()
        tensor_stride = convert_to_int_tensor(tensor_stride, self.D)
        success = MEB.get_index_map(coords.contiguous(), index_map, tensor_stride, self.D, self.net_metadata.ffi)
        if success < 0:
            raise ValueError('get_index_map failed')
        return index_map

    def permute_label(self, label, max_label, tensor_stride):
        if tensor_stride == 1 or np.prod(tensor_stride) == 1:
            return label
        tensor_stride = convert_to_int_tensor(tensor_stride, self.D)
        permutation = self.get_permutation(tensor_stride, 1)
        nrows = self.get_nrows(tensor_stride)
        label = label.contiguous().numpy()
        permutation = permutation.numpy()
        counter = np.zeros((nrows, max_label), dtype='int32')
        np.add.at(counter, (permutation, label), 1)
        return torch.from_numpy(np.argmax(counter, 1))

    def permute_feature(self, feat, tensor_stride, dtype=np.float32):
        tensor_stride = convert_to_int_tensor(tensor_stride, self.D)
        permutation = self.get_permutation(tensor_stride, 1)
        nrows = self.get_nrows(tensor_stride)
        feat_np = feat.contiguous().numpy()
        warped_feat = np.zeros((nrows, feat.size(1)), dtype=dtype)
        counter = np.zeros((nrows, 1), dtype='int32')
        for j in range(feat.size(1)):
            np.add.at(warped_feat, (permutation, j), feat_np[:, (j)])
        np.add.at(counter, permutation, 1)
        warped_feat = warped_feat / counter
        return torch.from_numpy(warped_feat)


class MinkowskiReLU(MinkowskiModuleBase):
    MODULE = torch.nn.ReLU


class MinkowskiPReLU(MinkowskiModuleBase):
    MODULE = torch.nn.PReLU


class MinkowskiELU(MinkowskiModuleBase):
    MODULE = torch.nn.ELU


class MinkowskiSELU(MinkowskiModuleBase):
    MODULE = torch.nn.SELU


class MinkowskiCELU(MinkowskiModuleBase):
    MODULE = torch.nn.CELU


class MinkowskiDropout(MinkowskiModuleBase):
    MODULE = torch.nn.Dropout


class MinkowskiThreshold(MinkowskiModuleBase):
    MODULE = torch.nn.Threshold


class MinkowskiSigmoid(MinkowskiModuleBase):
    MODULE = torch.nn.Sigmoid


class MinkowskiTanh(MinkowskiModuleBase):
    MODULE = torch.nn.Tanh


class MinkowskiSoftmax(MinkowskiModuleBase):
    MODULE = torch.nn.Softmax


class MinkowskiBatchNorm(Module):
    """A batch normalization layer for a sparse tensor.

    See the pytorch :attr:`torch.nn.BatchNorm1d` for more details.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MinkowskiBatchNorm, self).__init__()
        self.bn = torch.nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, input):
        output = self.bn(input.F)
        return SparseTensor(output, coords_key=input.coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        s = '({}, eps={}, momentum={}, affine={}, track_running_stats={})'.format(self.bn.num_features, self.bn.eps, self.bn.momentum, self.bn.affine, self.bn.track_running_stats)
        return self.__class__.__name__ + s


class MinkowskiSyncBatchNorm(MinkowskiBatchNorm):
    """A batch normalization layer with multi GPU synchronization.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None):
        Module.__init__(self)
        self.bn = torch.nn.SyncBatchNorm(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, process_group=process_group)

    def forward(self, input):
        output = self.bn(input.F)
        return SparseTensor(output, coords_key=input.coords_key, coords_manager=input.coords_man)

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

            >>> # Network with nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100)
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
                module_output.bn.weight.data = module.bn.weight.data.clone().detach()
                module_output.bn.bias.data = module.bn.bias.data.clone().detach()
                module_output.bn.weight.requires_grad = module.bn.weight.requires_grad
                module_output.bn.bias.requires_grad = module.bn.bias.requires_grad
            module_output.bn.running_mean = module.bn.running_mean
            module_output.bn.running_var = module.bn.running_var
            module_output.bn.num_batches_tracked = module.bn.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output


class GlobalPoolingMode(Enum):
    """
    Define the global pooling mode
    """
    AUTO = 0, 'AUTO'
    INDEX_SELECT = 1, 'INDEX_SELECT'
    SPARSE = 2, 'SPARSE'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class MinkowskiGlobalPoolingFunction(Function):

    @staticmethod
    def forward(ctx, input_features, average=True, mode=GlobalPoolingMode.AUTO, in_coords_key=None, out_coords_key=None, coords_manager=None):
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert isinstance(mode, GlobalPoolingMode), f'Mode must be an instance of GlobalPoolingMode, {mode}'
        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key
        ctx.in_feat = input_features
        ctx.average = average
        ctx.coords_manager = coords_manager
        ctx.mode = mode.value
        fw_fn = get_minkowski_function('GlobalPoolingForward', input_features)
        out_feat, num_nonzero = fw_fn(ctx.in_feat, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager, ctx.average, ctx.mode)
        ctx.num_nonzero = num_nonzero
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        bw_fn = get_minkowski_function('GlobalPoolingBackward', grad_out_feat)
        grad_in_feat = bw_fn(ctx.in_feat, grad_out_feat, ctx.num_nonzero, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager, ctx.average)
        return grad_in_feat, None, None, None, None, None


class MinkowskiGlobalPooling(MinkowskiModuleBase):
    """Pool all input features to one output.

    .. math::

        \\mathbf{y} = \\frac{1}{|\\mathcal{C}^\\text{in}|} \\sum_{\\mathbf{i} \\in
        \\mathcal{C}^\\text{in}} \\mathbf{x}_{\\mathbf{i}}

    """

    def __init__(self, average=True, mode=GlobalPoolingMode.AUTO):
        """Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        Args:
            :attr:`average` (bool): when True, return the averaged output. If
            not, return the sum of all input features.

        """
        super(MinkowskiGlobalPooling, self).__init__()
        assert isinstance(mode, GlobalPoolingMode), f'Mode must be an instance of GlobalPoolingMode. mode={mode}'
        self.mode = mode
        self.average = average
        self.pooling = MinkowskiGlobalPoolingFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        out_coords_key = CoordsKey(input.coords_key.D)
        output = self.pooling.apply(input.F, self.average, self.mode, input.coords_key, out_coords_key, input.coords_man)
        return SparseTensor(output, coords_key=out_coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '(average=' + str(self.average) + ')'


class MinkowskiStableInstanceNorm(Module):

    def __init__(self, num_features):
        Module.__init__(self)
        self.num_features = num_features
        self.eps = 1e-06
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_in = MinkowskiGlobalPooling()
        self.glob_sum = MinkowskiBroadcastAddition()
        self.glob_sum2 = MinkowskiBroadcastAddition()
        self.glob_mean = MinkowskiGlobalPooling()
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
        temp = SparseTensor(centered_in.F ** 2, coords_key=centered_in.coords_key, coords_manager=centered_in.coords_man)
        var_in = self.glob_mean(temp)
        instd_in = SparseTensor(1 / (var_in.F + self.eps).sqrt(), coords_key=var_in.coords_key, coords_manager=var_in.coords_man)
        x = self.glob_times(self.glob_sum2(x, neg_mean_in), instd_in)
        return SparseTensor(x.F * self.weight + self.bias, coords_key=x.coords_key, coords_manager=x.coords_man)


class MinkowskiInstanceNormFunction(Function):

    @staticmethod
    def forward(ctx, in_feat, mode=GlobalPoolingMode.AUTO, in_coords_key=None, glob_coords_key=None, coords_manager=None):
        assert isinstance(mode, GlobalPoolingMode), f'Mode must be an instance of GlobalPoolingMode, {mode}'
        if glob_coords_key is None:
            glob_coords_key = CoordsKey(in_coords_key.D)
        gpool_forward = get_minkowski_function('GlobalPoolingForward', in_feat)
        broadcast_forward = get_minkowski_function('BroadcastForward', in_feat)
        add = operation_type_to_int(OperationType.ADDITION)
        multiply = operation_type_to_int(OperationType.MULTIPLICATION)
        mean = in_feat.new()
        num_nonzero = in_feat.new()
        cpp_in_coords_key = in_coords_key.CPPCoordsKey
        cpp_glob_coords_key = glob_coords_key.CPPCoordsKey
        cpp_coords_manager = coords_manager.CPPCoordsManager
        mean, num_nonzero = gpool_forward(in_feat, cpp_in_coords_key, cpp_glob_coords_key, cpp_coords_manager, True, mode.value)
        centered_feat = broadcast_forward(in_feat, -mean, add, cpp_in_coords_key, cpp_glob_coords_key, cpp_coords_manager)
        variance, num_nonzero = gpool_forward(centered_feat ** 2, cpp_in_coords_key, cpp_glob_coords_key, cpp_coords_manager, True, mode.value)
        inv_std = 1 / (variance + 1e-08).sqrt()
        norm_feat = broadcast_forward(centered_feat, inv_std, multiply, cpp_in_coords_key, cpp_glob_coords_key, cpp_coords_manager)
        ctx.mode = mode
        ctx.in_coords_key, ctx.glob_coords_key = in_coords_key, glob_coords_key
        ctx.coords_manager = coords_manager
        ctx.save_for_backward(inv_std, norm_feat)
        return norm_feat

    @staticmethod
    def backward(ctx, out_grad):
        in_coords_key, glob_coords_key = ctx.in_coords_key, ctx.glob_coords_key
        coords_manager = ctx.coords_manager
        inv_std, norm_feat = ctx.saved_tensors
        gpool_forward = get_minkowski_function('GlobalPoolingForward', out_grad)
        broadcast_forward = get_minkowski_function('BroadcastForward', out_grad)
        add = operation_type_to_int(OperationType.ADDITION)
        multiply = operation_type_to_int(OperationType.MULTIPLICATION)
        cpp_in_coords_key = in_coords_key.CPPCoordsKey
        cpp_glob_coords_key = glob_coords_key.CPPCoordsKey
        cpp_coords_manager = coords_manager.CPPCoordsManager
        mean_dout, num_nonzero = gpool_forward(out_grad, cpp_in_coords_key, cpp_glob_coords_key, cpp_coords_manager, True, ctx.mode.value)
        mean_dout_feat, num_nonzero = gpool_forward(out_grad * norm_feat, cpp_in_coords_key, cpp_glob_coords_key, cpp_coords_manager, True, ctx.mode.value)
        feat_mean_dout_feat = broadcast_forward(norm_feat, mean_dout_feat, multiply, cpp_in_coords_key, cpp_glob_coords_key, cpp_coords_manager)
        unnorm_din = broadcast_forward(out_grad - feat_mean_dout_feat, -mean_dout, add, cpp_in_coords_key, cpp_glob_coords_key, cpp_coords_manager)
        norm_din = broadcast_forward(unnorm_din, inv_std, multiply, cpp_in_coords_key, cpp_glob_coords_key, cpp_coords_manager)
        return norm_din, None, None, None, None


class MinkowskiInstanceNorm(Module):
    """A instance normalization layer for a sparse tensor.

    """

    def __init__(self, num_features, mode=GlobalPoolingMode.AUTO):
        """
        Args:

            num_features (int): the dimension of the input feautres.

            mode (GlobalPoolingModel, optional): The internal global pooling computation mode.
        """
        Module.__init__(self)
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.reset_parameters()
        self.mode = mode
        self.inst_norm = MinkowskiInstanceNormFunction()

    def __repr__(self):
        s = f'(nchannels={self.num_features})'
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        output = self.inst_norm.apply(input.F, self.mode, input.coords_key, None, input.coords_man)
        output = output * self.weight + self.bias
        return SparseTensor(output, coords_key=input.coords_key, coords_manager=input.coords_man)


class MinkowskiLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(MinkowskiLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        output = self.linear(input.F)
        return SparseTensor(output, coords_key=input.coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        s = '(in_features={}, out_features={}, bias={})'.format(self.linear.in_features, self.linear.out_features, self.linear.bias is not None)
        return self.__class__.__name__ + s


class MinkowskiAvgPoolingFunction(Function):
    """
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, out_feat, ctx.num_nonzero, which are initialized inside
    the ffi function.
    """

    @staticmethod
    def forward(ctx, input_features, tensor_stride=1, stride=1, kernel_size=-1, dilation=1, region_type=0, region_offset=None, average=True, in_coords_key=None, out_coords_key=None, coords_manager=None):
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()
        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key.D)
        if region_offset is None:
            region_offset = torch.IntTensor()
        ctx.in_feat = input_features
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key, out_coords_key, coords_manager)
        ctx.use_avg = average
        D = in_coords_key.D
        out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()
        fw_fn = get_minkowski_function('AvgPoolingForward', input_features)
        fw_fn(ctx.in_feat, out_feat, ctx.num_nonzero, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), region_type, region_offset, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager, ctx.use_avg)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()
        grad_in_feat = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = get_minkowski_function('AvgPoolingBackward', grad_out_feat)
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), ctx.region_type, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager, ctx.use_avg)
        return grad_in_feat, None, None, None, None, None, None, None, None, None, None


class MinkowskiMaxPoolingFunction(Function):

    @staticmethod
    def forward(ctx, input_features, tensor_stride=1, stride=1, kernel_size=-1, dilation=1, region_type=0, region_offset=None, in_coords_key=None, out_coords_key=None, coords_manager=None):
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()
        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key.D)
        if region_offset is None:
            region_offset = torch.IntTensor()
        ctx.in_feat = input_features
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key, out_coords_key, coords_manager)
        D = in_coords_key.D
        out_feat = input_features.new()
        max_index = input_features.new().int()
        ctx.max_index = max_index
        fw_fn = get_minkowski_function('MaxPoolingForward', input_features)
        fw_fn(input_features, out_feat, max_index, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), region_type, region_offset, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()
        grad_in_feat = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = get_minkowski_function('MaxPoolingBackward', grad_out_feat)
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.max_index, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), ctx.region_type, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None, None, None, None, None


class MinkowskiPoolingTransposeFunction(Function):

    @staticmethod
    def forward(ctx, input_features, tensor_stride=1, stride=1, kernel_size=-1, dilation=1, region_type=-1, region_offset=None, average=False, in_coords_key=None, out_coords_key=None, coords_manager=None):
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key.D)
        if region_offset is None:
            region_offset = torch.IntTensor()
        ctx.in_feat = input_features
        out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key, out_coords_key, coords_manager)
        D = in_coords_key.D
        fw_fn = get_minkowski_function('PoolingTransposeForward', input_features)
        fw_fn(ctx.in_feat, out_feat, ctx.num_nonzero, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), region_type, region_offset, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = get_minkowski_function('PoolingTransposeBackward', grad_out_feat)
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero, convert_to_int_list(ctx.tensor_stride, D), convert_to_int_list(ctx.stride, D), convert_to_int_list(ctx.kernel_size, D), convert_to_int_list(ctx.dilation, D), ctx.region_type, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None, None, None, None, None, None


class MinkowskiGlobalSumPooling(MinkowskiGlobalPooling):

    def __init__(self, mode=GlobalPoolingMode.AUTO):
        """Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        """
        MinkowskiGlobalPooling.__init__(self, False, mode=mode)


class MinkowskiGlobalAvgPooling(MinkowskiGlobalPooling):

    def __init__(self, mode=GlobalPoolingMode.AUTO):
        """Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        """
        MinkowskiGlobalPooling.__init__(self, True, mode=mode)


class MinkowskiGlobalMaxPoolingFunction(Function):

    @staticmethod
    def forward(ctx, input_features, in_coords_key=None, out_coords_key=None, coords_manager=None):
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key
        ctx.in_feat = input_features
        out_feat = input_features.new()
        max_index = input_features.new().int()
        ctx.max_index = max_index
        ctx.coords_manager = coords_manager
        fw_fn = get_minkowski_function('GlobalMaxPoolingForward', input_features)
        fw_fn(ctx.in_feat, out_feat, ctx.max_index, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = get_minkowski_function('GlobalMaxPoolingBackward', grad_out_feat)
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.max_index, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None


class MinkowskiGlobalMaxPooling(MinkowskiModuleBase):
    """Max pool all input features to one output feature at the origin.

    .. math::

        \\mathbf{y} = \\frac{1}{|\\mathcal{C}^\\text{in}|} \\max_{\\mathbf{i} \\in
        \\mathcal{C}^\\text{in}} \\mathbf{x}_{\\mathbf{i}}

    """

    def __init__(self):
        """Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        """
        super(MinkowskiGlobalMaxPooling, self).__init__()
        self.pooling = MinkowskiGlobalMaxPoolingFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        out_coords_key = CoordsKey(input.coords_key.D)
        output = self.pooling.apply(input.F, input.coords_key, out_coords_key, input.coords_man)
        return SparseTensor(output, coords_key=out_coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__


class MinkowskiPruningFunction(Function):

    @staticmethod
    def forward(ctx, in_feat, mask, in_coords_key, out_coords_key, coords_manager):
        assert in_feat.size(0) == mask.size(0)
        assert isinstance(mask, torch.BoolTensor), 'Mask must be a cpu bool tensor.'
        if not in_feat.is_contiguous():
            in_feat = in_feat.contiguous()
        if not mask.is_contiguous():
            mask = mask.contiguous()
        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key
        ctx.coords_manager = coords_manager
        out_feat = in_feat.new()
        fw_fn = get_minkowski_function('PruningForward', in_feat)
        fw_fn(in_feat, out_feat, mask, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()
        grad_in_feat = grad_out_feat.new()
        bw_fn = get_minkowski_function('PruningBackward', grad_out_feat)
        bw_fn(grad_in_feat, grad_out_feat, ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None


class MinkowskiPruning(Module):
    """Remove specified coordinates from a :attr:`MinkowskiEngine.SparseTensor`.

    """

    def __init__(self):
        super(MinkowskiPruning, self).__init__()
        self.pruning = MinkowskiPruningFunction()

    def forward(self, input, mask):
        """
        Args:
            :attr:`input` (:attr:`MinkowskiEnigne.SparseTensor`): a sparse tensor
            to remove coordinates from.

            :attr:`mask` (:attr:`torch.BoolTensor`): mask vector that specifies
            which one to keep. Coordinates with False will be removed.

        Returns:
            A :attr:`MinkowskiEngine.SparseTensor` with C = coordinates
            corresponding to `mask == True` F = copy of the features from `mask ==
            True`.

        Example::

            >>> # Define inputs
            >>> input = SparseTensor(feats, coords=coords)
            >>> # Any boolean tensor can be used as the filter
            >>> mask = torch.rand(feats.size(0)) < 0.5
            >>> pruning = MinkowskiPruning()
            >>> output = pruning(input, mask)

        """
        assert isinstance(input, SparseTensor)
        out_coords_key = CoordsKey(input.coords_key.D)
        output = self.pruning.apply(input.F, mask, input.coords_key, out_coords_key, input.coords_man)
        return SparseTensor(output, coords_key=out_coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinkowskiUnionFunction(Function):

    @staticmethod
    def forward(ctx, in_coords_keys, out_coords_key, coords_manager, *in_feats):
        assert isinstance(in_feats, list) or isinstance(in_feats, tuple), 'Input must be a list or a set of Tensors'
        assert len(in_feats) > 1, 'input must be a set with at least 2 Tensors'
        in_feats = [in_feat.contiguous() for in_feat in in_feats]
        ctx.in_coords_keys = in_coords_keys
        ctx.out_coords_key = out_coords_key
        ctx.coords_manager = coords_manager
        fw_fn = get_minkowski_function('UnionForward', in_feats[0])
        return fw_fn(in_feats, [key.CPPCoordsKey for key in ctx.in_coords_keys], ctx.out_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager)

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()
        bw_fn = get_minkowski_function('UnionBackward', grad_out_feat)
        grad_in_feats = bw_fn(grad_out_feat, [key.CPPCoordsKey for key in ctx.in_coords_keys], ctx.out_coords_key.CPPCoordsKey, ctx.coords_manager.CPPCoordsManager)
        return None, None, None, *grad_in_feats


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
            >>>     coords_manager=input1.coords_man,  # Must use same coords manager
            >>>     force_creation=True  # The tensor stride [1, 1] already exists.
            >>> )
            >>> union = MinkowskiUnion()
            >>> output = union(input1, iput2)

        """
        for s in inputs:
            assert isinstance(s, SparseTensor), 'Inputs must be sparse tensors.'
        assert len(inputs) > 1, 'input must be a set with at least 2 SparseTensors'
        out_coords_key = CoordsKey(inputs[0].coords_key.D)
        output = self.union.apply([input.coords_key for input in inputs], out_coords_key, inputs[0].coords_man, *[input.F for input in inputs])
        return SparseTensor(output, coords_key=out_coords_key, coords_manager=inputs[0].coords_man)

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
        self.dec_block_s64s32 = nn.Sequential(ME.MinkowskiConvolutionTranspose(enc_ch[6], dec_ch[5], kernel_size=4, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(dec_ch[5]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[5]), ME.MinkowskiELU())
        self.dec_s32_cls = ME.MinkowskiConvolution(dec_ch[5], 1, kernel_size=1, has_bias=True, dimension=3)
        self.dec_block_s32s16 = nn.Sequential(ME.MinkowskiConvolutionTranspose(enc_ch[5], dec_ch[4], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(dec_ch[4]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[4]), ME.MinkowskiELU())
        self.dec_s16_cls = ME.MinkowskiConvolution(dec_ch[4], 1, kernel_size=1, has_bias=True, dimension=3)
        self.dec_block_s16s8 = nn.Sequential(ME.MinkowskiConvolutionTranspose(dec_ch[4], dec_ch[3], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(dec_ch[3]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[3]), ME.MinkowskiELU())
        self.dec_s8_cls = ME.MinkowskiConvolution(dec_ch[3], 1, kernel_size=1, has_bias=True, dimension=3)
        self.dec_block_s8s4 = nn.Sequential(ME.MinkowskiConvolutionTranspose(dec_ch[3], dec_ch[2], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(dec_ch[2]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[2]), ME.MinkowskiELU())
        self.dec_s4_cls = ME.MinkowskiConvolution(dec_ch[2], 1, kernel_size=1, has_bias=True, dimension=3)
        self.dec_block_s4s2 = nn.Sequential(ME.MinkowskiConvolutionTranspose(dec_ch[2], dec_ch[1], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(dec_ch[1]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[1]), ME.MinkowskiELU())
        self.dec_s2_cls = ME.MinkowskiConvolution(dec_ch[1], 1, kernel_size=1, has_bias=True, dimension=3)
        self.dec_block_s2s1 = nn.Sequential(ME.MinkowskiConvolutionTranspose(dec_ch[1], dec_ch[0], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(dec_ch[0]), ME.MinkowskiELU(), ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(dec_ch[0]), ME.MinkowskiELU())
        self.dec_s1_cls = ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, has_bias=True, dimension=3)
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool)
            cm = out.coords_man
            strided_target_key = cm.stride(target_key, out.tensor_stride[0], force_creation=True)
            ins, outs = cm.get_kernel_map(out.coords_key, strided_target_key, kernel_size=kernel_size, region_type=1)
            for curr_in in ins:
                target[curr_in] = 1
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
        keep_s32 = (dec_s32_cls.F > 0).cpu().squeeze()
        target = self.get_target(dec_s32, target_key)
        targets.append(target)
        out_cls.append(dec_s32_cls)
        if self.training:
            keep_s32 += target
        dec_s32 = self.pruning(dec_s32, keep_s32.cpu())
        dec_s16 = self.dec_block_s32s16(dec_s32)
        dec_s16 = dec_s16 + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).cpu().squeeze()
        target = self.get_target(dec_s16, target_key)
        targets.append(target)
        out_cls.append(dec_s16_cls)
        if self.training:
            keep_s16 += target
        dec_s16 = self.pruning(dec_s16, keep_s16.cpu())
        dec_s8 = self.dec_block_s16s8(dec_s16)
        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)
        target = self.get_target(dec_s8, target_key)
        targets.append(target)
        out_cls.append(dec_s8_cls)
        keep_s8 = (dec_s8_cls.F > 0).cpu().squeeze()
        if self.training:
            keep_s8 += target
        dec_s8 = self.pruning(dec_s8, keep_s8.cpu())
        dec_s4 = self.dec_block_s8s4(dec_s8)
        dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)
        target = self.get_target(dec_s4, target_key)
        targets.append(target)
        out_cls.append(dec_s4_cls)
        keep_s4 = (dec_s4_cls.F > 0).cpu().squeeze()
        if self.training:
            keep_s4 += target
        dec_s4 = self.pruning(dec_s4, keep_s4.cpu())
        dec_s2 = self.dec_block_s4s2(dec_s4)
        dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)
        target = self.get_target(dec_s2, target_key)
        targets.append(target)
        out_cls.append(dec_s2_cls)
        keep_s2 = (dec_s2_cls.F > 0).cpu().squeeze()
        if self.training:
            keep_s2 += target
        dec_s2 = self.pruning(dec_s2, keep_s2.cpu())
        dec_s1 = self.dec_block_s2s1(dec_s2)
        dec_s1_cls = self.dec_s1_cls(dec_s1)
        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)
        target = self.get_target(dec_s1, target_key)
        targets.append(target)
        out_cls.append(dec_s1_cls)
        keep_s1 = (dec_s1_cls.F > 0).cpu().squeeze()
        dec_s1 = self.pruning(dec_s1, keep_s1.cpu())
        return out_cls, targets, dec_s1


class STN3d(nn.Module):
    """Given a sparse tensor, generate a 3x3 transformation matrix per
    instance.
    """
    CONV_CHANNELS = [64, 128, 1024, 512, 256]
    FC_CHANNELS = [512, 256]
    KERNEL_SIZES = [1, 1, 1]
    STRIDES = [1, 1, 1]

    def __init__(self, D=3):
        super(STN3d, self).__init__()
        k = self.KERNEL_SIZES
        s = self.STRIDES
        c = self.CONV_CHANNELS
        self.block1 = nn.Sequential(ME.MinkowskiConvolution(3, c[0], kernel_size=k[0], stride=s[0], has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[0]), ME.MinkowskiReLU())
        self.block2 = nn.Sequential(ME.MinkowskiConvolution(c[0], c[1], kernel_size=k[1], stride=s[1], has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[1]), ME.MinkowskiReLU())
        self.block3 = nn.Sequential(ME.MinkowskiConvolution(c[1], c[2], kernel_size=k[2], stride=s[2], has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[2]), ME.MinkowskiReLU())
        self.block4 = nn.Sequential(ME.MinkowskiConvolution(c[2], c[3], kernel_size=1, has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[3]), ME.MinkowskiReLU())
        self.block5 = nn.Sequential(ME.MinkowskiConvolution(c[3], c[4], kernel_size=1, has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[4]), ME.MinkowskiReLU())
        self.fc6 = ME.MinkowskiConvolution(c[4], 9, kernel_size=1, has_bias=True, dimension=3)
        self.avgpool = ME.MinkowskiGlobalPooling()
        self.broadcast = ME.MinkowskiBroadcast()

    def forward(self, in_x):
        x = self.block1(in_x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc6(x)
        x._F += torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=x.dtype, device=x.device).repeat(len(x), 1)
        return self.broadcast(in_x, x)


class PointNetFeature(nn.Module):
    """
    You can think of a PointNet as a specialization of a convolutional neural
    network with kernel_size == 1, and stride == 1 that processes a sparse
    tensor where features are normalized coordinates.

    This generalization allows the network to process an arbitrary number of
    points.
    """
    CONV_CHANNELS = [256, 512, 1024]
    KERNEL_SIZES = [1, 1, 1]
    STRIDES = [1, 1, 1]

    def __init__(self):
        super(PointNetFeature, self).__init__()
        k = self.KERNEL_SIZES
        s = self.STRIDES
        c = self.CONV_CHANNELS
        self.stn = STN3d(D=3)
        self.block1 = nn.Sequential(ME.MinkowskiConvolution(6, c[0], kernel_size=k[0], stride=s[0], has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[0]), ME.MinkowskiReLU())
        self.block2 = nn.Sequential(ME.MinkowskiConvolution(c[0], c[1], kernel_size=k[1], stride=s[1], has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[1]), ME.MinkowskiReLU())
        self.block3 = nn.Sequential(ME.MinkowskiConvolution(c[1], c[2], kernel_size=k[2], stride=s[2], has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[2]), ME.MinkowskiReLU())
        self.avgpool = ME.MinkowskiGlobalPooling()
        self.concat = ME.MinkowskiBroadcastConcatenation()

    def forward(self, x):
        """
        Input is a spare tensor with features as centered coordinates N x 3
        """
        assert isinstance(x, ME.SparseTensor)
        assert x.F.shape[1] == 3
        T = self.stn(x)
        coords_feat_stn = torch.squeeze(torch.bmm(x.F.view(-1, 1, 3), T.F.view(-1, 3, 3)))
        x = ME.SparseTensor(torch.cat((coords_feat_stn, x.F), 1), coords_key=x.coords_key, coords_manager=x.coords_man)
        point_feat = self.block1(x)
        x = self.block2(point_feat)
        x = self.block3(x)
        glob_feat = self.avgpool(x)
        return self.concat(point_feat, glob_feat)


class PointNet(nn.Module):
    """
    You can think of a PointNet as a specialization of a convolutional neural
    network with kernel_size == 1, and stride == 1 that processes a sparse
    tensor where features are normalized coordinates.

    This generalization allows the network to process an arbitrary number of
    points.
    """
    CONV_CHANNELS = [512, 256, 128]
    KERNEL_SIZES = [1, 1, 1]
    STRIDES = [1, 1, 1]

    def __init__(self, out_channels, D=3):
        super(PointNet, self).__init__()
        k = self.KERNEL_SIZES
        s = self.STRIDES
        c = self.CONV_CHANNELS
        self.feat = PointNetFeature()
        self.block1 = nn.Sequential(ME.MinkowskiConvolution(1280, c[0], kernel_size=k[0], stride=s[0], has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[0]), ME.MinkowskiReLU())
        self.block2 = nn.Sequential(ME.MinkowskiConvolution(c[0], c[1], kernel_size=k[1], stride=s[1], has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[1]), ME.MinkowskiReLU())
        self.block3 = nn.Sequential(ME.MinkowskiConvolution(c[1], c[2], kernel_size=k[2], stride=s[2], has_bias=False, dimension=3), ME.MinkowskiInstanceNorm(c[2]), ME.MinkowskiReLU())
        self.conv4 = ME.MinkowskiConvolution(c[2], out_channels, kernel_size=1, has_bias=True, dimension=3)

    def forward(self, x):
        """
        Assume that x.F (features) are normalized coordinates or centered coordinates
        """
        assert isinstance(x, ME.SparseTensor)
        x = self.feat(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.conv4(x)


class GenerativeNet(nn.Module):
    CHANNELS = [1024, 512, 256, 128, 64, 32, 16]

    def __init__(self, resolution, in_nchannel=512):
        nn.Module.__init__(self)
        self.resolution = resolution
        ch = self.CHANNELS
        self.block1 = nn.Sequential(ME.MinkowskiConvolutionTranspose(in_nchannel, ch[0], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU(), ME.MinkowskiConvolutionTranspose(ch[0], ch[1], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU())
        self.block1_cls = ME.MinkowskiConvolution(ch[1], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block2 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[1], ch[2], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU())
        self.block2_cls = ME.MinkowskiConvolution(ch[2], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block3 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[2], ch[3], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU())
        self.block3_cls = ME.MinkowskiConvolution(ch[3], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block4 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[3], ch[4], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU())
        self.block4_cls = ME.MinkowskiConvolution(ch[4], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block5 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[4], ch[5], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU())
        self.block5_cls = ME.MinkowskiConvolution(ch[5], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block6 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[5], ch[6], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU())
        self.block6_cls = ME.MinkowskiConvolution(ch[6], 1, kernel_size=1, has_bias=True, dimension=3)
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool)
            cm = out.coords_man
            strided_target_key = cm.stride(target_key, out.tensor_stride[0], force_creation=True)
            ins, outs = cm.get_kernel_map(out.coords_key, strided_target_key, kernel_size=kernel_size, region_type=1)
            for curr_in in ins:
                target[curr_in] = 1
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
        keep1 = (out1_cls.F > 0).cpu().squeeze()
        if self.training:
            keep1 += target
        out1 = self.pruning(out1, keep1.cpu())
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).cpu().squeeze()
        if self.training:
            keep2 += target
        out2 = self.pruning(out2, keep2.cpu())
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).cpu().squeeze()
        if self.training:
            keep3 += target
        out3 = self.pruning(out3, keep3.cpu())
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).cpu().squeeze()
        if self.training:
            keep4 += target
        out4 = self.pruning(out4, keep4.cpu())
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        target = self.get_target(out5, target_key)
        targets.append(target)
        out_cls.append(out5_cls)
        keep5 = (out5_cls.F > 0).cpu().squeeze()
        if self.training:
            keep5 += target
        out5 = self.pruning(out5, keep5.cpu())
        out6 = self.block6(out5)
        out6_cls = self.block6_cls(out6)
        target = self.get_target(out6, target_key)
        targets.append(target)
        out_cls.append(out6_cls)
        keep6 = (out6_cls.F > 0).cpu().squeeze()
        out6 = self.pruning(out6, keep6.cpu())
        return out_cls, targets, out6


class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = 64, 128, 256, 512

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None
        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=D)
        self.layer1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)
        self.conv5 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(self.inplanes)
        self.glob_avg = ME.MinkowskiGlobalMaxPooling()
        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(ME.MinkowskiConvolution(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, dimension=self.D), ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, dimension=self.D))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.glob_avg(x)
        return self.final(x)


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = 1, 1, 1, 1


class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = 2, 2, 2, 2


class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = 3, 4, 6, 3


class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = 3, 4, 6, 3


class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = 3, 4, 23, 3


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
        self.block1 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[0], ch[0], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[0]), ME.MinkowskiELU(), ME.MinkowskiConvolutionTranspose(ch[0], ch[1], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[1]), ME.MinkowskiELU())
        self.block1_cls = ME.MinkowskiConvolution(ch[1], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block2 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[1], ch[2], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[2]), ME.MinkowskiELU())
        self.block2_cls = ME.MinkowskiConvolution(ch[2], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block3 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[2], ch[3], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[3]), ME.MinkowskiELU())
        self.block3_cls = ME.MinkowskiConvolution(ch[3], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block4 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[3], ch[4], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[4]), ME.MinkowskiELU())
        self.block4_cls = ME.MinkowskiConvolution(ch[4], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block5 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[4], ch[5], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[5]), ME.MinkowskiELU())
        self.block5_cls = ME.MinkowskiConvolution(ch[5], 1, kernel_size=1, has_bias=True, dimension=3)
        self.block6 = nn.Sequential(ME.MinkowskiConvolutionTranspose(ch[5], ch[6], kernel_size=2, stride=2, generate_new_coords=True, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU(), ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3), ME.MinkowskiBatchNorm(ch[6]), ME.MinkowskiELU())
        self.block6_cls = ME.MinkowskiConvolution(ch[6], 1, kernel_size=1, has_bias=True, dimension=3)
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool)
            cm = out.coords_man
            strided_target_key = cm.stride(target_key, out.tensor_stride[0], force_creation=True)
            ins, outs = cm.get_kernel_map(out.coords_key, strided_target_key, kernel_size=kernel_size, region_type=1)
            for curr_in in ins:
                target[curr_in] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z, target_key):
        out_cls, targets = [], []
        z.set_tensor_stride(self.resolution)
        out1 = self.block1(z)
        out1_cls = self.block1_cls(out1)
        target = self.get_target(out1, target_key)
        targets.append(target)
        out_cls.append(out1_cls)
        keep1 = (out1_cls.F > 0).cpu().squeeze()
        if self.training:
            keep1 += target
        out1 = self.pruning(out1, keep1.cpu())
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).cpu().squeeze()
        if self.training:
            keep2 += target
        out2 = self.pruning(out2, keep2.cpu())
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).cpu().squeeze()
        if self.training:
            keep3 += target
        out3 = self.pruning(out3, keep3.cpu())
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).cpu().squeeze()
        if self.training:
            keep4 += target
        out4 = self.pruning(out4, keep4.cpu())
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        target = self.get_target(out5, target_key)
        targets.append(target)
        out_cls.append(out5_cls)
        keep5 = (out5_cls.F > 0).cpu().squeeze()
        if self.training:
            keep5 += target
        out5 = self.pruning(out5, keep5.cpu())
        out6 = self.block6(out5)
        out6_cls = self.block6_cls(out6)
        target = self.get_target(out6, target_key)
        targets.append(target)
        out_cls.append(out6_cls)
        keep6 = (out6_cls.F > 0).cpu().squeeze()
        if keep6.sum() > 0:
            out6 = self.pruning(out6, keep6.cpu())
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
            zs += torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        out_cls, targets, sout = self.decoder(zs, gt_target)
        return out_cls, targets, sout, means, log_vars, zs

